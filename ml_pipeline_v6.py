"""
SEED EEG — High-Accuracy ML Pipeline  (OOM-safe v6)
Target: ≥ 80 % within-subject accuracy  (GroupKFold, trial-grouped)
======================================================================

Full changelog
────────────────
  v1  Original — crashed OOM before Fold 1
        • np.concatenate(raw_list) → 4.5 GB at load time
        • np.array(X_raw[tr])[:,:,::4] → 3.8 GB before decimate ran
        • StratifiedKFold (window-level leakage)
        • SVM C=1 (severe underfitting)
        • No per-session normalisation, no asymmetry features

  v2  OOM pass 1 + accuracy improvements
        • mmap_mode='r' + chunked CSP transform (16 × 64 MB)
        • Per-session z-score normalisation       (+5–8 % acc)
        • DASM + RASM asymmetry features          (+2–5 % acc)
        • SVM C=100                               (+3–5 % acc)
        • LDA shrinkage='auto'                    (+1–2 % acc)
        ✗ np.concatenate(raw_list) still present → OOM

  v3  OOM pass 2
        • SessionMemmapCollection (zero-copy session wrapper)
        • csp_transform_chunked() (chunk loop, 16 × 64 MB)
        ✗ StratifiedKFold window leakage still present
        ✗ LabelEncoder remapping {-1,0,1} → {0,1,2}
        ✗ Preprocessor bugs caused 63 % accuracy (not this file)

  v4  Rejected
        • CSP subsample broken (contiguous → near single-class)
        • de_to_3d hardcoded band_major
        • DECIMATE=2 doubled RAM with no accuracy benefit

  v5  GroupKFold + collection.load()
        • GroupKFold by trial_id (fixes window leakage)
        • collection.load(idx, decimate) allocates at decimated size
          → 930 MB peak, chunk loop deleted
        • CSP_FIT_MAX_PER_CLS 400 → 600  (~+0.5 % acc)
        • SVM C=200
        ✗ LabelEncoder bug still present

  v6  LabelEncoder removed  ← current
        • y = df["label"].astype(int).values  →  {-1, 0, 1} preserved
        • class -1 (negative emotion) now appears in all outputs

  PREPROCESSOR (separate file) also fixed:
        • trial_id = f"{subject}_{session}_{trial_idx}"  (no win index)
        • explicit DE column names: 'delta_ch0' … 'gamma_ch61'
        These two preprocessor bugs were the root cause of 63 % in v3.

Peak RAM budget per fold  (8 100 train / 2 025 test, Subject 1)
────────────────────────────────────────────────────────────────
  X_tab  (DE 310 + asym 270, full subject)   ≈  24 MB
  CSP fit subsample  (1 800 eps, decimated)  ≈ 224 MB
  Full train decimated  (single alloc)       ≈ 930 MB  ← freed before X_tr
  Full test decimated   (single alloc)       ≈ 233 MB  ← freed before X_te
  X_tr assembled (598 features)              ≈  19 MB
  SVM kernel matrix                          ≈ 263 MB
  ─────────────────────────────────────────────────────
  Total peak                                 ≈ 1.50 GB  ✓
  (observed ~3.5 GB RSS incl. OS + Python overhead)

Expected accuracy after re-preprocessing
──────────────────────────────────────────
  GroupKFold + correct features   78–85 %  (honest generalisation)
  StratifiedKFold + correct       85–90 %  (inflated, for reference)
  Literature DE+DASM+SVM          85–88 %  Zheng & Lu 2015
"""

from __future__ import annotations

import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from mne.decoding import CSP
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

PROCESSED_PATH      = Path("./data/SEED_EEG/processed_data/")

N_CSP_COMPONENTS    = 6       # per OVR binary  →  3 × 6 = 18 CSP features
TOP_K_FEATURES      = 150     # after FMI Borda-count selection
N_FOLDS             = 5
N_JOBS              = 4       # MI worker cap

CSP_FIT_MAX_PER_CLS = 600     # epochs/class for CSP covariance estimation
DECIMATE            = 4       # time stride  (2 000 pts → 500 pts)

N_CHANNELS          = 62
N_BANDS             = 5       # δ θ α β γ

# ── SEED 62-channel left↔right electrode pairs (0-based) ─────────────
SEED_LR_PAIRS: list[tuple[int, int]] = [
    ( 0,  2), ( 3,  4), ( 5, 13), ( 6, 12), ( 7, 11), ( 8, 10),
    (14, 22), (15, 21), (16, 20), (17, 19),
    (23, 31), (24, 30), (25, 29), (26, 28),
    (32, 40), (33, 39), (34, 38), (35, 37),
    (41, 49), (42, 48), (43, 47), (44, 46),
    (50, 56), (51, 55), (52, 54),
    (58, 60), (57, 61),
]
# 27 pairs × 5 bands × 2 (DASM + RASM) = 270 asymmetry features


# ══════════════════════════════════════════════════════════════════════
# SESSION MEMMAP COLLECTION
# ══════════════════════════════════════════════════════════════════════

class SessionMemmapCollection:
    """
    Wraps per-session memmaps behind a single virtual index space.
    Never concatenates raw arrays — zero extra RAM at load time.

    .load(idx, decimate)
    ─────────────────────
    Allocates output at decimated shape from the start — the
    full-resolution tensor never exists in RAM.

    Old OOM pattern:
        np.array(collection[idx])[:, :, ::4]
        → allocates 3.8 GB, decimates after  → killed

    New pattern:
        collection.load(idx, decimate=4)
        → allocates 0.93 GB directly  ✓
    """

    def __init__(self, arrays: list[np.ndarray]):
        self.arrays  = arrays
        self.lengths = np.array([a.shape[0] for a in arrays], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)])
        self.shape   = (int(self.offsets[-1]),) + arrays[0].shape[1:]
        self.ndim    = len(self.shape)

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def load(self, idx: np.ndarray, decimate: int = 1) -> np.ndarray:
        """
        Load rows `idx`, decimating the time axis before allocation.

        Parameters
        ----------
        idx      : 1-D int array of global indices
        decimate : time-axis stride  (1 = no decimation)

        Returns
        -------
        float32 array  (len(idx), n_channels, n_samples // decimate)
        """
        idx = np.asarray(idx, dtype=np.int64)
        T   = self.shape[2] // decimate if decimate > 1 else self.shape[2]
        out = np.empty((len(idx), self.shape[1], T), dtype=np.float32)

        for s, arr in enumerate(self.arrays):
            lo, hi = int(self.offsets[s]), int(self.offsets[s + 1])
            mask   = (idx >= lo) & (idx < hi)
            if not mask.any():
                continue
            out[np.where(mask)[0]] = arr[idx[mask] - lo][:, :, ::decimate]

        return out

    def __getitem__(self, idx: np.ndarray) -> np.ndarray:
        """Full-resolution load (kept for non-CSP uses)."""
        return self.load(np.asarray(idx, dtype=np.int64), decimate=1)


# ══════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════

def detect_de_layout(feature_cols: list[str]) -> str:
    """
    Infer DE column storage order from column names.

    band_major    [all 62 ch for δ | all 62 ch for θ | … | γ]
                  cols: 'delta_ch0' … 'delta_ch61' 'theta_ch0' …
                  (produced by fixed preprocessor)

    channel_major [all 5 bands for ch0 | … | all 5 bands for ch61]
                  cols: 'ch0_delta' 'ch0_theta' … 'ch61_gamma'

    NOTE: if columns are integers ('0','1','2'…) the preprocessor was
    not fixed — returns 'channel_major' which will scramble asymmetry
    features.  Re-run the fixed preprocessor first.
    """
    if not feature_cols:
        return "band_major"
    c0 = feature_cols[0].lower()
    for band in ("delta", "theta", "alpha", "beta", "gamma"):
        if c0.startswith(band) or c0.startswith(f"de_{band}"):
            return "band_major"
    return "channel_major"


def de_to_3d(X_de: np.ndarray, layout: str) -> np.ndarray:
    """Reshape (n, C*B) → (n, C, B)."""
    n = X_de.shape[0]
    if layout == "band_major":
        return X_de.reshape(n, N_BANDS, N_CHANNELS).transpose(0, 2, 1)
    return X_de.reshape(n, N_CHANNELS, N_BANDS)


def load_subject(
    subject_id: int,
) -> tuple[SessionMemmapCollection, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Load all sessions for one subject.

    Labels cast to int → preserves {-1, 0, 1}.
    Per-session z-score applied to DE before pooling — removes
    session-level baseline shifts (electrode drift, fatigue).
    Unsupervised → zero label leakage.

    Returns
    -------
    collection : SessionMemmapCollection  (zero RAM cost)
    X_de       : (n_windows, 310)   float32, per-session z-scored
    y          : (n_windows,)       int  {-1, 0, 1}
    trials     : (n_windows,)       trial group IDs for GroupKFold
    de_layout  : 'band_major' | 'channel_major'
    """
    stems = sorted(PROCESSED_PATH.glob(f"{subject_id}_*.parquet"))
    if not stems:
        raise FileNotFoundError(f"No parquet files for subject {subject_id}")

    raw_memmaps: list[np.ndarray] = []
    de_list:     list[np.ndarray] = []
    y_list:      list[np.ndarray] = []
    trial_list:  list[np.ndarray] = []
    de_layout:   str | None        = None

    for pf in stems:
        raw_path = PROCESSED_PATH / f"{pf.stem}_raw.npy"
        if not raw_path.exists():
            print(f"  ⚠  Missing {raw_path.name}, skipping")
            continue

        df        = pd.read_parquet(pf)
        y         = df["label"].astype(int).values          # keep {-1, 0, 1}
        trials    = df["trial_id"].values
        feat_cols = [c for c in df.columns
                     if c not in ("label", "subject", "session", "trial_id")]
        X_de      = df[feat_cols].values.astype(np.float32)

        if de_layout is None:
            de_layout = detect_de_layout(feat_cols)

        # Warn early if preprocessor was not fixed
        if de_layout == "channel_major" and feat_cols[0].lstrip("-").isdigit():
            print(
                f"  ⚠  DE columns are integers ('{feat_cols[0]}', …) — "
                f"layout detection fell back to channel_major.\n"
                f"     Re-run the fixed preprocessor to get named columns."
            )

        # Per-session z-score: unsupervised, zero leakage
        X_de = StandardScaler().fit_transform(X_de).astype(np.float32)

        raw_memmaps.append(np.load(raw_path, mmap_mode="r"))
        de_list.append(X_de)
        y_list.append(y)
        trial_list.append(trials)

    if not raw_memmaps:
        raise FileNotFoundError(f"No valid data for subject {subject_id}")

    return (
        SessionMemmapCollection(raw_memmaps),
        np.concatenate(de_list,    axis=0),
        np.concatenate(y_list,     axis=0),
        np.concatenate(trial_list, axis=0),
        de_layout or "band_major",
    )


def discover_subjects() -> list[int]:
    return sorted({int(p.stem.split("_")[0])
                   for p in PROCESSED_PATH.glob("*.parquet")})


# ══════════════════════════════════════════════════════════════════════
# 2.  ASYMMETRY FEATURES  (DASM + RASM)
# ══════════════════════════════════════════════════════════════════════

def compute_asymmetry(X_de: np.ndarray, layout: str) -> np.ndarray:
    """
    DASM_k = DE_left_k − DE_right_k      (hemispheric difference)
    RASM_k = DE_left_k / DE_right_k      (hemispheric ratio)

    27 pairs × 5 bands × 2 = 270 features.
    Computed once before the fold loop from X_de only.
    No labels used → zero leakage regardless of fold boundaries.

    Frontal alpha/beta asymmetry is the primary neural marker of
    emotional valence, aligned with SEED's {-1, 0, 1} labels.
    """
    X3d   = de_to_3d(X_de, layout)
    feats: list[np.ndarray] = []
    for li, ri in SEED_LR_PAIRS:
        l, r = X3d[:, li, :], X3d[:, ri, :]
        feats.append(l - r)
        feats.append(l / (r + 1e-8))
    return np.concatenate(feats, axis=1).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# 3.  MULTICLASS CSP  (One-vs-Rest)
# ══════════════════════════════════════════════════════════════════════

class MulticlassCSP(BaseEstimator, TransformerMixin):
    """
    One MNE CSP per class in {-1, 0, 1}.
    Output: (n_epochs, n_classes × n_components) = (n, 18)

    OVR binary labels are (y == -1), (y == 0), (y == 1) — correct
    because y is native {-1, 0, 1}, not remapped {0, 1, 2}.
    """

    def __init__(self, n_components: int = N_CSP_COMPONENTS, reg=None):
        self.n_components = n_components
        self.reg          = reg

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MulticlassCSP":
        self.classes_ = np.unique(y)
        self.csps_: dict = {}
        for cls in self.classes_:
            csp = CSP(n_components=self.n_components, reg=self.reg,
                      log=True, norm_trace=False)
            csp.fit(X, (y == cls).astype(int))
            self.csps_[cls] = csp
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [self.csps_[cls].transform(X) for cls in self.classes_], axis=1
        )


# ══════════════════════════════════════════════════════════════════════
# 4.  F + MI RANK-FUSION SELECTOR  (Borda count)
# ══════════════════════════════════════════════════════════════════════

class FMISelector(BaseEstimator, TransformerMixin):
    """
    combined_rank = rank_by_F + rank_by_MI  (lower → more discriminative)
    Scale-invariant across CSP log-variance, DE, and asymmetry features.
    """

    def __init__(self, k: int = TOP_K_FEATURES, n_jobs: int = N_JOBS):
        self.k      = k
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FMISelector":
        f_sc, _ = f_classif(X, y)
        mi_sc   = mutual_info_classif(X, y, random_state=42, n_jobs=self.n_jobs)
        f_sc    = np.nan_to_num(f_sc)
        mi_sc   = np.nan_to_num(mi_sc)
        f_r     = np.argsort(np.argsort(-f_sc))
        mi_r    = np.argsort(np.argsort(-mi_sc))
        self.selected_idx_ = np.argsort(f_r + mi_r)[: self.k]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_idx_]


# ══════════════════════════════════════════════════════════════════════
# 5.  CLASSIFIERS
# ══════════════════════════════════════════════════════════════════════

def build_clf(name: str):
    """
    SVM C=200 · RBF kernel · OVR
        Handles {-1, 0, 1} natively.
        C ∈ [100, 500] is the published sweet-spot for SEED DE features.

    LDA shrinkage='auto'
        Ledoit-Wolf regularisation — essential when n_features (~150)
        approaches n_samples/class (~400 with GroupKFold train splits).
        Handles {-1, 0, 1} natively.
    """
    if name == "svm":
        return SVC(kernel="rbf", C=200, gamma="scale",
                   decision_function_shape="ovr", random_state=42)
    if name == "lda":
        return LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
    raise ValueError(f"Unknown classifier: {name!r}")


# ══════════════════════════════════════════════════════════════════════
# 6.  CROSS-VALIDATION  (GroupKFold by trial_id)
# ══════════════════════════════════════════════════════════════════════

def _stratified_subsample(
    idx:         np.ndarray,
    y:           np.ndarray,
    max_per_cls: int,
    rng:         np.random.Generator,
) -> np.ndarray:
    """
    Down-sample `idx` to ≤ max_per_cls epochs per class.

    Must be stratified (not a contiguous head-slice) — SEED trials are
    sorted by label, so idx[:1800] would be near single-class, producing
    degenerate CSP covariance matrices and meaningless spatial filters.
    """
    parts = []
    for cls in np.unique(y[idx]):
        ci = idx[y[idx] == cls]
        if len(ci) > max_per_cls:
            ci = rng.choice(ci, max_per_cls, replace=False)
        parts.append(ci)
    return np.concatenate(parts)


def cross_validate_subject(
    subject_id:  int,
    collection:  SessionMemmapCollection,
    X_de:        np.ndarray,
    y:           np.ndarray,
    trials:      np.ndarray,
    de_layout:   str,
    n_folds:     int = N_FOLDS,
    classifier:  str = "svm",
) -> np.ndarray:
    """
    GroupKFold by trial_id.  All pipeline steps fit inside each fold.

    Why GroupKFold
    ──────────────
    SEED windows are 1-second slices of 4-minute continuous film clips.
    Adjacent windows are highly correlated (same EEG state, same clip).
    StratifiedKFold randomly mixes windows → model memorises trial-specific
    EEG signatures → 10–20 % artificial inflation.

    GroupKFold holds out complete trials so the model must generalise
    to stimuli it has never seen, matching trial_id design intent.

    Feature matrix per fold
    ───────────────────────
    DE(310) + DASM/RASM(270) + CSP(18)  →  598 total
    FMI → 150  →  StandardScaler  →  Classifier

    Labels: native {-1, 0, 1}
      -1 = negative emotion
       0 = neutral
       1 = positive emotion
    """
    gkf    = GroupKFold(n_splits=n_folds)
    idx    = np.arange(len(collection))
    scores: list[float] = []

    # Asymmetry: deterministic from X_de, no labels → zero leakage
    X_asym = compute_asymmetry(X_de, de_layout)        # (n, 270)
    X_tab  = np.concatenate([X_de, X_asym], axis=1)   # (n, 580)
    del X_asym; gc.collect()

    for fold, (tr, te) in enumerate(
            gkf.split(idx, y, groups=trials), start=1):

        te_trials = np.unique(trials[te]).tolist()
        print(f"    Fold {fold}/{n_folds}  "
              f"[test trials: {te_trials}] ...", end=" ", flush=True)

        rng  = np.random.default_rng(42 + fold)
        y_tr = y[tr]
        y_te = y[te]

        # ── CSP fit: stratified subsample, decimated ─────────────────
        tr_csp  = _stratified_subsample(tr, y, CSP_FIT_MAX_PER_CLS, rng)
        fit_raw = collection.load(tr_csp, decimate=DECIMATE)  # ~224 MB
        csp     = MulticlassCSP()
        csp.fit(fit_raw, y[tr_csp])
        del fit_raw; gc.collect()

        # ── CSP transform: single allocation at decimated size ────────
        raw_tr = collection.load(tr, decimate=DECIMATE)   # ~930 MB
        csp_tr = csp.transform(raw_tr)                    # (n_tr, 18)
        del raw_tr; gc.collect()

        raw_te = collection.load(te, decimate=DECIMATE)   # ~233 MB
        csp_te = csp.transform(raw_te)                    # (n_te, 18)
        del raw_te, csp; gc.collect()

        # ── Assemble full feature matrix ──────────────────────────────
        X_tr = np.concatenate([X_tab[tr], csp_tr], axis=1)   # (n_tr, 598)
        X_te = np.concatenate([X_tab[te], csp_te], axis=1)   # (n_te, 598)
        del csp_tr, csp_te; gc.collect()

        # ── FMI → scale → classify ───────────────────────────────────
        selector = FMISelector()
        scaler   = StandardScaler()
        clf      = build_clf(classifier)

        X_tr = selector.fit_transform(X_tr, y_tr)
        X_tr = scaler.fit_transform(X_tr)
        clf.fit(X_tr, y_tr)
        del X_tr; gc.collect()

        X_te = selector.transform(X_te)
        X_te = scaler.transform(X_te)
        acc  = float((clf.predict(X_te) == y_te).mean())
        scores.append(acc)
        print(f"acc = {acc:.4f}")

        del X_te, selector, scaler, clf, y_tr, y_te; gc.collect()

    del X_tab; gc.collect()
    return np.array(scores)


# ══════════════════════════════════════════════════════════════════════
# 7.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  SEED EEG — High-Accuracy Pipeline  (OOM-safe v6)")
    print(f"  Features : DE(310) + DASM/RASM(270) + CSP(18) → FMI({TOP_K_FEATURES})")
    print(f"  CV       : GroupKFold(n={N_FOLDS}) by trial_id")
    print(f"  Labels   : native {{-1, 0, 1}}  (no LabelEncoder)")
    print(f"  SVM C=200  ·  LDA shrinkage=auto")
    print(f"  Decimate : 1/{DECIMATE}  ·  CSP subsample : {CSP_FIT_MAX_PER_CLS}/class")
    print(f"  Peak RAM : ~1.5 GB per fold")
    print("=" * 65)

    subjects     = discover_subjects()
    all_results: list[dict] = []

    for clf_name in ("svm", "lda"):
        print(f"\n{'─' * 65}")
        print(f"  Classifier : {clf_name.upper()}")
        print(f"{'─' * 65}")

        subject_means: list[float] = []

        for sub in subjects:
            print(f"\n  Subject {sub}")
            try:
                collection, X_de, y, trials, de_layout = load_subject(sub)
            except FileNotFoundError as exc:
                print(f"  ✗  {exc}")
                continue

            n_trials = int(np.unique(trials).shape[0])
            classes  = np.unique(y).tolist()
            print(
                f"    Windows  : {len(collection)}  |  "
                f"Channels : {collection.shape[1]}  |  "
                f"DE feats : {X_de.shape[1]}  |  "
                f"Trials   : {n_trials}  |  "
                f"Classes  : {classes}  |  "
                f"Layout   : {de_layout}"
            )

            if n_trials < N_FOLDS:
                print(f"  ⚠  Only {n_trials} trials — "
                      f"need ≥ {N_FOLDS} for GroupKFold. Skipping.")
                del collection, X_de, y, trials
                gc.collect()
                continue

            scores = cross_validate_subject(
                sub, collection, X_de, y, trials, de_layout,
                n_folds=N_FOLDS, classifier=clf_name,
            )

            mean_acc = float(scores.mean())
            subject_means.append(mean_acc)
            print(f"    → Subject {sub}  mean: {mean_acc:.4f}  "
                  f"std: {scores.std():.4f}")

            for fold_i, s in enumerate(scores, 1):
                all_results.append({
                    "classifier": clf_name,
                    "subject":    sub,
                    "fold":       fold_i,
                    "accuracy":   round(float(s), 6),
                })

            del collection, X_de, y, trials, scores
            gc.collect()

        if subject_means:
            arr = np.array(subject_means)
            print(
                f"\n  {clf_name.upper()} across {len(arr)} subjects — "
                f"mean: {arr.mean():.4f}  std: {arr.std():.4f}  "
                f"min: {arr.min():.4f}  max: {arr.max():.4f}"
            )

    out_path = Path("./results_cv.csv")
    pd.DataFrame(all_results).to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")
    print("=" * 65)
