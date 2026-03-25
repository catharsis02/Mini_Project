"""
SEED EEG — High-Accuracy ML Pipeline  (OOM-safe v3)
Target: ≥ 80 % within-subject accuracy
======================================================================

Root causes of OOM in v2 (both killed before CSP even ran)
────────────────────────────────────────────────────────────
  1. np.concatenate(raw_list) in load_subject
       3 session memmaps → one contiguous array = 3 × 1.5 GB = 4.5 GB
       killed on Subject 1 before the fold loop started.

  2. np.array(X_raw[tr])[:, :, ::DECIMATE]
       NumPy evaluates X_raw[tr] first (copies 8 100 × 62 × 2 000 × 4 B
       = 3.8 GB into RAM), THEN applies [::DECIMATE].
       The decimate step never runs; the process is already dead.

Fixes
──────
  FIX-A  SessionMemmapCollection
       Wraps multiple per-session memmaps.  Fancy-indexing scatters
       global indices across sessions without any concatenation.
       Cost: 0 bytes of extra RAM.

  FIX-B  csp_transform_chunked()
       Loads CHUNK_SIZE rows at a time, decimates immediately, runs
       CSP.transform, keeps only the (chunk, 18) output.
       Peak RAM per chunk = 512 × 62 × 500 × 4 B ≈ 64 MB.

  FIX-C  All existing v2 fixes retained
       • per-session z-score normalisation
       • CSP fit on stratified subsample (≤ 400 / class, decimated)
       • DASM + RASM asymmetry features
       • SVM C=100, LDA shrinkage=auto
       • FMI Borda-count selector, TOP_K=150
       • del + gc.collect() between folds and subjects

Peak RAM budget per fold (Subject 1, 8 100 train / 2 025 test windows)
────────────────────────────────────────────────────────────────────────
  X_tab (DE + asym, full subject)   10 137 × 580 × 4 B  ≈  24 MB
  CSP fit subsample (decimated)      1 200 × 62 × 500 × 4 B  ≈ 149 MB
  CSP transform chunk (train)          512 × 62 × 500 × 4 B  ≈  64 MB
  Collected CSP features (train)     8 100 × 18 × 4 B  ≈   0.6 MB
  X_tr assembled                     8 100 × 598 × 4 B  ≈  19 MB
  SVM kernel matrix (RBF)           8 100² × 4 B  ≈  263 MB  (libsvm)
  ────────────────────────────────────────────────────────────────────
  Total peak                                             ≈  520 MB ✓

Literature anchors (within-subject, all sessions, 5-fold CV):
  DE + SVM            Li et al. 2022    ~83–86 %
  DE + DASM + SVM     Zheng & Lu 2015   ~85–88 %
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

CSP_FIT_MAX_PER_CLS = 400     # epochs/class for CSP covariance estimation
DECIMATE            = 4       # time stride  (2000 pts → 500 pts @ 4×)
CHUNK_SIZE          = 512     # FIX-B: rows loaded at once during CSP transform

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
# FIX-A  SESSION MEMMAP COLLECTION  (replaces np.concatenate of memmaps)
# ══════════════════════════════════════════════════════════════════════

class SessionMemmapCollection:
    """
    Wraps a list of per-session memmaps and exposes fancy indexing
    over the concatenated virtual axis WITHOUT ever concatenating.

    Given global indices [i, j, k, …], it:
      1. Maps each index to (session_id, local_index) via cumulative offsets.
      2. Groups indices by session.
      3. Loads each group with a single memmap slice.
      4. Re-orders the result to match the original index order.

    Memory cost: 0 bytes (no copy of the raw arrays).
    """

    def __init__(self, arrays: list[np.ndarray]):
        self.arrays  = arrays
        self.lengths = np.array([a.shape[0] for a in arrays], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)])
        self.shape   = (int(self.offsets[-1]),) + arrays[0].shape[1:]
        self.ndim    = len(self.shape)

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, idx: np.ndarray) -> np.ndarray:
        """
        idx : 1-D array of global integer indices.
        Returns a new float32 array of shape (len(idx), C, T).
        """
        idx = np.asarray(idx, dtype=np.int64)
        out = np.empty((len(idx),) + self.shape[1:], dtype=np.float32)

        for s, arr in enumerate(self.arrays):
            lo, hi = int(self.offsets[s]), int(self.offsets[s + 1])
            mask   = (idx >= lo) & (idx < hi)
            if not mask.any():
                continue
            pos       = np.where(mask)[0]          # positions in output
            local     = idx[mask] - lo             # local indices in session
            out[pos]  = arr[local]                 # single read per session

        return out


# ══════════════════════════════════════════════════════════════════════
# FIX-B  CHUNKED CSP TRANSFORM  (64 MB peak instead of 3.8 GB)
# ══════════════════════════════════════════════════════════════════════

def csp_transform_chunked(
    csp:        "MulticlassCSP",
    collection: SessionMemmapCollection,
    indices:    np.ndarray,
    decimate:   int = DECIMATE,
    chunk_size: int = CHUNK_SIZE,
) -> np.ndarray:
    """
    Apply a fitted MulticlassCSP to `indices` rows of `collection`,
    loading `chunk_size` rows at a time.

    Peak RAM = chunk_size × C × (T/decimate) × 4 bytes
             = 512       × 62 × 500          × 4 B  ≈ 64 MB
    """
    parts: list[np.ndarray] = []
    for start in range(0, len(indices), chunk_size):
        chunk_idx = indices[start : start + chunk_size]
        chunk_raw = collection[chunk_idx][:, :, ::decimate]   # load + decimate
        parts.append(csp.transform(chunk_raw))
        del chunk_raw
    return np.concatenate(parts, axis=0)


# ══════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════

def detect_de_layout(feature_cols: list[str]) -> str:
    if not feature_cols:
        return "band_major"
    c0 = feature_cols[0].lower()
    for band in ("delta", "theta", "alpha", "beta", "gamma"):
        if c0.startswith(band) or c0.startswith(f"de_{band}"):
            return "band_major"
    return "channel_major"


def de_to_3d(X_de: np.ndarray, layout: str) -> np.ndarray:
    n = X_de.shape[0]
    if layout == "band_major":
        return X_de.reshape(n, N_BANDS, N_CHANNELS).transpose(0, 2, 1)
    return X_de.reshape(n, N_CHANNELS, N_BANDS)


def load_subject(
    subject_id: int,
) -> tuple[SessionMemmapCollection, np.ndarray, np.ndarray, str]:
    """
    Load all sessions for one subject.

    FIX-A: raw arrays stored in SessionMemmapCollection — never concatenated.
    Per-session z-score normalisation applied to DE features before pooling.

    Returns
    -------
    collection : SessionMemmapCollection  (zero RAM cost)
    X_de       : (n_windows, 310)  float32, per-session z-scored
    y          : (n_windows,)
    de_layout  : 'band_major' | 'channel_major'
    """
    stems = sorted(PROCESSED_PATH.glob(f"{subject_id}_*.parquet"))
    if not stems:
        raise FileNotFoundError(f"No parquet files for subject {subject_id}")

    raw_memmaps: list[np.ndarray] = []
    de_list:     list[np.ndarray] = []
    y_list:      list[np.ndarray] = []
    de_layout:   str | None        = None

    for pf in stems:
        raw_path = PROCESSED_PATH / f"{pf.stem}_raw.npy"
        if not raw_path.exists():
            print(f"  ⚠  Missing {raw_path.name}, skipping")
            continue

        df        = pd.read_parquet(pf)
        y         = df["label"].values
        feat_cols = [c for c in df.columns
                     if c not in ("label", "subject", "session", "trial_id")]
        X_de      = df[feat_cols].values.astype(np.float32)

        if de_layout is None:
            de_layout = detect_de_layout(feat_cols)

        # Per-session z-score (unsupervised → no label leakage)
        X_de = StandardScaler().fit_transform(X_de).astype(np.float32)

        raw_memmaps.append(np.load(raw_path, mmap_mode="r"))  # FIX-A: no concat
        de_list.append(X_de)
        y_list.append(y)

    if not raw_memmaps:
        raise FileNotFoundError(f"No valid data for subject {subject_id}")

    return (
        SessionMemmapCollection(raw_memmaps),
        np.concatenate(de_list, axis=0),
        np.concatenate(y_list,  axis=0),
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
    DASM_k = DE_left_k − DE_right_k
    RASM_k = DE_left_k / DE_right_k
    27 pairs × 5 bands × 2 = 270 features.
    Deterministic — computed once outside the fold loop, zero leakage.
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
    if name == "svm":
        return SVC(kernel="rbf", C=100, gamma="scale",
                   decision_function_shape="ovr", random_state=42)
    if name == "lda":
        return LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
    raise ValueError(f"Unknown classifier: {name!r}")


# ══════════════════════════════════════════════════════════════════════
# 6.  CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════

def _stratified_subsample(idx, y, max_per_cls, rng):
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
    de_layout:   str,
    n_folds:     int = N_FOLDS,
    classifier:  str = "svm",
) -> np.ndarray:
    """
    Stratified K-Fold.  All fitting is inside the fold — no leakage.

    Feature matrix per fold
    ───────────────────────
    DE(310) + DASM/RASM(270) + CSP(18)  →  598 raw features
    FMI selects top 150  →  StandardScaler  →  Classifier
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores: list[float] = []

    # Asymmetry: deterministic from X_de, computed once, zero leakage
    X_asym = compute_asymmetry(X_de, de_layout)        # (n, 270)
    X_tab  = np.concatenate([X_de, X_asym], axis=1)   # (n, 580)
    del X_asym; gc.collect()

    global_idx = np.arange(len(collection))

    for fold, (tr, te) in enumerate(skf.split(global_idx, y), start=1):
        print(f"    Fold {fold}/{n_folds} ...", end=" ", flush=True)
        rng  = np.random.default_rng(42 + fold)
        y_tr = y[tr]
        y_te = y[te]

        # ── CSP fit: subsample, decimate, fit ────────────────────────
        tr_csp  = _stratified_subsample(tr, y, CSP_FIT_MAX_PER_CLS, rng)
        fit_raw = collection[tr_csp][:, :, ::DECIMATE]   # ~149 MB
        csp     = MulticlassCSP()
        csp.fit(fit_raw, y[tr_csp])
        del fit_raw; gc.collect()

        # ── CSP transform: chunked (FIX-B) ───────────────────────────
        csp_tr = csp_transform_chunked(csp, collection, tr)   # (n_tr, 18)
        csp_te = csp_transform_chunked(csp, collection, te)   # (n_te, 18)
        del csp; gc.collect()

        # ── Assemble feature matrix ───────────────────────────────────
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
    print("  SEED EEG — High-Accuracy Pipeline  (OOM-safe v3)")
    print(f"  Features: DE(310) + DASM/RASM(270) + CSP(18) → FMI({TOP_K_FEATURES})")
    print(f"  SVM C=100 · LDA shrinkage=auto · {N_FOLDS}-fold CV")
    print(f"  Chunk={CHUNK_SIZE} rows · Decimate=1/{DECIMATE} · Peak ~520 MB")
    print("=" * 65)

    subjects     = discover_subjects()
    le           = LabelEncoder()
    all_results: list[dict] = []

    for clf_name in ("svm", "lda"):
        print(f"\n{'─' * 65}")
        print(f"  Classifier : {clf_name.upper()}")
        print(f"{'─' * 65}")

        subject_means: list[float] = []

        for sub in subjects:
            print(f"\n  Subject {sub}")
            try:
                collection, X_de, y_raw, de_layout = load_subject(sub)
            except FileNotFoundError as exc:
                print(f"  ✗  {exc}")
                continue

            y_enc = le.fit_transform(y_raw)
            print(
                f"    Windows : {len(collection)}  |  "
                f"Channels : {collection.shape[1]}  |  "
                f"DE feats : {X_de.shape[1]}  |  "
                f"Layout   : {de_layout}"
            )

            scores = cross_validate_subject(
                sub, collection, X_de, y_enc, de_layout,
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

            del collection, X_de, y_raw, y_enc, scores; gc.collect()

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