"""
SEED EEG — High-Accuracy ML Pipeline  (OOM-safe v7, hardened)
Target: ≥ 78 % within-subject accuracy  (GroupKFold, trial-grouped)
======================================================================

Root cause of ~24 % accuracy in v6  (below 33 % chance level)
──────────────────────────────────────────────────────────────
  RASM = DE_left / (DE_right + 1e-8) was computed AFTER z-scoring.
  After StandardScaler, DE values have zero mean, so DE_right ≈ 0
  is common.  When DE_right (float32) ≈ -1e-8, the float64 denominator
  becomes ~6e-17, giving RASM = 1.6e+17.  StandardScaler then fails
  with "Input X contains infinity" or produces z-scores that completely
  dominate the SVM RBF kernel → predictions worse than random.

v7 fix: compute asymmetry from raw (positive) DE BEFORE z-scoring,
        then z-score [DE + DASM + RASM] together as one block.
        Extreme RASM values are clipped to ±500 and nan/inf replaced
        by 0 before z-scoring, making the pipeline robust to any edge
        case in the DE values.

Hardening additions (v7 vs v6)
──────────────────────────────────────────────────────────────────────
  HARD-1  _stratified_subsample
           np.concatenate([]) raises ValueError.  Added guard so an
           empty parts list raises a clear RuntimeError instead.

  HARD-2  _compute_asymmetry_raw
           After RASM computation, apply nan_to_num (nan→0, ±inf→±500)
           then clip to [-500, 500].  Prevents RASM edge cases (very
           small DE_right after float32 precision effects) from
           propagating as huge values through z-scoring.

  HARD-3  load_subject
           Assert X_de_raw.shape[1] == N_CHANNELS * N_BANDS.
           Gives a clear error if the preprocessor produced a wrong
           channel count instead of a cryptic reshape failure later.

  HARD-4  cross_validate_subject
           Assert len(collection) == len(X_tab) == len(y) == len(trials)
           before GroupKFold.  A row count mismatch (possible if the
           preprocessor was run with the unpatched bug) would otherwise
           crash sklearn with an opaque "inconsistent number of samples"
           error deep inside GroupKFold.split().

Full changelog
────────────────
  v1  Original — OOM before Fold 1
  v2  OOM pass 1 + accuracy features
  v3  OOM pass 2 — SessionMemmapCollection + chunk loop
  v4  Rejected — broken CSP subsample, hardcoded layout
  v5  GroupKFold + collection.load() — chunk loop deleted
  v6  LabelEncoder removed; RASM bug caused 24 % accuracy
  v7  RASM on raw DE + hardening  ← current

Peak RAM budget per fold (8 100 train / 2 025 test)
─────────────────────────────────────────────────────
  X_tab in RAM (580 features, full subject)   ≈  24 MB
  CSP fit subsample  (1 800 eps, decimated)   ≈ 224 MB
  Full train decimated  (single alloc)        ≈ 930 MB  freed before X_tr
  Full test decimated   (single alloc)        ≈ 233 MB  freed before X_te
  X_tr  (598 features)                        ≈  19 MB
  SVM kernel matrix                           ≈ 263 MB
  ────────────────────────────────────────────────────────
  Total peak                                  ≈ 1.50 GB  ✓

Expected accuracy (GroupKFold, correct preprocessing)
──────────────────────────────────────────────────────
  78–85 %  (honest trial-level generalisation)
  Literature: DE + DASM + SVM  ~85–88 %  Zheng & Lu 2015
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

CSP_FIT_MAX_PER_CLS = 600     # max epochs/class for CSP covariance fit
DECIMATE            = 4       # time stride  (2 000 pts → 500 pts)

N_CHANNELS          = 62
N_BANDS             = 5       # δ θ α β γ

# Largest sensible RASM value before z-scoring.
# Clips extreme RASM outliers caused by near-zero DE denominators
# (e.g. float32 precision giving denominator ≈ 6e-17 → RASM ≈ 1.6e17).
RASM_CLIP = 500.0

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

assert len(SEED_LR_PAIRS) == 27, "Expected 27 LR electrode pairs"
assert max(max(l, r) for l, r in SEED_LR_PAIRS) < N_CHANNELS, \
    "Electrode pair index exceeds N_CHANNELS"
assert TOP_K_FEATURES < N_CHANNELS * N_BANDS + 270 + N_CSP_COMPONENTS * 3, \
    "TOP_K_FEATURES must be less than total feature count"


# ══════════════════════════════════════════════════════════════════════
# SESSION MEMMAP COLLECTION
# ══════════════════════════════════════════════════════════════════════

class SessionMemmapCollection:
    """
    Wraps per-session memmaps behind a single virtual index space.
    Never concatenates the raw arrays — zero extra RAM at load time.

    .load(idx, decimate)
    ─────────────────────
    Allocates output at decimated shape before reading any data, so
    the full-resolution tensor never exists in RAM.

    Peak RAM = len(idx) × N_CHANNELS × (n_samples // decimate) × 4 B
    Train fold: 8100 × 62 × 500 × 4 B ≈ 930 MB (with DECIMATE=4)
    """

    def __init__(self, arrays: list[np.ndarray]):
        if not arrays:
            raise ValueError("SessionMemmapCollection requires at least one array")
        for i, a in enumerate(arrays):
            if a.ndim != 3:
                raise ValueError(f"Array {i} must be 3-D (windows, channels, samples), "
                                 f"got shape {a.shape}")
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
        if idx.ndim != 1:
            raise ValueError(f"idx must be 1-D, got shape {idx.shape}")
        if len(idx) == 0:
            return np.empty((0, self.shape[1], self.shape[2] // decimate),
                            dtype=np.float32)
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

    band_major    'delta_ch0' … 'gamma_ch61'  (fixed preprocessor)
    channel_major 'ch0_delta' … 'ch61_gamma'

    Falls back to 'channel_major' for integer column names
    (unfixed preprocessor).  The caller warns about this.
    """
    if not feature_cols:
        return "band_major"
    c0 = feature_cols[0].lower()
    for band in ("delta", "theta", "alpha", "beta", "gamma"):
        if c0.startswith(band) or c0.startswith(f"de_{band}"):
            return "band_major"
    return "channel_major"


def de_to_3d(X_de: np.ndarray, layout: str) -> np.ndarray:
    """
    Reshape (n, N_CHANNELS * N_BANDS) → (n, N_CHANNELS, N_BANDS).

    band_major   : reshape (n, N_BANDS, N_CHANNELS), transpose axes 1↔2
    channel_major: reshape (n, N_CHANNELS, N_BANDS) directly
    """
    n = X_de.shape[0]
    if layout == "band_major":
        return X_de.reshape(n, N_BANDS, N_CHANNELS).transpose(0, 2, 1)
    return X_de.reshape(n, N_CHANNELS, N_BANDS)


def _compute_asymmetry_raw(X_de_raw: np.ndarray, layout: str) -> np.ndarray:
    """
    Compute DASM and RASM from RAW (positive) DE values.

    DASM_k = DE_left_k  −  DE_right_k
    RASM_k = DE_left_k  /  DE_right_k

    Why raw DE, not z-scored DE?
    ────────────────────────────
    DE = 0.5 * log(2πe * σ²).  After the per-window channel
    normalisation in preprocessing, σ² can be as small as 1e-10,
    making DE as low as -10.09 (negative) and near-zero values common.

    After StandardScaler (z-scoring), DE values span [-3, +3] with
    zero mean, so DE_right ≈ 0 frequently.  Float32 representation of
    -1e-8 in float64 is -1.000e-8; adding 1e-8 gives a denominator of
    ~6e-17, making RASM ≈ 1.6e+17.  StandardScaler then crashes
    ("Input X contains infinity") or the SVM kernel is destroyed.

    On raw DE values (typically [−10, +10] range), RASM is bounded
    to [-1e10, +1e10] in the worst case, and after clipping to ±500
    and z-scoring it behaves well.

    Hardening (HARD-2)
    ──────────────────
    After computing RASM, apply nan_to_num (nan→0, ±inf→±RASM_CLIP)
    then clip to [-RASM_CLIP, +RASM_CLIP].  This handles:
      • NaN from nan DE inputs (shouldn't happen but guarded)
      • Inf from exact-zero denominators (shouldn't happen but guarded)
      • Extreme values from near-zero denominators (confirmed possible)

    Output: (n, 27 pairs × 5 bands × 2) = (n, 270) float32
    """
    X3d   = de_to_3d(X_de_raw, layout)      # (n, 62, 5)
    feats: list[np.ndarray] = []
    for li, ri in SEED_LR_PAIRS:
        l = X3d[:, li, :]                    # (n, 5)
        r = X3d[:, ri, :]                    # (n, 5)
        feats.append(l - r)                  # DASM
        rasm = l / (r + 1e-8)               # RASM — 1e-8 guard
        # HARD-2: replace any nan/inf produced by float32 precision effects
        rasm = np.nan_to_num(rasm, nan=0.0, posinf=RASM_CLIP, neginf=-RASM_CLIP)
        rasm = np.clip(rasm, -RASM_CLIP, RASM_CLIP)
        feats.append(rasm)

    X_asym = np.concatenate(feats, axis=1).astype(np.float32)

    # Final safety check — should never fire after nan_to_num + clip
    if not np.isfinite(X_asym).all():
        n_bad = (~np.isfinite(X_asym)).sum()
        print(f"  ⚠  {n_bad} non-finite asymmetry values after clipping — zeroing")
        X_asym = np.nan_to_num(X_asym, nan=0.0, posinf=0.0, neginf=0.0)

    return X_asym


def load_subject(
    subject_id: int,
) -> tuple[SessionMemmapCollection, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Load all sessions for one subject.

    Asymmetry (DASM + RASM) is computed per session from RAW DE values,
    then [DE + DASM + RASM] are z-scored together as one block.
    Per-session z-score removes session-level EEG baseline shifts.
    Unsupervised transform → zero label leakage.

    Returns
    -------
    collection : SessionMemmapCollection   zero RAM cost
    X_tab      : (n, 580)  float32  [DE(310) + DASM(135) + RASM(135)]
                 per-session z-scored, asymmetry from raw DE
    y          : (n,)  int  {-1, 0, 1}
    trials     : (n,)  str  trial group IDs for GroupKFold
    de_layout  : 'band_major' | 'channel_major'
    """
    stems = sorted(PROCESSED_PATH.glob(f"{subject_id}_*.parquet"))
    if not stems:
        raise FileNotFoundError(f"No parquet files for subject {subject_id}")

    raw_memmaps: list[np.ndarray] = []
    tab_list:    list[np.ndarray] = []
    y_list:      list[np.ndarray] = []
    trial_list:  list[np.ndarray] = []
    de_layout:   str | None        = None

    for pf in stems:
        raw_path = PROCESSED_PATH / f"{pf.stem}_raw.npy"
        if not raw_path.exists():
            print(f"  ⚠  Missing {raw_path.name}, skipping session")
            continue

        df        = pd.read_parquet(pf)
        y_sess    = df["label"].astype(int).values     # keep {-1, 0, 1}
        trials    = df["trial_id"].values
        feat_cols = [c for c in df.columns
                     if c not in ("label", "subject", "session", "trial_id")]
        X_de_raw  = df[feat_cols].values.astype(np.float32)

        # HARD-3: catch wrong channel count early with a clear message
        expected_de_features = N_CHANNELS * N_BANDS
        if X_de_raw.shape[1] != expected_de_features:
            raise ValueError(
                f"{pf.name}: expected {expected_de_features} DE features "
                f"({N_CHANNELS} channels × {N_BANDS} bands), "
                f"got {X_de_raw.shape[1]}.  "
                f"Re-run preprocessing with the corrected n_channels."
            )

        if de_layout is None:
            de_layout = detect_de_layout(feat_cols)

        if feat_cols and feat_cols[0].lstrip("-").isdigit():
            print(
                f"  ⚠  {pf.name}: DE columns are raw integers ('{feat_cols[0]}', …).\n"
                f"     Preprocessor was not re-run with the fix.\n"
                f"     RASM will be computed on correctly reshaped data only if\n"
                f"     your layout is actually band_major — verify manually."
            )

        # Compute asymmetry from raw (positive) DE — BEFORE z-scoring.
        # See _compute_asymmetry_raw docstring for why this order matters.
        X_asym = _compute_asymmetry_raw(X_de_raw, de_layout)   # (n, 270)

        # Stack, then z-score the whole [DE | DASM | RASM] block per session.
        X_combined = np.concatenate([X_de_raw, X_asym], axis=1)   # (n, 580)
        X_combined = StandardScaler().fit_transform(X_combined).astype(np.float32)

        # Final guard: z-scored output should always be finite
        if not np.isfinite(X_combined).all():
            n_bad = (~np.isfinite(X_combined)).sum()
            print(f"  ⚠  {n_bad} non-finite values after z-score in {pf.name} — zeroing")
            X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)

        raw_memmaps.append(np.load(raw_path, mmap_mode="r"))
        tab_list.append(X_combined)
        y_list.append(y_sess)
        trial_list.append(trials)

        del X_de_raw, X_asym, X_combined

    if not raw_memmaps:
        raise FileNotFoundError(f"No valid session data found for subject {subject_id}")

    return (
        SessionMemmapCollection(raw_memmaps),
        np.concatenate(tab_list,   axis=0),
        np.concatenate(y_list,     axis=0),
        np.concatenate(trial_list, axis=0),
        de_layout or "band_major",
    )


def discover_subjects() -> list[int]:
    """Return sorted list of subject IDs found in PROCESSED_PATH."""
    subjects = set()
    for p in PROCESSED_PATH.glob("*.parquet"):
        try:
            subjects.add(int(p.stem.split("_")[0]))
        except ValueError:
            pass   # ignore non-subject parquet files
    return sorted(subjects)


# ══════════════════════════════════════════════════════════════════════
# 2.  MULTICLASS CSP  (One-vs-Rest)
# ══════════════════════════════════════════════════════════════════════

class MulticlassCSP(BaseEstimator, TransformerMixin):
    """
    One MNE CSP per class in {-1, 0, 1} (OVR strategy).

    Binary label for each CSP: (y == cls).astype(int) → {0, 1}
    Output: (n_epochs, n_classes × n_components) = (n, 18)
    """

    def __init__(self, n_components: int = N_CSP_COMPONENTS, reg=None):
        self.n_components = n_components
        self.reg          = reg

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MulticlassCSP":
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError(
                f"MulticlassCSP.fit requires ≥ 2 classes, "
                f"got {self.classes_}.  Check _stratified_subsample."
            )
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
# 3.  F + MI RANK-FUSION SELECTOR  (Borda count)
# ══════════════════════════════════════════════════════════════════════

class FMISelector(BaseEstimator, TransformerMixin):
    """
    Select top-k features by fusing ANOVA-F and MI rankings.

    combined_rank = rank_by_F + rank_by_MI  (lower → more discriminative)
    Scale-invariant: works across CSP log-variance, DE, DASM, RASM.
    """

    def __init__(self, k: int = TOP_K_FEATURES, n_jobs: int = N_JOBS):
        self.k      = k
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FMISelector":
        if X.shape[1] < self.k:
            raise ValueError(
                f"FMISelector: X has {X.shape[1]} features but k={self.k}. "
                f"Reduce TOP_K_FEATURES."
            )
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
# 4.  CLASSIFIERS
# ══════════════════════════════════════════════════════════════════════

def build_clf(name: str):
    """
    SVM C=200 · RBF · OVR — handles {-1, 0, 1} natively.
    LDA shrinkage='auto' (Ledoit-Wolf) — essential when n_features
    (~150) approaches n_samples/class (~400 with GroupKFold).
    """
    if name == "svm":
        return SVC(kernel="rbf", C=200, gamma="scale",
                   decision_function_shape="ovr", random_state=42)
    if name == "lda":
        return LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
    raise ValueError(f"Unknown classifier: {name!r}.  Choose 'svm' or 'lda'.")


# ══════════════════════════════════════════════════════════════════════
# 5.  CROSS-VALIDATION  (GroupKFold by trial_id)
# ══════════════════════════════════════════════════════════════════════

def _stratified_subsample(
    idx:         np.ndarray,
    y:           np.ndarray,
    max_per_cls: int,
    rng:         np.random.Generator,
) -> np.ndarray:
    """
    Down-sample `idx` to ≤ max_per_cls epochs per class.

    Stratified (not contiguous) — SEED trials are label-sorted so
    taking the first N indices would be near-single-class, producing
    degenerate CSP covariance matrices.

    HARD-1: raises RuntimeError if no classes found (guards
    np.concatenate([]) which raises the cryptic ValueError
    "need at least one array to concatenate").
    """
    parts = []
    for cls in np.unique(y[idx]):
        ci = idx[y[idx] == cls]
        if len(ci) > max_per_cls:
            ci = rng.choice(ci, max_per_cls, replace=False)
        parts.append(ci)

    if not parts:
        raise RuntimeError(
            "_stratified_subsample: no classes found in training fold.  "
            "Check that y contains {-1, 0, 1} labels."
        )
    return np.concatenate(parts)


def cross_validate_subject(
    collection:  SessionMemmapCollection,
    X_tab:       np.ndarray,
    y:           np.ndarray,
    trials:      np.ndarray,
    n_folds:     int = N_FOLDS,
    classifier:  str = "svm",
) -> np.ndarray:
    """
    GroupKFold by trial_id.  All pipeline steps fit inside each fold.

    Why GroupKFold
    ──────────────
    SEED windows are 1-second slices of 4-minute film clips.  Adjacent
    windows are highly correlated (same EEG state, same clip).
    StratifiedKFold leaks: the same trial's windows land in both train
    and test, inflating accuracy 10–20 %.
    GroupKFold holds out complete trials → genuine generalisation.

    Feature matrix per fold
    ───────────────────────
    X_tab(580) + CSP(18)  →  598 total
    FMI → 150  →  StandardScaler  →  Classifier

    Labels: native {-1, 0, 1}
      -1 = negative emotion
       0 = neutral
       1 = positive emotion

    HARD-4: asserts all inputs have consistent length before calling
    GroupKFold.split(), which otherwise crashes with an opaque
    "inconsistent number of samples" error.
    """
    # HARD-4: catch row-count mismatches from unpatched preprocessor
    n = len(collection)
    if not (n == len(X_tab) == len(y) == len(trials)):
        raise ValueError(
            f"Row count mismatch: collection={n}, X_tab={len(X_tab)}, "
            f"y={len(y)}, trials={len(trials)}.  "
            f"Re-run preprocessing — the raw.npy and parquet files are out of sync."
        )

    gkf    = GroupKFold(n_splits=n_folds)
    idx    = np.arange(n)
    scores: list[float] = []

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
        fit_raw = collection.load(tr_csp, decimate=DECIMATE)   # ~224 MB
        csp     = MulticlassCSP()
        csp.fit(fit_raw, y[tr_csp])
        del fit_raw; gc.collect()

        # ── CSP transform: single allocation at decimated size ────────
        # collection.load() allocates at (n, 62, T//DECIMATE) directly —
        # the full-resolution tensor never exists in RAM.
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

    return np.array(scores)


# ══════════════════════════════════════════════════════════════════════
# 6.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  SEED EEG — High-Accuracy Pipeline  (OOM-safe v7)")
    print(f"  Features : DE(310) + DASM(135) + RASM(135) + CSP(18)")
    print(f"             → FMI({TOP_K_FEATURES})  total raw: 598")
    print(f"  CV       : GroupKFold(n={N_FOLDS}) by trial_id")
    print(f"  Labels   : native {{-1, 0, 1}}  (no LabelEncoder)")
    print(f"  SVM C=200  ·  LDA shrinkage=auto")
    print(f"  Decimate : 1/{DECIMATE}  ·  CSP subsample : {CSP_FIT_MAX_PER_CLS}/class")
    print(f"  RASM clip: ±{RASM_CLIP}  (computed on raw DE, before z-score)")
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
                collection, X_tab, y, trials, de_layout = load_subject(sub)
            except (FileNotFoundError, ValueError) as exc:
                print(f"  ✗  {exc}")
                continue

            n_trials = int(np.unique(trials).shape[0])
            classes  = np.unique(y).tolist()
            print(
                f"    Windows  : {len(collection)}  |  "
                f"Channels : {collection.shape[1]}  |  "
                f"Tab feats: {X_tab.shape[1]}  |  "
                f"Trials   : {n_trials}  |  "
                f"Classes  : {classes}  |  "
                f"Layout   : {de_layout}"
            )

            if len(classes) < 3:
                print(f"  ⚠  Only {len(classes)} class(es) in subject {sub} — "
                      f"skipping (need 3 for SEED).")
                del collection, X_tab, y, trials
                gc.collect()
                continue

            if n_trials < N_FOLDS:
                print(f"  ⚠  Only {n_trials} trial group(s) — "
                      f"need ≥ {N_FOLDS} for GroupKFold. Skipping.")
                del collection, X_tab, y, trials
                gc.collect()
                continue

            try:
                scores = cross_validate_subject(
                    collection, X_tab, y, trials,
                    n_folds=N_FOLDS, classifier=clf_name,
                )
            except Exception as exc:
                print(f"  ✗  Subject {sub} CV failed: {exc}")
                del collection, X_tab, y, trials
                gc.collect()
                continue

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

            del collection, X_tab, y, trials, scores
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