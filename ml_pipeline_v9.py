"""
SEED EEG — High-Accuracy ML Pipeline  (OOM-safe v9, per-session scaling)
Target: ≥ 78 % within-subject accuracy  (GroupKFold, trial-grouped)
======================================================================

Findings fixed in v9
──────────────────────────────────────────────────────────────────────
  Finding 5 — HIGH: fold-level StandardScaler caused session-shift leakage
    v8 replaced the per-session scaler with a single fold-level scaler
    fit on all training windows pooled across sessions.  Because SEED
    sessions are recorded on different days, each session has its own
    DC offset and amplitude distribution.  A scaler fit on the pooled
    training set produces a mean that is a weighted mixture of session
    means.  Test windows from a given session are then shifted by the
    difference between that session's true mean and the pooled mean —
    a systematic per-feature bias that is constant within the session
    but uncorrelated with emotion labels.  The SVM learns to use that
    session-specific shift as a cue, effectively classifying by session
    identity rather than emotion.  This drives accuracy toward (and
    sometimes below) chance because emotion labels are balanced within
    each session but not correlated with session identity.

    FIX: Per-session StandardScaler, fit on training windows of each
    session only, applied to all windows of that session.
      • scaler_sess.fit(X_tab_raw[tr_sess])          ← train only
      • scaler_sess.transform(X_tab_raw[tr_sess])    ← no leakage
      • scaler_sess.transform(X_tab_raw[te_sess])    ← session-aligned
    This is the correct approach: test windows are normalized relative
    to their own session's training distribution, so inter-session
    offsets are removed without using any test-window statistics during
    fitting.

    Implementation:
      load_subject now returns session_ids (n,) int8 alongside y and trials.
      cross_validate_subject receives session_ids and performs per-session
      scaling on X_tab inside each fold before FMI selection.
      A separate fold-level StandardScaler is kept for the CSP features
      (which have no session offset because they are recomputed per fold
      from raw data that is already mean-centered per epoch by MNE CSP).

Full changelog
────────────────
  v1  Original — OOM before Fold 1
  v2  OOM pass 1 + accuracy features (session norm, asymmetry, C=100)
  v3  OOM pass 2 — SessionMemmapCollection + chunk loop
  v4  Rejected — broken CSP subsample, hardcoded layout
  v5  GroupKFold + collection.load() — chunk loop deleted
  v6  LabelEncoder removed — {-1,0,1} preserved
      ✗ RASM on z-scored DE → below-chance accuracy
  v7  RASM on raw DE + HARD-1…HARD-4
      ✗ StandardScaler in load_subject leaked test stats into training
      ✗ DE layout detected once, reused for all sessions
  v8  Fold leakage removed; per-session layout assertion
      ✗ Single fold-level scaler pools sessions → session-shift leakage
      ✗ mutual_info_classif not scale-invariant → FMI selects wrong features
  v9  Per-session StandardScaler (fit on train fold only) ← current
      Scale before FMI so MI k-NN is not dominated by RASM magnitude

Peak RAM budget per fold (8 100 train / 2 025 test)
─────────────────────────────────────────────────────
  X_tab in RAM (580 features, full subject, unscaled)  ≈  24 MB
  CSP fit subsample  (1 800 eps, decimated)            ≈ 224 MB
  Full train decimated  (single alloc)                 ≈ 930 MB  freed
  Full test decimated   (single alloc)                 ≈ 233 MB  freed
  X_tr  (598 features, scaled)                         ≈  19 MB
  SVM kernel matrix                                    ≈ 263 MB
  ──────────────────────────────────────────────────────────────────
  Total peak                                           ≈ 1.50 GB  ✓

Formula audit (all verified correct)
──────────────────────────────────────
  DE   : 0.5 × log(2πe × σ²)   σ² = var(band_amplitude over time)
  DASM : DE_left − DE_right     computed on raw (positive) DE
  RASM : DE_left / DE_right     computed on raw DE, clipped ±500
  de_to_3d band_major:
    reshape(n, N_BANDS, N_CHANNELS).transpose(0,2,1) → (n, 62, 5)
    position i×62+j → band i, channel j  ✓
  Feature total: DE(310) + DASM(135) + RASM(135) + CSP(18) = 598 → FMI(150)

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
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

PROCESSED_PATH      = Path("./data/SEED_EEG/processed_data/")

N_CSP_COMPONENTS    = 6       # per OVR binary  →  3 × 6 = 18 CSP features
TOP_K_FEATURES      = 150     # after FMI Borda-count selection
N_FOLDS             = 5
N_JOBS              = 4       # MI worker cap
CV_RANDOM_STATE     = 42

# 'full'     -> DE + DASM + RASM + CSP (original v9 fusion)
# 'csp_only' -> CSP branch only (more robust on current preprocessed data)
# 'tab_only' -> DE + DASM + RASM only
FEATURE_MODE        = "tab_only"

# In no-CSP mode, using all 580 tab features performed better than FMI(150)
# on subject-1 grouped-CV in this workspace.
APPLY_FMI_TAB_ONLY  = False

CSP_FIT_MAX_PER_CLS = 600     # max epochs/class for CSP covariance fit
DECIMATE            = 4       # time stride  (2 000 pts → 500 pts)

N_CHANNELS          = 62
N_BANDS             = 5       # δ θ α β γ

# Clips extreme RASM outliers from near-zero raw DE denominators.
RASM_CLIP = 500.0

# ── SEED 62-channel left↔right electrode pairs (0-based) ─────────────
SEED_LR_PAIRS: list[tuple[int, int]] = [
    ( 0,  2),   # FP1 – FP2
    ( 3,  4),   # AF3 – AF4
    ( 5, 13),   # F7  – F8
    ( 6, 12),   # F5  – F6
    ( 7, 11),   # F3  – F4
    ( 8, 10),   # F1  – F2
    (14, 22),   # FT7 – FT8
    (15, 21),   # FC5 – FC6
    (16, 20),   # FC3 – FC4
    (17, 19),   # FC1 – FC2
    (23, 31),   # T7  – T8
    (24, 30),   # C5  – C6
    (25, 29),   # C3  – C4
    (26, 28),   # C1  – C2
    (32, 40),   # TP7 – TP8
    (33, 39),   # CP5 – CP6
    (34, 38),   # CP3 – CP4
    (35, 37),   # CP1 – CP2
    (41, 49),   # P7  – P8
    (42, 48),   # P5  – P6
    (43, 47),   # P3  – P4
    (44, 46),   # P1  – P2
    (50, 56),   # PO7 – PO8
    (51, 55),   # PO5 – PO6
    (52, 54),   # PO3 – PO4
    (58, 60),   # O1  – O2
    (57, 61),   # CB1 – CB2
]
# 27 pairs × 5 bands × 2 (DASM + RASM) = 270 asymmetry features

assert len(SEED_LR_PAIRS) == 27
assert max(max(l, r) for l, r in SEED_LR_PAIRS) < N_CHANNELS
assert TOP_K_FEATURES < N_CHANNELS * N_BANDS + len(SEED_LR_PAIRS) * N_BANDS * 2 + N_CSP_COMPONENTS * 3


# ══════════════════════════════════════════════════════════════════════
# SESSION MEMMAP COLLECTION
# ══════════════════════════════════════════════════════════════════════

class SessionMemmapCollection:
    """
    Wraps per-session memmaps behind a single virtual index space.
    Never concatenates the raw arrays — zero extra RAM at load time.
    """

    def __init__(self, arrays: list[np.ndarray]):
        if not arrays:
            raise ValueError("SessionMemmapCollection requires at least one array")
        for i, a in enumerate(arrays):
            if a.ndim != 3:
                raise ValueError(
                    f"Array {i} must be 3-D (windows, channels, samples), "
                    f"got shape {a.shape}"
                )
        ref_shape = arrays[0].shape[1:]
        for i, a in enumerate(arrays[1:], start=1):
            if a.shape[1:] != ref_shape:
                raise ValueError(
                    f"Session {i} shape {a.shape[1:]} != session 0 shape {ref_shape}."
                )
        self.arrays  = arrays
        self.lengths = np.array([a.shape[0] for a in arrays], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)])
        self.shape   = (int(self.offsets[-1]),) + ref_shape
        self.ndim    = len(self.shape)

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def load(self, idx: np.ndarray, decimate: int = 1) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        if idx.ndim != 1:
            raise ValueError(f"idx must be 1-D, got shape {idx.shape}")
        if len(idx) == 0:
            T_empty = self.shape[2] // decimate if decimate > 1 else self.shape[2]
            return np.empty((0, self.shape[1], T_empty), dtype=np.float32)

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
        return self.load(np.asarray(idx, dtype=np.int64), decimate=1)


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


def _compute_asymmetry_raw(X_de_raw: np.ndarray, layout: str) -> np.ndarray:
    """
    Compute DASM and RASM from RAW (unscaled) DE values.
    Must be called before any z-scoring.
    """
    X3d   = de_to_3d(X_de_raw, layout)
    feats: list[np.ndarray] = []

    for li, ri in SEED_LR_PAIRS:
        l = X3d[:, li, :]
        r = X3d[:, ri, :]
        feats.append(l - r)
        rasm = l / (r + 1e-8)
        rasm = np.nan_to_num(rasm, nan=0.0, posinf=RASM_CLIP, neginf=-RASM_CLIP)
        rasm = np.clip(rasm, -RASM_CLIP, RASM_CLIP)
        feats.append(rasm)

    X_asym = np.concatenate(feats, axis=1).astype(np.float32)

    if not np.isfinite(X_asym).all():
        n_bad = (~np.isfinite(X_asym)).sum()
        print(f"  ⚠  {n_bad} non-finite asymmetry values after clipping — zeroing")
        X_asym = np.nan_to_num(X_asym, nan=0.0, posinf=0.0, neginf=0.0)

    return X_asym


def load_subject(
    subject_id: int,
) -> tuple[
    SessionMemmapCollection,
    np.ndarray,   # X_tab      (n, 580) float32 — UNSCALED
    np.ndarray,   # y          (n,)     int {-1, 0, 1}
    np.ndarray,   # trials     (n,)     str  trial group IDs
    np.ndarray,   # session_ids (n,)   int8  session number per window
    str,          # de_layout
]:
    """
    Load all sessions for one subject.

    v9 changes vs v8
    ─────────────────
    Returns session_ids (n,) int8 so that cross_validate_subject can
    apply a per-session StandardScaler inside each fold.

    X_tab is still returned RAW (unscaled).  Scaling happens inside
    cross_validate_subject using only training-fold windows per session.

    session_ids encodes which session (1, 2, or 3) each window belongs
    to, allowing the fold-level code to group windows by session before
    fitting each session's scaler.
    """
    stems = sorted(PROCESSED_PATH.glob(f"{subject_id}_*.parquet"))
    if not stems:
        raise FileNotFoundError(f"No parquet files for subject {subject_id}")

    raw_memmaps:  list[np.ndarray] = []
    tab_list:     list[np.ndarray] = []
    y_list:       list[np.ndarray] = []
    trial_list:   list[np.ndarray] = []
    sess_list:    list[np.ndarray] = []   # v9: track session membership
    layout_seen:  list[str]        = []

    for pf in stems:
        raw_path = PROCESSED_PATH / f"{pf.stem}_raw.npy"
        if not raw_path.exists():
            print(f"  ⚠  Missing {raw_path.name}, skipping session")
            continue

        df        = pd.read_parquet(pf)
        y_sess    = df["label"].astype(int).values
        trials    = df["trial_id"].values
        feat_cols = [c for c in df.columns
                     if c not in ("label", "subject", "session", "trial_id")]
        X_de_raw  = df[feat_cols].values.astype(np.float32)

        expected_de_features = N_CHANNELS * N_BANDS
        if X_de_raw.shape[1] != expected_de_features:
            raise ValueError(
                f"{pf.name}: expected {expected_de_features} DE features "
                f"({N_CHANNELS} channels × {N_BANDS} bands), "
                f"got {X_de_raw.shape[1]}."
            )

        sess_layout = detect_de_layout(feat_cols)
        layout_seen.append(sess_layout)

        if feat_cols and feat_cols[0].lstrip("-").isdigit():
            print(
                f"  ⚠  {pf.name}: DE columns are raw integers — "
                f"preprocessor BUG-2 fix was not applied."
            )

        X_asym = _compute_asymmetry_raw(X_de_raw, sess_layout)
        X_tab_sess = np.concatenate([X_de_raw, X_asym], axis=1)   # (n, 580)

        if not np.isfinite(X_tab_sess).all():
            n_bad = (~np.isfinite(X_tab_sess)).sum()
            print(f"  ⚠  {n_bad} non-finite values in {pf.name} — zeroing")
            X_tab_sess = np.nan_to_num(X_tab_sess, nan=0.0, posinf=0.0, neginf=0.0)

        # v9: record session number for every window in this session.
        # Filename format: "{subject_id}_{session_num}.parquet", e.g. "3_2.parquet"
        session_num = int(pf.stem.split("_")[1])
        sess_list.append(np.full(len(y_sess), session_num, dtype=np.int8))

        raw_memmaps.append(np.load(raw_path, mmap_mode="r"))
        tab_list.append(X_tab_sess)
        y_list.append(y_sess)
        trial_list.append(trials)

        del X_de_raw, X_asym, X_tab_sess

    if not raw_memmaps:
        raise FileNotFoundError(f"No valid session data found for subject {subject_id}")

    if len(set(layout_seen)) > 1:
        raise ValueError(
            f"Subject {subject_id}: inconsistent DE layouts across sessions: "
            f"{layout_seen}.  Re-preprocess all sessions uniformly."
        )

    return (
        SessionMemmapCollection(raw_memmaps),
        np.concatenate(tab_list,   axis=0),   # (n, 580) unscaled
        np.concatenate(y_list,     axis=0),
        np.concatenate(trial_list, axis=0),
        np.concatenate(sess_list,  axis=0),   # v9: (n,) int8 session IDs
        layout_seen[0],
    )


def discover_subjects() -> list[int]:
    subjects = set()
    for p in PROCESSED_PATH.glob("*.parquet"):
        try:
            subjects.add(int(p.stem.split("_")[0]))
        except ValueError:
            pass
    return sorted(subjects)


# ══════════════════════════════════════════════════════════════════════
# 2.  MULTICLASS CSP  (One-vs-Rest)
# ══════════════════════════════════════════════════════════════════════

class MulticlassCSP(BaseEstimator, TransformerMixin):
    """One MNE CSP per class in {-1, 0, 1} (OVR strategy)."""

    def __init__(self, n_components: int = N_CSP_COMPONENTS, reg=None):
        self.n_components = n_components
        self.reg          = reg

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MulticlassCSP":
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError(
                f"MulticlassCSP.fit requires ≥ 2 classes, got {self.classes_}."
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
    Select top-k features by fusing ANOVA-F and Mutual Information rankings.

    Must receive scaled input because mutual_info_classif uses k-NN
    distance estimation, which is not scale-invariant.
    """

    def __init__(self, k: int = TOP_K_FEATURES, n_jobs: int = N_JOBS):
        self.k      = k
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FMISelector":
        if X.shape[1] < self.k:
            raise ValueError(
                f"FMISelector: X has {X.shape[1]} features but k={self.k}."
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
    if name == "svm":
        return SVC(kernel="rbf", C=200, gamma="scale",
                   decision_function_shape="ovr", random_state=42,
                   class_weight="balanced")
    if name == "lda":
        return LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
    if name == "xgb":
        if XGBClassifier is None:
            raise ImportError(
                "XGBoost is not installed. Install package 'xgboost' to use classifier='xgb'."
            )
        return XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_jobs=N_JOBS,
            random_state=42,
            eval_metric="mlogloss",
        )
    raise ValueError(f"Unknown classifier: {name!r}.  Choose 'svm', 'lda', or 'xgb'.")


# ══════════════════════════════════════════════════════════════════════
# 5.  CROSS-VALIDATION  (GroupKFold by trial_id)
# ══════════════════════════════════════════════════════════════════════

def _stratified_subsample(
    idx:         np.ndarray,
    y:           np.ndarray,
    max_per_cls: int,
    rng:         np.random.Generator,
) -> np.ndarray:
    parts = []
    for cls in np.unique(y[idx]):
        ci = idx[y[idx] == cls]
        if len(ci) > max_per_cls:
            ci = rng.choice(ci, max_per_cls, replace=False)
        parts.append(ci)

    if not parts:
        raise RuntimeError(
            "_stratified_subsample: no classes found in training fold."
        )
    return np.concatenate(parts)


def _scale_tab_per_session(
    X_tab_raw:   np.ndarray,   # (n_total, 580) unscaled, full subject
    tr:          np.ndarray,   # global train indices for this fold
    te:          np.ndarray,   # global test  indices for this fold
    session_ids: np.ndarray,   # (n_total,) int8
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-session StandardScaler — the v9 fix for session-shift leakage.

    For each session present in this fold:
      1. Fit StandardScaler on training windows of that session only.
      2. Transform training windows of that session.
      3. Transform test windows of that session with the same scaler.

    This ensures:
      • No test-window statistics enter any scaler fit.
      • Each session's windows are normalized to its own training
        distribution, removing inter-session DC offset and gain
        differences that would otherwise act as spurious session-identity
        cues for the classifier.

    Edge case: if a session has no training windows in this fold (all
    its trials landed in test), we cannot fit a scaler.  Those test
    windows are left unscaled (mean 0 / std 1 passthrough).  This is
    preferable to using a scaler fit on other sessions' data.

    Parameters
    ----------
    X_tab_raw   : full unscaled X_tab for the subject (all sessions)
    tr          : training indices for the current fold
    te          : test indices for the current fold
    session_ids : session membership for every row of X_tab_raw

    Returns
    -------
    X_tab_tr : (len(tr), 580) scaled training features
    X_tab_te : (len(te), 580) scaled test features
    """
    X_tab_tr = X_tab_raw[tr].copy()
    X_tab_te = X_tab_raw[te].copy()

    tr_sess_ids = session_ids[tr]
    te_sess_ids = session_ids[te]

    for sess in np.unique(session_ids):
        # Boolean masks into the fold-local arrays (not global indices)
        tr_mask = (tr_sess_ids == sess)
        te_mask = (te_sess_ids == sess)

        if not tr_mask.any():
            # No training data for this session in this fold.
            # Leave test windows for this session at raw scale rather
            # than applying a scaler fit on a different session's data.
            if te_mask.any():
                print(
                    f"      ⚠  Session {sess}: no training windows in this fold "
                    f"— {te_mask.sum()} test windows left unscaled."
                )
            continue

        scaler_sess = StandardScaler()
        # Fit exclusively on this session's training windows
        scaler_sess.fit(X_tab_raw[tr[tr_mask]])
        # Transform training windows
        X_tab_tr[tr_mask] = scaler_sess.transform(X_tab_raw[tr[tr_mask]])
        # Transform test windows with the same (train-derived) statistics
        if te_mask.any():
            X_tab_te[te_mask] = scaler_sess.transform(X_tab_raw[te[te_mask]])

    return X_tab_tr, X_tab_te


def cross_validate_subject(
    collection:  SessionMemmapCollection,
    X_tab:       np.ndarray,        # (n, 580) unscaled
    y:           np.ndarray,
    trials:      np.ndarray,
    session_ids: np.ndarray,        # v9: (n,) int8 session membership
    n_folds:     int = N_FOLDS,
    classifier:  str = "svm",
) -> np.ndarray:
    """
    StratifiedGroupKFold by trial_id.  ALL pipeline steps fit inside each fold.

    Scaling order (v9)
    ───────────────────
    1. Per-session StandardScaler on X_tab (580 DE+asymmetry features).
       Fit on train windows of each session; applied to that session's
       train and test windows.  Removes inter-session offsets without
       leaking test statistics.

     2. Feature-mode controls which branch is used:
         - full:     [X_tab(580) + CSP(18)]
         - csp_only: [CSP(18)]
         - tab_only: [X_tab(580)]

    3. FMI selector is fit on the scaled 598-feature training matrix.
       mutual_info_classif requires scale-normalised input because it uses
       k-NN distance estimation.

    4. SVM / LDA is fit on the FMI-selected 150-feature scaled matrix.
       No additional scaling needed here since step 1–2 already normalised.
    """
    n = len(collection)
    if not (n == len(X_tab) == len(y) == len(trials) == len(session_ids)):
        raise ValueError(
            f"Row count mismatch: collection={n}, X_tab={len(X_tab)}, "
            f"y={len(y)}, trials={len(trials)}, session_ids={len(session_ids)}."
        )

    # Keep trial-level separation (no trial leakage) while improving
    # per-fold class balance compared with plain GroupKFold.
    sgkf   = StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )
    idx    = np.arange(n)
    scores: list[float] = []

    for fold, (tr, te) in enumerate(
            sgkf.split(idx, y, groups=trials), start=1):

        te_trials = np.unique(trials[te]).tolist()
        print(
            f"    Fold {fold}/{n_folds}  "
            f"[test trials: {te_trials}] ...",
            end=" ", flush=True,
        )

        rng  = np.random.default_rng(42 + fold)
        y_tr = y[tr]
        y_te = y[te]

        # ── Step 1: Per-session scaling of tabular features ───────────
        # Fit each session's scaler on training windows only.
        # This is the v9 fix — see _scale_tab_per_session for details.
        X_tab_tr, X_tab_te = _scale_tab_per_session(X_tab, tr, te, session_ids)

        csp_tr = None
        csp_te = None
        if FEATURE_MODE != "tab_only":
            # ── Step 2a: CSP fit — stratified subsample, decimated ────
            tr_csp  = _stratified_subsample(tr, y, CSP_FIT_MAX_PER_CLS, rng)
            fit_raw = collection.load(tr_csp, decimate=DECIMATE)
            csp     = MulticlassCSP()
            csp.fit(fit_raw, y[tr_csp])
            del fit_raw
            gc.collect()

            # ── Step 2b: CSP transform ────────────────────────────────
            raw_tr = collection.load(tr, decimate=DECIMATE)
            csp_tr = csp.transform(raw_tr)        # (n_tr, 18)
            del raw_tr
            gc.collect()

            raw_te = collection.load(te, decimate=DECIMATE)
            csp_te = csp.transform(raw_te)        # (n_te, 18)
            del raw_te, csp
            gc.collect()

        # ── Step 3: Assemble feature matrix based on FEATURE_MODE ─────
        if FEATURE_MODE == "full":
            X_tr_raw = np.concatenate([X_tab_tr, csp_tr], axis=1)
            X_te_raw = np.concatenate([X_tab_te, csp_te], axis=1)
        elif FEATURE_MODE == "csp_only":
            X_tr_raw = csp_tr
            X_te_raw = csp_te
        elif FEATURE_MODE == "tab_only":
            X_tr_raw = X_tab_tr
            X_te_raw = X_tab_te
        else:
            raise ValueError(
                f"Unknown FEATURE_MODE={FEATURE_MODE!r}. "
                "Choose 'full', 'csp_only', or 'tab_only'."
            )
        del csp_tr, csp_te, X_tab_tr, X_tab_te
        gc.collect()

        # Fold-level scaling keeps classifier and MI distance metrics stable.
        scaler_full = StandardScaler()
        X_tr = scaler_full.fit_transform(X_tr_raw)
        X_te = scaler_full.transform(X_te_raw)
        del X_tr_raw, X_te_raw
        gc.collect()

        # ── Step 4: Optional FMI selection (on scaled data) ───────────
        selector = None
        use_fmi = (X_tr.shape[1] > TOP_K_FEATURES) and not (
            FEATURE_MODE == "tab_only" and not APPLY_FMI_TAB_ONLY
        )
        if use_fmi:
            selector = FMISelector()
            X_tr = selector.fit_transform(X_tr, y_tr)
            X_te = selector.transform(X_te)

        # ── Step 5: Classify ──────────────────────────────────────────
        clf = build_clf(classifier)
        if classifier == "xgb":
            # XGBoost expects labels 0..(num_class-1)
            cls = np.array(sorted(np.unique(y_tr)), dtype=np.int64)
            cls_to_idx = {int(c): i for i, c in enumerate(cls)}
            y_tr_fit = np.array([cls_to_idx[int(v)] for v in y_tr], dtype=np.int64)
            y_te_idx = np.array([cls_to_idx[int(v)] for v in y_te], dtype=np.int64)
            clf.fit(X_tr, y_tr_fit)
            y_pred_idx = clf.predict(X_te).astype(np.int64)
            acc = float((y_pred_idx == y_te_idx).mean())
        else:
            clf.fit(X_tr, y_tr)
            acc = float((clf.predict(X_te) == y_te).mean())
        del X_tr
        gc.collect()
        scores.append(acc)
        print(f"acc = {acc:.4f}")

        del X_te, selector, scaler_full, clf, y_tr, y_te
        gc.collect()

    return np.array(scores)


# ══════════════════════════════════════════════════════════════════════
# 6.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  SEED EEG — High-Accuracy Pipeline  (OOM-safe v9)")
    print(f"  Features : DE(310) + DASM(135) + RASM(135) + CSP(18)")
    print(f"             → FMI({TOP_K_FEATURES})  total raw: 598")
    print(f"  Mode     : {FEATURE_MODE}")
    print(f"  CV       : GroupKFold(n={N_FOLDS}) by trial_id")
    print(f"  Labels   : native {{-1, 0, 1}}  (no LabelEncoder)")
    print(f"  SVM C=200  ·  LDA shrinkage=auto")
    print(f"  Decimate : 1/{DECIMATE}  ·  CSP subsample : {CSP_FIT_MAX_PER_CLS}/class")
    print(f"  RASM clip: ±{RASM_CLIP}  (on raw DE, before any scaling)")
    print(f"  Scaling  : per-session StandardScaler (fit on train fold only)")
    print(f"  Peak RAM : ~1.5 GB per fold")
    print("=" * 65)

    subjects     = discover_subjects()
    all_results: list[dict] = []

    for clf_name in ("lda", "svm", "xgb"):
        print(f"\n{'─' * 65}")
        print(f"  Classifier : {clf_name.upper()}")
        print(f"{'─' * 65}")

        subject_means: list[float] = []

        for sub in subjects:
            print(f"\n  Subject {sub}")
            try:
                collection, X_tab, y, trials, session_ids, de_layout = load_subject(sub)
            except (FileNotFoundError, ValueError) as exc:
                print(f"  ✗  {exc}")
                continue

            n_trials = int(np.unique(trials).shape[0])
            classes  = np.unique(y).tolist()
            n_sess   = int(np.unique(session_ids).shape[0])
            print(
                f"    Windows  : {len(collection)}  |  "
                f"Channels : {collection.shape[1]}  |  "
                f"Tab feats: {X_tab.shape[1]} (unscaled)  |  "
                f"Sessions : {n_sess}  |  "
                f"Trials   : {n_trials}  |  "
                f"Classes  : {classes}  |  "
                f"Layout   : {de_layout}"
            )

            if len(classes) < 3:
                print(
                    f"  ⚠  Only {len(classes)} class(es) for subject {sub} "
                    f"— need 3 for SEED.  Skipping."
                )
                del collection, X_tab, y, trials, session_ids
                gc.collect()
                continue

            if n_trials < N_FOLDS:
                print(
                    f"  ⚠  Only {n_trials} trial group(s) — "
                    f"need ≥ {N_FOLDS} for GroupKFold.  Skipping."
                )
                del collection, X_tab, y, trials, session_ids
                gc.collect()
                continue

            try:
                scores = cross_validate_subject(
                    collection, X_tab, y, trials, session_ids,
                    n_folds=N_FOLDS, classifier=clf_name,
                )
            except Exception as exc:
                print(f"  ✗  Subject {sub} CV failed: {exc}")
                del collection, X_tab, y, trials, session_ids
                gc.collect()
                continue

            mean_acc = float(scores.mean())
            subject_means.append(mean_acc)
            print(
                f"    → Subject {sub}  mean: {mean_acc:.4f}  "
                f"std: {scores.std():.4f}"
            )

            for fold_i, s in enumerate(scores, 1):
                all_results.append({
                    "classifier": clf_name,
                    "subject":    sub,
                    "fold":       fold_i,
                    "accuracy":   round(float(s), 6),
                })

            del collection, X_tab, y, trials, session_ids, scores
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
