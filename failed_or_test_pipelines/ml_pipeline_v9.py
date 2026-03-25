"""
SEED EEG — Binary Classification ML Pipeline (v10)
===================================================
Binary: Positive (1) vs Negative (-1)  — Neutral (0) excluded

Key Design Principles
─────────────────────
  1. BINARY CLASSIFICATION
     - Only positive (1) and negative (-1) labels
     - Neutral (0) excluded

  2. WITHIN-SESSION CV (NO SESSION MIXING)
     - CV is done WITHIN each session separately
     - Train and test from SAME session
     - Results aggregated across sessions
     - Prevents session-specific artifacts from leaking

  3. PER-SESSION SCALING (NO GLOBAL SCALING)
     - Scale using ONLY that session's training data
     - No mixing of session statistics

  4. TRIAL-GROUPED CV
     - GroupKFold by trial_id within each session
     - Windows from same trial stay together

Memory Budget: ~1.5 GB per fold
Expected Accuracy: 70-80%
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
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════
# STRATIFIED GROUP KFOLD (custom implementation)
# ══════════════════════════════════════════════════════════════════════
class StratifiedGroupKFold:
    """
    Stratified K-Fold that respects group membership.
    Ensures no group appears in both train and test, while maintaining
    class balance in train/test splits.
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y, groups):
        """
        Yield (train, test) indices.

        Parameters
        ----------
        X : array-like, shape (n_samples,)
        y : array-like, shape (n_samples,)  labels
        groups : array-like, shape (n_samples,)  group IDs
        """
        X = np.asarray(X)
        y = np.asarray(y)
        groups = np.asarray(groups)

        # Encode labels to non-negative for bincount
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        unique_groups = np.unique(groups)
        unique_classes = np.unique(y_encoded)
        rng = np.random.default_rng(self.random_state)

        # Map each group to its dominant class
        group_to_class = {}
        for g in unique_groups:
            mask = groups == g
            cls_counts = np.bincount(y_encoded[mask])
            dominant_class = np.argmax(cls_counts)
            group_to_class[g] = int(dominant_class)

        # Separate groups by class
        groups_by_class = {}
        for c in unique_classes:
            groups_by_class[int(c)] = [g for g in unique_groups if group_to_class[g] == int(c)]

        if self.shuffle:
            for c in unique_classes:
                rng.shuffle(groups_by_class[int(c)])

        # Distribute groups into folds, stratified by class
        folds = [[] for _ in range(self.n_splits)]
        for c in unique_classes:
            for i, g in enumerate(groups_by_class[int(c)]):
                folds[i % self.n_splits].append(g)

        # Yield splits
        fold_groups = [set(f) for f in folds]
        for test_fold_idx in range(self.n_splits):
            test_group_mask = np.isin(groups, list(fold_groups[test_fold_idx]))
            train_group_mask = ~test_group_mask

            yield np.where(train_group_mask)[0], np.where(test_group_mask)[0]


# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

PROCESSED_PATH = Path("./data/SEED_EEG/processed_data/")

N_CSP_COMPONENTS = 6
TOP_K_FEATURES = 0
N_FOLDS = 5  # Within-session CV folds
N_JOBS = 4
RANDOM_STATE = 42

FEATURE_MODE = "full"  # 'full', 'csp_only', 'tab_only'
CV_MODE = "stratified_group"  # 'group' (trial-grouped) | 'stratified_group' (stratified + no leakage) — NO LEAKY OPTIONS
CLASSIFIERS = ("svm",)

CSP_FIT_MAX_PER_CLS = 300
DECIMATE = 5

N_CHANNELS = 62
N_BANDS = 5


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING (PER SESSION)
# ══════════════════════════════════════════════════════════════════════

def load_session(subject_id: int, session_id: int) -> tuple:
    """
    Load ONE session for one subject.

    Returns
    -------
    raw_data : np.memmap of all raw windows for the session
    raw_idx : (n,) int indices of binary windows in raw_data
    X_tab : (n, 310) float32 — DE features (unscaled)
    y : (n,) int — labels {-1, 1}
    trials : (n,) str — trial IDs for GroupKFold
    """
    pf = PROCESSED_PATH / f"{subject_id}_{session_id}.parquet"
    raw_path = PROCESSED_PATH / f"{subject_id}_{session_id}_raw.npy"

    if not pf.exists() or not raw_path.exists():
        raise FileNotFoundError(f"Missing files for subject {subject_id} session {session_id}")

    df = pd.read_parquet(pf)
    y = df["label"].astype(int).values
    trials = df["trial_id"].values

    # BINARY FILTER: Remove neutral (0)
    binary_mask = y != 0
    if not binary_mask.any():
        raise ValueError(f"No binary labels in session {session_id}")

    y = y[binary_mask]
    trials = trials[binary_mask]
    binary_indices = np.where(binary_mask)[0]

    feat_cols = [c for c in df.columns
                 if c not in ("label", "subject", "session", "trial_id")]
    X_tab = df[feat_cols].values.astype(np.float32)[binary_mask]

    # Keep raw as memmap and index it per fold to avoid loading full session into RAM.
    raw_data = np.load(raw_path, mmap_mode="r")

    return raw_data, binary_indices.astype(np.int64), X_tab, y, trials


def discover_sessions(subject_id: int) -> list[int]:
    """Find all sessions for a subject."""
    sessions = []
    for p in PROCESSED_PATH.glob(f"{subject_id}_*.parquet"):
        try:
            sess = int(p.stem.split("_")[1])
            sessions.append(sess)
        except (IndexError, ValueError):
            pass
    return sorted(sessions)


def discover_subjects() -> list[int]:
    """Find all subjects."""
    subjects = set()
    for p in PROCESSED_PATH.glob("*.parquet"):
        try:
            subjects.add(int(p.stem.split("_")[0]))
        except ValueError:
            pass
    return sorted(subjects)


# ══════════════════════════════════════════════════════════════════════
# BINARY CSP
# ══════════════════════════════════════════════════════════════════════

class BinaryCSP(BaseEstimator, TransformerMixin):
    """Binary CSP wrapper."""

    def __init__(self, n_components: int = N_CSP_COMPONENTS, reg=None):
        self.n_components = n_components
        self.reg = reg

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BinaryCSP":
        self.csp_ = CSP(
            n_components=self.n_components,
            reg=self.reg,
            log=True,
            norm_trace=False
        )
        y_binary = (y == 1).astype(int)
        self.csp_.fit(X, y_binary)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.csp_.transform(X)


# ══════════════════════════════════════════════════════════════════════
# FEATURE SELECTOR
# ══════════════════════════════════════════════════════════════════════

class FMISelector(BaseEstimator, TransformerMixin):
    """Select top-k features by F+MI rank fusion."""

    def __init__(self, k: int = TOP_K_FEATURES, n_jobs: int = N_JOBS):
        self.k = k
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FMISelector":
        if X.shape[1] <= self.k:
            self.selected_idx_ = np.arange(X.shape[1])
            return self

        f_sc, _ = f_classif(X, y)
        mi_sc = mutual_info_classif(X, y, random_state=42, n_jobs=self.n_jobs)
        f_sc = np.nan_to_num(f_sc)
        mi_sc = np.nan_to_num(mi_sc)
        f_r = np.argsort(np.argsort(-f_sc))
        mi_r = np.argsort(np.argsort(-mi_sc))
        self.selected_idx_ = np.argsort(f_r + mi_r)[:self.k]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_idx_]


# ══════════════════════════════════════════════════════════════════════
# CLASSIFIER
# ══════════════════════════════════════════════════════════════════════

def build_clf(name: str):
    if name == "svm":
        return SVC(
            kernel="rbf",
            C=10,           # Reduced from 30 to prevent overfitting
            gamma=0.001,    # Reduced from 0.01 for better generalization
            random_state=42,
            class_weight="balanced"
        )
    if name == "lda":
        return LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
    raise ValueError(f"Unknown classifier: {name}")


# ══════════════════════════════════════════════════════════════════════
# WITHIN-SESSION CV (NO SESSION MIXING)
# ══════════════════════════════════════════════════════════════════════

def _stratified_subsample(
    idx: np.ndarray,
    y: np.ndarray,
    max_per_cls: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stratified subsample for CSP fitting."""
    parts = []
    for cls in np.unique(y[idx]):
        ci = idx[y[idx] == cls]
        if len(ci) > max_per_cls:
            ci = rng.choice(ci, max_per_cls, replace=False)
        parts.append(ci)
    if not parts:
        return np.array([], dtype=np.int64)
    return np.concatenate(parts)


def _load_raw_batch(
    raw_data: np.ndarray,
    raw_idx: np.ndarray,
    rows: np.ndarray,
    decimate: int,
) -> np.ndarray:
    """Load a fold subset from memmap and decimate before model use."""
    return np.asarray(raw_data[raw_idx[rows], :, ::decimate], dtype=np.float32)


def cv_within_session(
    raw_data: np.ndarray,
    raw_idx: np.ndarray,
    X_tab: np.ndarray,
    y: np.ndarray,
    trials: np.ndarray,
    n_folds: int = N_FOLDS,
    classifier: str = "svm",
) -> np.ndarray:
    """
    Cross-validation WITHIN a single session.

    - GroupKFold by trial_id (windows from same trial stay together)
    - Scale using ONLY this session's training data
    - No mixing with other sessions

    Returns
    -------
    scores : array of fold accuracies
    """
    n = len(y)
    unique_trials = np.unique(trials)
    if CV_MODE == "group":
        actual_folds = min(n_folds, len(unique_trials))
        if actual_folds < 2:
            print(f"        Only {len(unique_trials)} trial(s), cannot do CV")
            return np.array([])
        split_iter = GroupKFold(n_splits=actual_folds).split(np.arange(n), y, groups=trials)
    elif CV_MODE == "stratified_group":
        actual_folds = min(n_folds, len(unique_trials))
        if actual_folds < 2:
            print(f"        Only {len(unique_trials)} trial(s), cannot do CV")
            return np.array([])
        split_iter = StratifiedGroupKFold(
            n_splits=actual_folds,
            shuffle=True,
            random_state=RANDOM_STATE,
        ).split(np.arange(n), y, groups=trials)
    # REMOVED LEAKY "stratified" option to prevent data leakage
    # Only honest GroupKFold methods are allowed
    else:
        msg = f"Unknown CV_MODE: {CV_MODE}. Only 'group' and 'stratified_group' are allowed (no data leakage)."
        raise ValueError(msg)

    scores = []

    for fold, (tr, te) in enumerate(split_iter, start=1):
        y_tr, y_te = y[tr], y[te]

        # Check both classes in train
        if len(np.unique(y_tr)) < 2:
            print(f"        Fold {fold}: only 1 class in train, skipping")
            continue

        rng = np.random.default_rng(42 + fold)

        # ── Step 1: Scale tabular features (WITHIN SESSION ONLY) ───────
        scaler_tab = StandardScaler()
        X_tab_tr = scaler_tab.fit_transform(X_tab[tr])
        X_tab_te = scaler_tab.transform(X_tab[te])

        # ── Step 2: CSP ────────────────────────────────────────────────
        tr_csp = _stratified_subsample(tr, y, CSP_FIT_MAX_PER_CLS, rng)
        if tr_csp.size == 0:
            print(f"        Fold {fold}: empty CSP subsample, skipping")
            continue

        X_raw_dec = _load_raw_batch(raw_data, raw_idx, tr_csp, DECIMATE)

        csp = BinaryCSP()
        csp.fit(X_raw_dec, y[tr_csp])
        del X_raw_dec
        gc.collect()

        X_raw_tr = _load_raw_batch(raw_data, raw_idx, tr, DECIMATE)
        csp_tr = csp.transform(X_raw_tr)
        del X_raw_tr

        X_raw_te = _load_raw_batch(raw_data, raw_idx, te, DECIMATE)
        csp_te = csp.transform(X_raw_te)
        del X_raw_te

        del csp
        gc.collect()

        # Scale CSP features
        scaler_csp = StandardScaler()
        csp_tr = scaler_csp.fit_transform(csp_tr)
        csp_te = scaler_csp.transform(csp_te)

        # ── Step 3: Assemble features ──────────────────────────────────
        if FEATURE_MODE == "full":
            X_tr = np.concatenate([X_tab_tr, csp_tr], axis=1)
            X_te = np.concatenate([X_tab_te, csp_te], axis=1)
        elif FEATURE_MODE == "csp_only":
            X_tr, X_te = csp_tr, csp_te
        elif FEATURE_MODE == "tab_only":
            X_tr, X_te = X_tab_tr, X_tab_te
        else:
            raise ValueError(f"Unknown FEATURE_MODE: {FEATURE_MODE}")

        del X_tab_tr, X_tab_te, csp_tr, csp_te
        gc.collect()

        # ── Step 4: Feature selection ──────────────────────────────────
        if TOP_K_FEATURES > 0 and X_tr.shape[1] > TOP_K_FEATURES:
            selector = FMISelector(k=TOP_K_FEATURES)
            X_tr = selector.fit_transform(X_tr, y_tr)
            X_te = selector.transform(X_te)

        # ── Step 5: Classify ───────────────────────────────────────────
        clf = build_clf(classifier)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        acc = float((y_pred == y_te).mean())

        scores.append(acc)
        print(f"        Fold {fold}: acc={acc:.4f}")

        del X_tr, X_te, clf
        gc.collect()

    return np.array(scores)


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  SEED EEG — Binary Classification Pipeline (v10)")
    print("  Binary: Positive (1) vs Negative (-1)")
    print("  CV: WITHIN-SESSION (no session mixing)")
    print("  Scaling: Per-session training data only")
    print(f"  CV mode: {CV_MODE}")
    print(f"  Feature mode: {FEATURE_MODE}")
    print(f"  Folds per session: {N_FOLDS}")
    print("=" * 70)

    subjects = discover_subjects()
    print(f"\nFound {len(subjects)} subjects: {subjects}")

    all_results = []

    for clf_name in CLASSIFIERS:
        print(f"\n{'─' * 70}")
        print(f"  Classifier: {clf_name.upper()}")
        print(f"{'─' * 70}")

        subject_session_scores = []

        for sub in subjects:
            print(f"\n  Subject {sub}")
            sessions = discover_sessions(sub)

            if not sessions:
                print(f"    ✗ No sessions found")
                continue

            for sess in sessions:
                print(f"    Session {sess}")
                try:
                    raw_data, raw_idx, X_tab, y, trials = load_session(sub, sess)
                except (FileNotFoundError, ValueError) as exc:
                    print(f"      ✗ {exc}")
                    continue

                classes = np.unique(y).tolist()
                n_trials = len(np.unique(trials))

                print(f"      Windows: {len(y)} | Trials: {n_trials} | Classes: {classes}")

                if len(classes) < 2:
                    print(f"      ⚠ Only 1 class, skipping")
                    del raw_data, raw_idx, X_tab, y, trials
                    gc.collect()
                    continue

                try:
                    scores = cv_within_session(
                        raw_data, raw_idx, X_tab, y, trials,
                        n_folds=N_FOLDS,
                        classifier=clf_name,
                    )
                except Exception as exc:
                    print(f"      ✗ CV failed: {exc}")
                    import traceback
                    traceback.print_exc()
                    del raw_data, raw_idx, X_tab, y, trials
                    gc.collect()
                    continue

                if len(scores) > 0:
                    mean_acc = float(scores.mean())
                    subject_session_scores.append(mean_acc)
                    print(f"      → Session {sess} mean: {mean_acc:.4f}")

                    for fold_i, s in enumerate(scores, 1):
                        all_results.append({
                            "classifier": clf_name,
                            "subject": sub,
                            "session": sess,
                            "fold": fold_i,
                            "accuracy": round(float(s), 6),
                        })

                del raw_data, raw_idx, X_tab, y, trials, scores
                gc.collect()

        if subject_session_scores:
            arr = np.array(subject_session_scores)
            print(f"\n  {clf_name.upper()} across {len(arr)} subject-sessions — "
                  f"mean: {arr.mean():.4f} std: {arr.std():.4f} "
                  f"min: {arr.min():.4f} max: {arr.max():.4f}")

    # Save results
    out_path = Path("./results_within_session.csv")
    pd.DataFrame(all_results).to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")
    print("=" * 70)
