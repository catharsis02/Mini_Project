"""
Mini ML Pipeline - Memory-safe within-session CV
Target: accuracy > 0.7
"""

import gc
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
DATA_PATH = Path("./mini_data_stockwell")

N_CSP = 4  # Reduced
N_FOLDS = 3  # Reduced for small data
DECIMATE = 5  # More aggressive decimation


# ─────────────────────────────
# LOAD SESSION (Memory-safe)
# ─────────────────────────────
def load_session(subject, session):
    """Load one session."""
    pf = DATA_PATH / f"{subject}_{session}.parquet"
    raw_path = DATA_PATH / f"{subject}_{session}_raw.npy"

    if not pf.exists() or not raw_path.exists():
        raise FileNotFoundError(f"Missing {subject}_{session}")

    df = pd.read_parquet(pf)
    y = df["label"].values.astype(int)
    trials = df["trial_id"].values

    feat_cols = [c for c in df.columns if c not in ("label", "subject", "session", "trial_id")]
    X_tab = df[feat_cols].values.astype(np.float32)

    # Load raw as memmap to save RAM
    X_raw = np.load(raw_path, mmap_mode='r')

    return X_raw, X_tab, y, trials


# ─────────────────────────────
# CV WITHIN SESSION (Memory-safe)
# ─────────────────────────────
def cv_session(X_raw, X_tab, y, trials, clf_type="svm"):
    """
    Within-session CV with GroupKFold.
    Memory-safe: use memmapped raw, process in chunks.
    """
    unique_trials = np.unique(trials)
    n_folds = min(N_FOLDS, len(unique_trials))

    if n_folds < 2:
        print(f"    Only {len(unique_trials)} trial(s), skip")
        return []

    gkf = GroupKFold(n_splits=n_folds)
    scores = []

    for fold, (tr, te) in enumerate(gkf.split(np.arange(len(y)), y, groups=trials), 1):
        y_tr, y_te = y[tr], y[te]

        if len(np.unique(y_tr)) < 2:
            print(f"    Fold {fold}: only 1 class, skip")
            continue

        # ── Scale tabular features ──
        scaler = StandardScaler()
        X_tab_tr = scaler.fit_transform(X_tab[tr])
        X_tab_te = scaler.transform(X_tab[te])

        # ── CSP (load raw data only when needed) ──
        try:
            # Subsample for CSP fit
            max_per_cls = min(100, len(tr) // 2)
            tr_sub = []
            for cls in np.unique(y_tr):
                cls_idx = tr[y_tr == cls]
                if len(cls_idx) > max_per_cls:
                    cls_idx = np.random.choice(cls_idx, max_per_cls, replace=False)
                tr_sub.extend(cls_idx)
            tr_sub = np.array(tr_sub)

            # Load only needed raw data, decimated
            X_raw_sub = np.array(X_raw[tr_sub])[:, :, ::DECIMATE].astype(np.float32)

            csp = CSP(n_components=N_CSP, log=True, norm_trace=False, reg=0.01)
            y_sub = y[tr_sub]
            csp.fit(X_raw_sub, (y_sub == 1).astype(int))
            del X_raw_sub
            gc.collect()

            # Transform train
            X_raw_tr = np.array(X_raw[tr])[:, :, ::DECIMATE].astype(np.float32)
            csp_tr = csp.transform(X_raw_tr)
            del X_raw_tr
            gc.collect()

            # Transform test
            X_raw_te = np.array(X_raw[te])[:, :, ::DECIMATE].astype(np.float32)
            csp_te = csp.transform(X_raw_te)
            del X_raw_te, csp
            gc.collect()

            # Scale CSP
            scaler_csp = StandardScaler()
            csp_tr = scaler_csp.fit_transform(csp_tr)
            csp_te = scaler_csp.transform(csp_te)

            # Combine
            X_tr = np.concatenate([X_tab_tr, csp_tr], axis=1)
            X_te = np.concatenate([X_tab_te, csp_te], axis=1)

            del csp_tr, csp_te

        except Exception as e:
            print(f"    Fold {fold}: CSP failed ({e}), tab only")
            X_tr, X_te = X_tab_tr, X_tab_te

        del X_tab_tr, X_tab_te
        gc.collect()

        # ── Classify ──
        if clf_type == "svm":
            clf = SVC(kernel="rbf", C=100, gamma="scale", class_weight="balanced")
        else:
            clf = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")

        clf.fit(X_tr, y_tr)
        acc = (clf.predict(X_te) == y_te).mean()
        scores.append(acc)
        print(f"    Fold {fold}: acc={acc:.4f}")

        del X_tr, X_te, clf
        gc.collect()

    return scores


# ─────────────────────────────
# DISCOVER
# ─────────────────────────────
def discover_data():
    data = []
    for pf in DATA_PATH.glob("*.parquet"):
        parts = pf.stem.split("_")
        if len(parts) == 2:
            data.append((int(parts[0]), int(parts[1])))
    return sorted(data)


# ─────────────────────────────
# MAIN
# ─────────────────────────────
def main():
    print("=" * 60)
    print("Mini ML Pipeline - Within-Session CV (Memory-safe)")
    print("=" * 60)

    sessions = discover_data()
    print(f"Found {len(sessions)} session files: {sessions}")

    if not sessions:
        print("No data! Run mini_preprocess_stockwell.py first.")
        return 0.0

    all_scores = []

    for clf_type in ["svm", "lda"]:
        print(f"\n{'─'*60}")
        print(f"Classifier: {clf_type.upper()}")
        print(f"{'─'*60}")

        clf_scores = []

        for sub, sess in sessions:
            print(f"\n  Subject {sub}, Session {sess}")
            try:
                X_raw, X_tab, y, trials = load_session(sub, sess)
            except FileNotFoundError as e:
                print(f"    ✗ {e}")
                continue

            print(f"    Windows: {len(y)}, Classes: {np.unique(y).tolist()}")

            scores = cv_session(X_raw, X_tab, y, trials, clf_type)

            if scores:
                mean = np.mean(scores)
                clf_scores.extend(scores)
                print(f"    → Mean: {mean:.4f}")

            del X_raw, X_tab
            gc.collect()

        if clf_scores:
            overall = np.mean(clf_scores)
            print(f"\n  {clf_type.upper()} Overall: {overall:.4f} ({len(clf_scores)} folds)")
            all_scores.extend(clf_scores)

    if all_scores:
        final = np.mean(all_scores)
        print(f"\n{'='*60}")
        print(f"FINAL ACCURACY: {final:.4f}")
        print(f"{'='*60}")
        return final
    return 0.0


if __name__ == "__main__":
    accuracy = main()

    if accuracy < 0.7:
        print(f"\n⚠ Accuracy {accuracy:.4f} < 0.7 - needs investigation!")
    else:
        print(f"\n✓ Accuracy {accuracy:.4f} >= 0.7 - PASSED!")
