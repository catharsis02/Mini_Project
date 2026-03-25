"""
Mini ML Pipeline - FIXED VERSION
Cross-session CV (not within-session) + proper handling
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings

warnings.filterwarnings("ignore")

DATA_PATH = Path("./mini_data_stockwell")


def load_all_data():
    """Load all sessions and pool together for cross-session CV."""
    all_X = []
    all_y = []
    all_sessions = []
    all_trials = []

    for pf in sorted(DATA_PATH.glob("*.parquet")):
        df = pd.read_parquet(pf)

        feat_cols = [c for c in df.columns if c not in ('label', 'subject', 'session', 'trial_id')]
        X = df[feat_cols].values
        y = df['label'].values

        # Session identifier for grouping
        session_id = pf.stem  # e.g., "1_1", "1_2"

        all_X.append(X)
        all_y.append(y)
        all_sessions.append([session_id] * len(y))
        all_trials.append(df['trial_id'].values)

    return (
        np.vstack(all_X),
        np.hstack(all_y),
        np.hstack(all_sessions),
        np.hstack(all_trials)
    )


def remove_zero_variance(X, threshold=1e-6):
    """Remove features with near-zero variance."""
    var = X.var(axis=0)
    mask = var > threshold
    return X[:, mask], mask


def cross_session_cv(X, y, sessions, n_splits=3):
    """
    Cross-session CV: group by session, not by trial.
    This ensures we test generalization across different sessions.
    """
    # Remove zero-variance features
    X_clean, _ = remove_zero_variance(X)

    print(f"\nData: {len(y)} samples, {X_clean.shape[1]} features, {len(np.unique(sessions))} sessions")
    print(f"Class balance: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Use sessions as groups (not trials!)
    n_splits = min(n_splits, len(np.unique(sessions)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits)

    results = {"svm": [], "lda": []}

    for fold, (tr, te) in enumerate(sgkf.split(X_clean, y, groups=sessions), 1):
        print(f"\nFold {fold}:")
        print(f"  Train: {len(tr)} samples from {len(np.unique(sessions[tr]))} sessions")
        print(f"  Test:  {len(te)} samples from {len(np.unique(sessions[te]))} sessions")

        if len(np.unique(y[tr])) < 2:
            print(f"  Skipped: only 1 class in training")
            continue

        # Scale
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_clean[tr])
        X_te = scaler.transform(X_clean[te])

        # SVM
        svm = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced')
        svm.fit(X_tr, y[tr])
        acc_svm = (svm.predict(X_te) == y[te]).mean()
        results["svm"].append(acc_svm)
        print(f"  SVM: {acc_svm:.4f}")

        # LDA
        lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
        lda.fit(X_tr, y[tr])
        acc_lda = (lda.predict(X_te) == y[te]).mean()
        results["lda"].append(acc_lda)
        print(f"  LDA: {acc_lda:.4f}")

    return results


def main():
    print("=" * 60)
    print("Mini ML - FIXED: Cross-Session CV")
    print("=" * 60)

    X, y, sessions, trials = load_all_data()

    results = cross_session_cv(X, y, sessions, n_splits=3)

    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)

    all_scores = []
    for clf_name, scores in results.items():
        if scores:
            mean = np.mean(scores)
            std = np.std(scores)
            print(f"{clf_name.upper()}: {mean:.4f} ± {std:.4f} ({len(scores)} folds)")
            all_scores.extend(scores)

    if all_scores:
        final = np.mean(all_scores)
        print(f"\nFINAL ACCURACY: {final:.4f}")
        print("=" * 60)

        if final >= 0.6:
            print("\n✓ PASSED! Ready to scale up")
            return final
        else:
            print(f"\n⚠ {final:.4f} < 0.6 - still needs work")
            return final

    return 0.0


if __name__ == "__main__":
    main()
