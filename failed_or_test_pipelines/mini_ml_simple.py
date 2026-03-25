"""
Mini ML Pipeline - Within-Session CV (proper approach)
Do CV within each session, aggregate results
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
import warnings

warnings.filterwarnings("ignore")

DATA_PATH = Path("./mini_data_bandpower")


def load_session(filepath):
    df = pd.read_parquet(filepath)
    feat_cols = [c for c in df.columns if c not in ('label','subject','session','trial_id')]
    X = df[feat_cols].values.astype(np.float32)
    y = df['label'].values.astype(int)
    trials = df['trial_id'].values
    return X, y, trials


def remove_zero_variance(X, threshold=1e-6):
    var = X.var(axis=0)
    mask = var > threshold
    return X[:, mask], mask


def cv_within_session(X, y, trials, n_folds=3):
    """CV within a single session."""
    unique_trials = np.unique(trials)
    n_folds = min(n_folds, len(unique_trials))

    if n_folds < 2:
        return []

    # Remove zero-var features
    X_clean, _ = remove_zero_variance(X)

    gkf = GroupKFold(n_splits=n_folds)
    scores = []

    for tr, te in gkf.split(X_clean, y, groups=trials):
        if len(np.unique(y[tr])) < 2:
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_clean[tr])
        X_te = scaler.transform(X_clean[te])

        clf = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced')
        clf.fit(X_tr, y[tr])
        acc = (clf.predict(X_te) == y[te]).mean()
        scores.append(acc)

    return scores


def main():
    print("=" * 60)
    print("Mini ML - Within-Session GroupKFold")
    print("=" * 60)

    all_scores = []

    for pf in sorted(DATA_PATH.glob("*.parquet")):
        sess_name = pf.stem
        X, y, trials = load_session(pf)

        print(f"\n{sess_name}: {len(y)} samples, {len(np.unique(trials))} trials")

        scores = cv_within_session(X, y, trials, n_folds=3)

        if scores:
            mean = np.mean(scores)
            print(f"  Folds: {[f'{s:.2f}' for s in scores]} → Mean: {mean:.4f}")
            all_scores.extend(scores)
        else:
            print("  Skipped (not enough data)")

    if all_scores:
        final = np.mean(all_scores)
        print(f"\n{'='*60}")
        print(f"OVERALL: {final:.4f} ± {np.std(all_scores):.4f}")
        print(f"{'='*60}")

        if final >= 0.7:
            print("\n✓ PASSED!")
        else:
            print(f"\n⚠ {final:.4f} < 0.7")
        return final

    return 0.0


if __name__ == "__main__":
    main()
