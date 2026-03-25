"""
Data Leakage Comparison: StratifiedKFold vs GroupKFold
====================================================
This script demonstrates the difference between:
- StratifiedKFold (data leakage) - windows from same trial can be in train/test
- GroupKFold (honest) - trials never mixed between train/test

Shows why honest CV gives lower but more realistic accuracy.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import gc

PROCESSED_PATH = Path("./data/SEED_EEG/processed_data/")
RANDOM_STATE = 42

def load_single_session_for_comparison(subject: int = 1, session: int = 1):
    """Load a single session to demonstrate the difference."""
    pf = PROCESSED_PATH / f"{subject}_{session}.parquet"

    if not pf.exists():
        raise FileNotFoundError(f"File not found: {pf}")

    df = pd.read_parquet(pf)
    feat_cols = [c for c in df.columns if c not in ('label', 'subject', 'session', 'trial_id')]

    X = df[feat_cols].values
    y = df['label'].values
    trials = df['trial_id'].values

    # Filter to binary classification
    binary_mask = y != 0
    X = X[binary_mask]
    y = y[binary_mask]
    trials = trials[binary_mask]

    print(f"Session {subject}_{session}:")
    print(f"  Total windows: {len(y)}")
    print(f"  Unique trials: {len(np.unique(trials))}")
    print(f"  Classes: {np.unique(y)}")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    return X, y, trials

def leaky_stratified_cv(X, y, trials, n_splits=5):
    """
    LEAKY Cross-Validation using StratifiedKFold.

    Problem: Windows from the same trial can appear in both training and test sets.
    This leads to inflated accuracy because the model learns trial-specific patterns.
    """
    print(f"\n{'='*50}")
    print("LEAKY CROSS-VALIDATION (StratifiedKFold)")
    print("⚠ Windows from same trial can be in train AND test")
    print(f"{'='*50}")

    # Feature selection and scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    selector = SelectKBest(f_classif, k=min(50, X_scaled.shape[1]))
    X_selected = selector.fit_transform(X_scaled, y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_selected, y), 1):
        # Check for trial leakage
        train_trials = set(trials[train_idx])
        test_trials = set(trials[test_idx])
        shared_trials = train_trials.intersection(test_trials)

        print(f"  Fold {fold}:")
        print(f"    Train: {len(train_idx)} windows, {len(train_trials)} trials")
        print(f"    Test:  {len(test_idx)} windows, {len(test_trials)} trials")
        print(f"    🚨 LEAKAGE: {len(shared_trials)} trials appear in BOTH train and test")

        # Train classifier
        clf = LinearDiscriminantAnalysis()
        clf.fit(X_selected[train_idx], y[train_idx])

        y_pred = clf.predict(X_selected[test_idx])
        accuracy = accuracy_score(y[test_idx], y_pred)
        scores.append(accuracy)

        print(f"    Accuracy: {accuracy:.4f} (inflated due to leakage)")

    mean_acc = np.mean(scores)
    std_acc = np.std(scores)
    print(f"\nLEAKY Results: {mean_acc:.4f} ± {std_acc:.4f}")
    print("⚠ This accuracy is ARTIFICIALLY HIGH due to data leakage!")

    return scores

def honest_group_cv(X, y, trials, n_splits=5):
    """
    HONEST Cross-Validation using GroupKFold.

    Solution: Trials are never mixed between training and test sets.
    This gives the true generalization performance.
    """
    print(f"\n{'='*50}")
    print("HONEST CROSS-VALIDATION (GroupKFold)")
    print("✅ Trials NEVER mixed between train and test")
    print(f"{'='*50}")

    # Feature selection and scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    selector = SelectKBest(f_classif, k=min(50, X_scaled.shape[1]))
    X_selected = selector.fit_transform(X_scaled, y)

    unique_trials = np.unique(trials)
    n_splits = min(n_splits, len(unique_trials))

    gkf = GroupKFold(n_splits=n_splits)
    scores = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_selected, y, groups=trials), 1):
        # Verify no trial leakage
        train_trials = set(trials[train_idx])
        test_trials = set(trials[test_idx])
        shared_trials = train_trials.intersection(test_trials)

        print(f"  Fold {fold}:")
        print(f"    Train: {len(train_idx)} windows, {len(train_trials)} trials")
        print(f"    Test:  {len(test_idx)} windows, {len(test_trials)} trials")

        if shared_trials:
            print(f"    🚨 ERROR: {len(shared_trials)} trials leaked!")
            continue
        else:
            print(f"    ✅ NO LEAKAGE: Completely separate trials")

        # Train classifier
        clf = LinearDiscriminantAnalysis()
        clf.fit(X_selected[train_idx], y[train_idx])

        y_pred = clf.predict(X_selected[test_idx])
        accuracy = accuracy_score(y[test_idx], y_pred)
        scores.append(accuracy)

        print(f"    Accuracy: {accuracy:.4f} (honest performance)")

    mean_acc = np.mean(scores)
    std_acc = np.std(scores)
    print(f"\nHONEST Results: {mean_acc:.4f} ± {std_acc:.4f}")
    print("✅ This accuracy reflects TRUE generalization ability!")

    return scores

def main():
    print("="*70)
    print("DATA LEAKAGE DEMONSTRATION")
    print("Comparing StratifiedKFold (leaky) vs GroupKFold (honest)")
    print("="*70)

    try:
        # Load data
        X, y, trials = load_single_session_for_comparison(subject=1, session=1)

        # Demonstrate leaky CV
        leaky_scores = leaky_stratified_cv(X, y, trials)

        # Demonstrate honest CV
        honest_scores = honest_group_cv(X, y, trials)

        # Compare results
        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}")

        leaky_mean = np.mean(leaky_scores)
        honest_mean = np.mean(honest_scores)
        inflation = ((leaky_mean - honest_mean) / honest_mean) * 100

        print(f"Leaky StratifiedKFold:  {leaky_mean:.4f} ± {np.std(leaky_scores):.4f}")
        print(f"Honest GroupKFold:      {honest_mean:.4f} ± {np.std(honest_scores):.4f}")
        print(f"Accuracy inflation:     +{inflation:.1f}% due to data leakage")

        print(f"\n📊 KEY INSIGHTS:")
        print(f"• StratifiedKFold allows windows from same trial in train AND test")
        print(f"• This creates data leakage → artificially high accuracy")
        print(f"• GroupKFold prevents trial mixing → honest performance")
        print(f"• The honest accuracy ({honest_mean:.1%}) is the true performance")
        print(f"• Use GroupKFold to get realistic estimates for EEG classification")

        # Save comparison results
        results_df = pd.DataFrame({
            'method': ['leaky_stratified'] * len(leaky_scores) + ['honest_group'] * len(honest_scores),
            'fold': list(range(1, len(leaky_scores)+1)) + list(range(1, len(honest_scores)+1)),
            'accuracy': leaky_scores + honest_scores
        })

        results_df.to_csv('leaky_vs_honest_comparison.csv', index=False)
        print(f"\n💾 Results saved to: leaky_vs_honest_comparison.csv")

    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Please ensure you have processed data in ./data/SEED_EEG/processed_data/")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()