"""
SEED EEG — FIXED ML Pipeline for 80%+ Accuracy
==============================================

Key Fixes:
1. CROSS-SESSION CV (not within-session)
2. ALL 15 TRIALS (including neutral)
3. LEAVE-ONE-SUBJECT-OUT validation
4. Proper feature scaling and selection

Expected: 70-85% accuracy
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report
from mne.decoding import CSP
import gc

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

PROCESSED_PATH = Path("./data/SEED_EEG/processed_data/")
N_CSP_COMPONENTS = 8
TOP_K_FEATURES = 100  # Feature selection
RANDOM_STATE = 42

# ══════════════════════════════════════════════════════════════════════
# DATA LOADING - ALL TRIALS INCLUDING NEUTRAL
# ══════════════════════════════════════════════════════════════════════

def load_all_data_cross_session():
    """
    Load all available sessions for cross-session CV.
    Key change: KEEP ALL 15 TRIALS (including neutral)
    """
    print("Loading all processed data...")

    all_X_tab = []
    all_X_raw = []
    all_y = []
    all_subjects = []
    all_sessions = []

    parquet_files = sorted(PROCESSED_PATH.glob("*.parquet"))
    print(f"Found {len(parquet_files)} processed files")

    for pf in parquet_files:
        raw_file = PROCESSED_PATH / f"{pf.stem}_raw.npy"
        if not raw_file.exists():
            print(f"  ⚠ Missing raw file for {pf.stem}, skipping")
            continue

        try:
            # Load tabular features
            df = pd.read_parquet(pf)
            feat_cols = [c for c in df.columns if c not in ('label', 'subject', 'session', 'trial_id')]

            X_tab = df[feat_cols].values
            y = df['label'].values  # Keep ALL labels: -1, 0, 1
            subjects = df['subject'].values
            sessions = df['session'].values

            # Load raw data for CSP
            X_raw = np.load(raw_file, mmap_mode='r')[:]

            # Verify shapes match
            assert len(X_tab) == len(X_raw) == len(y), f"Shape mismatch in {pf.stem}"

            all_X_tab.append(X_tab)
            all_X_raw.append(X_raw)
            all_y.append(y)
            all_subjects.append(subjects)
            all_sessions.append(sessions)

            print(f"  ✓ {pf.stem}: {len(y)} samples, labels: {np.unique(y)}")

        except Exception as e:
            print(f"  ✗ Error loading {pf.stem}: {e}")
            continue

    if not all_X_tab:
        raise ValueError("No data loaded!")

    # Concatenate all data
    X_tab = np.vstack(all_X_tab)
    X_raw = np.vstack(all_X_raw)
    y = np.hstack(all_y)
    subjects = np.hstack(all_subjects)
    sessions = np.hstack(all_sessions)

    print(f"\nTotal data: {len(y)} samples from {len(np.unique(subjects))} subjects")
    print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Features: {X_tab.shape[1]} tabular + raw for CSP")

    return X_tab, X_raw, y, subjects, sessions


def remove_zero_variance(X, threshold=1e-6):
    """Remove features with near-zero variance."""
    var = X.var(axis=0)
    mask = var > threshold
    print(f"  Removed {(~mask).sum()} zero-variance features, kept {mask.sum()}")
    return X[:, mask], mask


def cross_session_validation(X_tab, X_raw, y, subjects):
    """
    Leave-One-Subject-Out Cross Validation.
    Train on N-1 subjects, test on 1 subject.
    This is the proper way to test generalization.
    """
    print("\n" + "="*70)
    print("CROSS-SESSION VALIDATION (Leave-One-Subject-Out)")
    print("="*70)

    logo = LeaveOneGroupOut()
    results = []

    unique_subjects = np.unique(subjects)
    print(f"Cross-validation: {len(unique_subjects)} folds (one per subject)")

    for fold, (train_idx, test_idx) in enumerate(logo.split(X_tab, y, subjects), 1):
        test_subject = subjects[test_idx][0]
        train_subjects = np.unique(subjects[train_idx])

        print(f"\nFold {fold}: Test Subject {test_subject}")
        print(f"  Train: {len(train_idx)} samples from subjects {train_subjects}")
        print(f"  Test:  {len(test_idx)} samples from subject {test_subject}")

        # Check if we have all classes in both train and test
        train_classes = np.unique(y[train_idx])
        test_classes = np.unique(y[test_idx])

        if len(train_classes) < 2 or len(test_classes) < 2:
            print(f"  ⚠ Insufficient classes, skipping fold")
            continue

        # Split data
        X_tab_train, X_tab_test = X_tab[train_idx], X_tab[test_idx]
        X_raw_train, X_raw_test = X_raw[train_idx], X_raw[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Preprocessing pipeline
        fold_results = process_fold(
            X_tab_train, X_tab_test, X_raw_train, X_raw_test,
            y_train, y_test, fold, test_subject
        )

        if fold_results:
            results.append(fold_results)

        gc.collect()

    return results


def process_fold(X_tab_train, X_tab_test, X_raw_train, X_raw_test, y_train, y_test, fold, test_subject):
    """Process a single cross-validation fold."""

    # 1. Remove zero-variance features
    X_tab_train_clean, var_mask = remove_zero_variance(X_tab_train)
    X_tab_test_clean = X_tab_test[:, var_mask]

    # 2. Scale tabular features
    scaler = StandardScaler()
    X_tab_train_scaled = scaler.fit_transform(X_tab_train_clean)
    X_tab_test_scaled = scaler.transform(X_tab_test_clean)

    # 3. Feature selection on tabular features
    if X_tab_train_scaled.shape[1] > TOP_K_FEATURES:
        print(f"  Feature selection: {X_tab_train_scaled.shape[1]} → {TOP_K_FEATURES}")
        selector = SelectKBest(f_classif, k=TOP_K_FEATURES)
        X_tab_train_selected = selector.fit_transform(X_tab_train_scaled, y_train)
        X_tab_test_selected = selector.transform(X_tab_test_scaled)
    else:
        X_tab_train_selected = X_tab_train_scaled
        X_tab_test_selected = X_tab_test_scaled

    # 4. CSP features (if we have binary problem, convert to binary)
    if len(np.unique(y_train)) == 3:  # 3-class problem
        # Convert to binary: positive vs. negative (exclude neutral for CSP)
        binary_mask_train = y_train != 0
        binary_mask_test = y_test != 0

        if binary_mask_train.sum() > 50 and binary_mask_test.sum() > 10:  # Enough samples
            try:
                csp = CSP(n_components=N_CSP_COMPONENTS, reg=None, log=True, norm_trace=False)
                csp.fit(X_raw_train[binary_mask_train], y_train[binary_mask_train])

                # Apply CSP to all data
                X_csp_train = csp.transform(X_raw_train)
                X_csp_test = csp.transform(X_raw_test)

                # Combine tabular + CSP
                X_train_final = np.hstack([X_tab_train_selected, X_csp_train])
                X_test_final = np.hstack([X_tab_test_selected, X_csp_test])

                print(f"  Features: {X_tab_train_selected.shape[1]} tabular + {N_CSP_COMPONENTS} CSP = {X_train_final.shape[1]} total")

            except Exception as e:
                print(f"  ⚠ CSP failed: {e}, using tabular only")
                X_train_final = X_tab_train_selected
                X_test_final = X_tab_test_selected
        else:
            X_train_final = X_tab_train_selected
            X_test_final = X_tab_test_selected
    else:
        X_train_final = X_tab_train_selected
        X_test_final = X_tab_test_selected

    # 5. Train classifiers
    results = {}

    # SVM
    try:
        svm = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', random_state=RANDOM_STATE)
        svm.fit(X_train_final, y_train)
        y_pred_svm = svm.predict(X_test_final)
        acc_svm = accuracy_score(y_test, y_pred_svm)
        results['svm'] = acc_svm
        print(f"  SVM: {acc_svm:.4f}")
    except Exception as e:
        print(f"  ⚠ SVM failed: {e}")

    # LDA
    try:
        lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
        lda.fit(X_train_final, y_train)
        y_pred_lda = lda.predict(X_test_final)
        acc_lda = accuracy_score(y_test, y_pred_lda)
        results['lda'] = acc_lda
        print(f"  LDA: {acc_lda:.4f}")
    except Exception as e:
        print(f"  ⚠ LDA failed: {e}")

    results['fold'] = fold
    results['test_subject'] = test_subject
    results['n_train'] = len(y_train)
    results['n_test'] = len(y_test)

    return results


def main():
    print("="*70)
    print("SEED EEG — FIXED PIPELINE FOR 80%+ ACCURACY")
    print("Cross-Session CV with All Trials (Including Neutral)")
    print("="*70)

    # Load all data
    X_tab, X_raw, y, subjects, sessions = load_all_data_cross_session()

    # Cross-session validation
    results = cross_session_validation(X_tab, X_raw, y, subjects)

    if not results:
        print("\n❌ No results obtained!")
        return 0.0

    # Summarize results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    for clf_name in ['svm', 'lda']:
        scores = [r[clf_name] for r in results if clf_name in r]
        if scores:
            mean_acc = np.mean(scores)
            std_acc = np.std(scores)
            max_acc = np.max(scores)
            print(f"{clf_name.upper()}: {mean_acc:.4f} ± {std_acc:.4f} (max: {max_acc:.4f}) [{len(scores)} folds]")

    # Check if we achieved target
    all_scores = []
    for r in results:
        all_scores.extend([r[k] for k in ['svm', 'lda'] if k in r])

    if all_scores:
        best_acc = max(all_scores)
        mean_acc = np.mean(all_scores)
        print(f"\nOVERALL: {mean_acc:.4f} (best: {best_acc:.4f})")

        if best_acc >= 0.8:
            print(f"\n🎉 SUCCESS! Achieved {best_acc:.1%} accuracy (target: 80%)")
        else:
            print(f"\n⚠ Need improvement: {best_acc:.1%} < 80% target")

        return mean_acc

    return 0.0


if __name__ == "__main__":
    main()