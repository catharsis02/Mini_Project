"""
Mini ML Pipeline - FIXED V2: Cross-Session CV + CSP + RASM
"""

import gc
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
import warnings

warnings.filterwarnings("ignore")

DATA_PATH = Path("./mini_data_stockwell")
N_CSP = 4
DECIMATE = 5


def load_all_data():
    """Load all sessions: features + raw data for CSP."""
    all_X_feat = []
    all_X_raw = []
    all_y = []
    all_sessions = []

    for pf in sorted(DATA_PATH.glob("*.parquet")):
        df = pd.read_parquet(pf)
        raw_path = DATA_PATH / f"{pf.stem}_raw.npy"

        feat_cols = [c for c in df.columns if c not in ('label', 'subject', 'session', 'trial_id')]
        X_feat = df[feat_cols].values
        y = df['label'].values

        # Load raw as memmap
        X_raw = np.load(raw_path, mmap_mode='r')

        session_id = pf.stem

        all_X_feat.append(X_feat)
        all_X_raw.append(np.array(X_raw))  # Convert to array for stacking
        all_y.append(y)
        all_sessions.append([session_id] * len(y))

    return (
        np.vstack(all_X_feat),
        np.vstack(all_X_raw),
        np.hstack(all_y),
        np.hstack(all_sessions)
    )


def remove_zero_variance(X, threshold=1e-6):
    """Remove features with near-zero variance."""
    var = X.var(axis=0)
    mask = var > threshold
    return X[:, mask], mask


def cross_session_cv_with_csp(X_feat, X_raw, y, sessions, n_splits=3):
    """
    Cross-session CV with CSP features.
    """
    # Remove zero-variance features from tabular
    X_feat_clean, _ = remove_zero_variance(X_feat)

    print(f"\nData: {len(y)} samples, {X_feat_clean.shape[1]} features, {len(np.unique(sessions))} sessions")
    print(f"Raw shape: {X_raw.shape}")
    print(f"Class balance: {dict(zip(*np.unique(y, return_counts=True)))}")

    n_splits = min(n_splits, len(np.unique(sessions)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits)

    results = {"svm": [], "lda": []}

    for fold, (tr, te) in enumerate(sgkf.split(X_feat_clean, y, groups=sessions), 1):
        print(f"\nFold {fold}:")
        print(f"  Train: {len(tr)} samples")
        print(f"  Test:  {len(te)} samples")

        if len(np.unique(y[tr])) < 2:
            print(f"  Skipped: only 1 class")
            continue

        # ── Scale tabular features ──
        scaler_feat = StandardScaler()
        X_feat_tr = scaler_feat.fit_transform(X_feat_clean[tr])
        X_feat_te = scaler_feat.transform(X_feat_clean[te])

        # ── CSP on decimated raw data ──
        try:
            # Subsample for CSP fit (max 100 per class)
            max_per_cls = min(100, len(tr) // 2)
            tr_sub = []
            for cls in np.unique(y[tr]):
                cls_idx = tr[y[tr] == cls]
                if len(cls_idx) > max_per_cls:
                    cls_idx = np.random.choice(cls_idx, max_per_cls, replace=False)
                tr_sub.extend(cls_idx)
            tr_sub = np.array(tr_sub)

            # Decimate for speed
            X_raw_sub = X_raw[tr_sub][:, :, ::DECIMATE].astype(np.float32)

            csp = CSP(n_components=N_CSP, log=True, norm_trace=False, reg=0.01)
            csp.fit(X_raw_sub, (y[tr_sub] == 1).astype(int))
            del X_raw_sub
            gc.collect()

            # Transform train and test
            X_raw_tr = X_raw[tr][:, :, ::DECIMATE].astype(np.float32)
            csp_tr = csp.transform(X_raw_tr)
            del X_raw_tr

            X_raw_te = X_raw[te][:, :, ::DECIMATE].astype(np.float32)
            csp_te = csp.transform(X_raw_te)
            del X_raw_te, csp
            gc.collect()

            # Scale CSP
            scaler_csp = StandardScaler()
            csp_tr = scaler_csp.fit_transform(csp_tr)
            csp_te = scaler_csp.transform(csp_te)

            # Combine features
            X_tr = np.concatenate([X_feat_tr, csp_tr], axis=1)
            X_te = np.concatenate([X_feat_te, csp_te], axis=1)

            print(f"  Features: {X_feat_tr.shape[1]} (tabular) + {csp_tr.shape[1]} (CSP) = {X_tr.shape[1]}")

        except Exception as e:
            print(f"  CSP failed ({e}), using tabular only")
            X_tr, X_te = X_feat_tr, X_feat_te

        # ── Classify ──
        # SVM
        svm = SVC(kernel='rbf', C=100, gamma='scale', class_weight='balanced')
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

        del X_tr, X_te
        gc.collect()

    return results


def main():
    print("=" * 60)
    print("Mini ML - FIXED V2: Cross-Session CV + CSP + RASM")
    print("=" * 60)

    X_feat, X_raw, y, sessions = load_all_data()

    results = cross_session_cv_with_csp(X_feat, X_raw, y, sessions, n_splits=3)

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
            print(f"\n⚠ {final:.4f} < 0.6 - needs more work")
            return final

    return 0.0


if __name__ == "__main__":
    main()
