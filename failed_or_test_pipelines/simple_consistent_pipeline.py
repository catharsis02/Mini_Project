"""
Simple Consistent EEG Pipeline for Reliable Performance
=====================================================
Focus on simplicity, consistency and honest cross-validation

Strategy:
1. Simple but effective feature engineering
2. Conservative, well-validated classifiers
3. Subject-specific normalization
4. Strong but appropriate regularization

Target: Consistent 60%+ mean accuracy with low variance
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from mne.decoding import CSP
from scipy import signal
import gc
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ══════════════════════════════════════════════════════════════════════
# SIMPLE & STABLE CONFIG
# ══════════════════════════════════════════════════════════════════════

PROCESSED_PATH = Path("./data/SEED_EEG/processed_data/")

# Conservative settings for reliability
N_CSP_COMPONENTS = 3  # Minimal overfitting risk
TOP_K_FEATURES = 50   # Conservative feature count
N_FOLDS = 5
RANDOM_STATE = 42
DECIMATE_FACTOR = 3

# Conservative frequency bands
SIMPLE_FREQ_BANDS = {
    'alpha': (8, 12),   # Well-established alpha band
    'beta': (13, 25),   # Conservative beta
}

@dataclass
class SimpleResult:
    fold: int
    accuracy: float
    classifier: str
    subject: int
    session: int

# ══════════════════════════════════════════════════════════════════════
# SIMPLE FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════

class SimpleFeatureEngineering:
    """Simple, reliable feature engineering focused on consistency."""

    def __init__(self):
        # Simple preprocessing
        self.variance_filter = VarianceThreshold(threshold=0.001)  # Very minimal filtering
        self.scaler = RobustScaler()  # Robust to outliers
        self.csp_filters = {}
        self.feature_selector = SelectKBest(score_func=f_classif, k=TOP_K_FEATURES)

    def fit(self, X_tab: np.ndarray, X_raw: np.ndarray, y: np.ndarray, sfreq: float = 200/3):
        """Simple fit with minimal complexity."""
        print("      Fitting simple feature engineering...")

        # 1. Basic tabular feature processing
        X_tab_filtered = self.variance_filter.fit_transform(X_tab)

        # 2. Simple CSP
        self._fit_simple_csp(X_raw, y, sfreq)

        # 3. Extract simple feature set
        all_features = self._extract_simple_features(X_tab_filtered, X_raw, sfreq)

        # 4. Basic scaling
        self.scaler.fit(all_features)
        all_features_scaled = self.scaler.transform(all_features)

        # 5. Simple feature selection
        if all_features_scaled.shape[1] > TOP_K_FEATURES:
            self.feature_selector.fit(all_features_scaled, y)
        else:
            self.feature_selector = None

        return self

    def transform(self, X_tab: np.ndarray, X_raw: np.ndarray, sfreq: float = 200/3) -> np.ndarray:
        """Simple transform."""
        # Process features
        X_tab_filtered = self.variance_filter.transform(X_tab)
        all_features = self._extract_simple_features(X_tab_filtered, X_raw, sfreq)
        all_features_scaled = self.scaler.transform(all_features)

        # Feature selection if fitted
        if self.feature_selector is not None:
            final_features = self.feature_selector.transform(all_features_scaled)
        else:
            final_features = all_features_scaled

        return final_features

    def _fit_simple_csp(self, X_raw: np.ndarray, y: np.ndarray, sfreq: float):
        """Simple CSP with minimal bands."""
        print("        Fitting simple CSP...")

        for band_name, (low, high) in SIMPLE_FREQ_BANDS.items():
            try:
                # Simple butterworth filter
                sos = signal.butter(4, [low, high], btype='band', fs=sfreq, output='sos')
                X_filtered = signal.sosfiltfilt(sos, X_raw, axis=2)

                # Simple CSP
                csp = CSP(
                    n_components=N_CSP_COMPONENTS,
                    reg=None,  # No regularization for simplicity
                    log=True,
                    norm_trace=False
                )
                csp.fit(X_filtered, y)
                self.csp_filters[band_name] = csp

            except Exception as e:
                print(f"          CSP failed for {band_name}: {e}")

    def _extract_simple_features(self, X_tab: np.ndarray, X_raw: np.ndarray, sfreq: float) -> np.ndarray:
        """Extract minimal, reliable feature set."""
        features = []

        # 1. Processed tabular features
        features.append(X_tab)

        # 2. Simple CSP features
        for band_name, csp_filter in self.csp_filters.items():
            try:
                csp_features = csp_filter.transform(X_raw)
                features.append(csp_features)
            except:
                continue

        # 3. Basic band powers (only if have features)
        if len(features) < 2:  # Only add if CSP didn't work
            band_powers = self._extract_basic_powers(X_raw, sfreq)
            if band_powers.size > 0:
                features.append(band_powers)

        if not features:
            # Fallback to raw channel means
            features.append(np.mean(X_raw, axis=2))

        return np.hstack(features)

    def _extract_basic_powers(self, X_raw: np.ndarray, sfreq: float) -> np.ndarray:
        """Simple band power features."""
        n_samples, n_channels, n_times = X_raw.shape
        band_features = []

        # Use only first few channels
        max_channels = min(8, n_channels)

        for i in range(n_samples):
            sample_features = []

            for ch in range(0, max_channels, 2):  # Every other channel
                signal_data = X_raw[i, ch, :]

                # Simple variance in each band
                for band_name, (low, high) in SIMPLE_FREQ_BANDS.items():
                    try:
                        sos = signal.butter(3, [low, high], btype='band', fs=sfreq, output='sos')
                        filtered = signal.sosfiltfilt(sos, signal_data)
                        power = np.var(filtered)
                        sample_features.append(np.log(power + 1e-10))
                    except:
                        sample_features.append(0)

            band_features.append(sample_features)

        return np.array(band_features)

# ══════════════════════════════════════════════════════════════════════
# SIMPLE CLASSIFIERS
# ══════════════════════════════════════════════════════════════════════

def create_simple_classifiers():
    """Create simple, reliable classifiers."""
    classifiers = {}

    # Simple SVM
    classifiers['svm'] = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',
        random_state=RANDOM_STATE
    )

    # Conservative Random Forest
    classifiers['rf'] = RandomForestClassifier(
        n_estimators=50,
        max_depth=6,
        min_samples_split=10,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # Simple LDA
    classifiers['lda'] = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.1)

    # Ridge classifier
    classifiers['ridge'] = RidgeClassifier(
        alpha=1.0,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )

    return classifiers

# ══════════════════════════════════════════════════════════════════════
# SIMPLIFIED PIPELINE
# ══════════════════════════════════════════════════════════════════════

def discover_sessions() -> List[Tuple[int, int]]:
    """Find available sessions."""
    sessions = []
    for pf in PROCESSED_PATH.glob("*.parquet"):
        try:
            parts = pf.stem.split("_")
            if len(parts) >= 2:
                subject = int(parts[0])
                session = int(parts[1])
                raw_file = PROCESSED_PATH / f"{pf.stem}_raw.npy"
                if raw_file.exists():
                    sessions.append((subject, session))
        except (ValueError, IndexError):
            continue
    return sorted(sessions)

def load_session_data_simple(subject: int, session: int) -> Optional[Tuple]:
    """Simple, robust data loading."""
    pf = PROCESSED_PATH / f"{subject}_{session}.parquet"
    raw_file = PROCESSED_PATH / f"{subject}_{session}_raw.npy"

    if not pf.exists() or not raw_file.exists():
        return None

    try:
        df = pd.read_parquet(pf)
        feat_cols = [c for c in df.columns if c not in ('label', 'subject', 'session', 'trial_id')]

        X_tab = df[feat_cols].values.astype(np.float32)
        y = df['label'].values
        trials = df['trial_id'].values

        # Simple binary filtering
        binary_mask = (y != 0) & ~pd.isna(y)
        if not binary_mask.any():
            return None

        X_tab = X_tab[binary_mask]
        y = y[binary_mask]
        trials = trials[binary_mask]

        # Clean data
        X_tab = np.nan_to_num(X_tab, nan=0, posinf=0, neginf=0)

        # Load raw data simply
        X_raw = np.load(raw_file, mmap_mode='r')
        X_raw = X_raw[binary_mask]
        X_raw = X_raw[:, :, ::DECIMATE_FACTOR].astype(np.float32)
        X_raw = np.nan_to_num(X_raw, nan=0, posinf=0, neginf=0)

        return X_tab, X_raw, y, trials, subject, session

    except Exception as e:
        print(f"  Error loading {subject}_{session}: {e}")
        return None

def simple_cv_per_session(
    X_tab: np.ndarray, X_raw: np.ndarray, y: np.ndarray, trials: np.ndarray,
    subject: int, session: int, sfreq: float = 200/3
) -> List[SimpleResult]:
    """Simple cross-validation focused on reliability."""

    unique_trials = np.unique(trials)
    n_trials = len(unique_trials)

    if n_trials < N_FOLDS:
        n_splits = max(2, min(n_trials, 3))
    else:
        n_splits = N_FOLDS

    gkf = GroupKFold(n_splits=n_splits)
    results = []

    print(f"    Simple {n_splits}-fold CV")

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_tab, y, groups=trials), 1):
        # Verify no leakage
        train_trials = set(trials[train_idx])
        test_trials = set(trials[test_idx])

        if train_trials.intersection(test_trials):
            continue

        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        print(f"    Fold {fold}: {len(train_idx)} train, {len(test_idx)} test")

        try:
            # Simple feature engineering
            feat_eng = SimpleFeatureEngineering()
            feat_eng.fit(X_tab[train_idx], X_raw[train_idx], y_train, sfreq)

            X_train_final = feat_eng.transform(X_tab[train_idx], X_raw[train_idx], sfreq)
            X_test_final = feat_eng.transform(X_tab[test_idx], X_raw[test_idx], sfreq)

            print(f"      Simple features: {X_train_final.shape[1]}")

            # Test simple models
            models = create_simple_classifiers()

            for model_name, clf in models.items():
                try:
                    clf.fit(X_train_final, y_train)
                    y_pred = clf.predict(X_test_final)
                    accuracy = accuracy_score(y_test, y_pred)

                    result = SimpleResult(
                        fold=fold,
                        accuracy=accuracy,
                        classifier=model_name,
                        subject=subject,
                        session=session
                    )

                    results.append(result)
                    print(f"      {model_name}: {accuracy:.4f}")

                except Exception as e:
                    print(f"      {model_name} failed: {e}")

        except Exception as e:
            print(f"    Fold {fold} failed: {e}")

        gc.collect()

    return results

def main():
    """Run simple, consistent pipeline."""
    print("="*70)
    print("SIMPLE CONSISTENT EEG PIPELINE")
    print("Priority: Reliable performance and honest CV")
    print("Target: Consistent 60%+ mean accuracy")
    print("="*70)

    sessions = discover_sessions()
    print(f"Found {len(sessions)} sessions")

    if not sessions:
        print("No data found!")
        return []

    all_results = []
    target_sessions = min(10, len(sessions))  # Start with smaller test

    for subject, session in sessions[:target_sessions]:
        print(f"\nProcessing Subject {subject}, Session {session}")

        session_data = load_session_data_simple(subject, session)
        if session_data is None:
            continue

        X_tab, X_raw, y, trials, _, _ = session_data

        if len(np.unique(y)) < 2:
            continue

        session_results = simple_cv_per_session(
            X_tab, X_raw, y, trials, subject, session
        )

        if session_results:
            all_results.extend(session_results)

            # Session feedback
            session_accs = [r.accuracy for r in session_results]
            if session_accs:
                mean_acc = np.mean(session_accs)
                print(f"  Session Mean: {mean_acc:.4f}")

        del X_tab, X_raw, y, trials
        gc.collect()

    # Results analysis
    print(f"\n{'='*70}")
    print("SIMPLE PIPELINE RESULTS")
    print(f"{'='*70}")

    if all_results:
        df = pd.DataFrame([{
            'classifier': r.classifier,
            'subject': r.subject,
            'session': r.session,
            'fold': r.fold,
            'accuracy': r.accuracy
        } for r in all_results])

        print("\nOverall Performance:")
        overall_mean = df['accuracy'].mean()
        overall_std = df['accuracy'].std()
        print(f"Mean Accuracy: {overall_mean:.4f} ± {overall_std:.4f}")

        # By classifier
        print("\nBy Classifier:")
        for model in df['classifier'].unique():
            model_data = df[df['classifier'] == model]
            accuracies = model_data['accuracy'].values

            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)

            print(f"{model:>8}: {mean_acc:.4f} ± {std_acc:.4f}")

        # Save results
        df.to_csv('simple_consistent_results.csv', index=False)
        print(f"\nResults saved → simple_consistent_results.csv")

        # Improvement assessment
        if overall_mean >= 0.60:
            print(f"\n✅ SUCCESS: {overall_mean:.1%} mean accuracy achieved")
            print("   Better than baseline ~50% with honest CV")
        elif overall_mean >= 0.55:
            print(f"\n📈 PROGRESS: {overall_mean:.1%} improvement over baseline")
        else:
            print(f"\n📊 BASELINE: {overall_mean:.1%} (similar to current)")

        # Stability assessment
        if overall_std <= 0.10:
            print(f"   High stability: {overall_std:.3f} standard deviation")
        elif overall_std <= 0.15:
            print(f"   Good stability: {overall_std:.3f} standard deviation")
        else:
            print(f"   Variable performance: {overall_std:.3f} standard deviation")

    return all_results

if __name__ == "__main__":
    results = main()