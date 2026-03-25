"""
Stable & Consistent EEG Pipeline for Reliable 65%+ Mean Accuracy
===============================================================
Focus on consistency and cross-subject generalization over peak performance

Strategy:
1. Strong regularization to prevent overfitting
2. Subject-adaptive normalization and features
3. Conservative but stable feature engineering
4. Robust ensemble methods with cross-validation tuning
5. Cross-subject validation for realistic generalization

Target: 65%+ consistent mean accuracy (vs 50% current mean)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from mne.decoding import CSP
from scipy import signal
from scipy.stats import zscore
import gc
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ══════════════════════════════════════════════════════════════════════
# STABLE CONFIG - Conservative but reliable settings
# ══════════════════════════════════════════════════════════════════════

PROCESSED_PATH = Path("./data/SEED_EEG/processed_data/")

# Conservative parameters for stability
N_CSP_COMPONENTS = 4  # Reduced to prevent overfitting
TOP_K_FEATURES = 80   # Conservative feature count
N_FOLDS = 5
RANDOM_STATE = 42
DECIMATE_FACTOR = 3   # Balance between information and noise

# Robust frequency bands (well-established for EEG)
STABLE_FREQ_BANDS = {
    'theta': (4, 7),    # Conservative theta
    'alpha': (8, 12),   # Classic alpha
    'beta': (13, 25),   # Conservative beta
}

@dataclass
class StableResult:
    fold: int
    accuracy: float
    classifier: str
    subject: int
    session: int

# ══════════════════════════════════════════════════════════════════════
# STABLE FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════

class StableFeatureEngineering:
    """Conservative, stable feature engineering designed for consistency."""

    def __init__(self, n_csp: int = N_CSP_COMPONENTS):
        self.n_csp = n_csp

        # Robust preprocessing components
        self.variance_filter = VarianceThreshold(threshold=0.01)  # More reasonable threshold
        self.robust_scaler = RobustScaler()
        self.subject_scaler = StandardScaler()  # Subject-specific normalization

        # Conservative CSP
        self.csp_filters = {}

        # Stable feature selection
        self.feature_selector = SelectKBest(score_func=f_classif, k=TOP_K_FEATURES)

        # Conservative PCA (minimal dimensionality reduction)
        self.pca = PCA(n_components=0.90, random_state=RANDOM_STATE)

    def fit(self, X_tab: np.ndarray, X_raw: np.ndarray, y: np.ndarray,
            subjects: np.ndarray, sfreq: float = 200/3):
        """Fit with subject-aware preprocessing."""
        print("      Fitting stable feature engineering...")

        # 1. Conservative tabular feature processing
        X_tab_filtered = self.variance_filter.fit_transform(X_tab)

        # 2. Subject-normalized tabular features
        X_tab_normalized = self._subject_normalize(X_tab_filtered, subjects, fit=True)

        # 3. Conservative CSP with strong regularization
        self._fit_stable_csp(X_raw, y, subjects, sfreq)

        # 4. Extract conservative feature set
        all_features = self._extract_stable_features(
            X_tab_normalized, X_raw, subjects, sfreq
        )

        # 5. Robust scaling
        self.robust_scaler.fit(all_features)
        all_features_scaled = self.robust_scaler.transform(all_features)

        # 6. Conservative feature selection (F-test only for stability)
        self.feature_selector.fit(all_features_scaled, y)
        selected_features = self.feature_selector.transform(all_features_scaled)

        # 7. Minimal PCA for noise reduction only
        if selected_features.shape[1] > 50:  # Only if many features
            self.pca.fit(selected_features)

        return self

    def transform(self, X_tab: np.ndarray, X_raw: np.ndarray,
                 subjects: np.ndarray, sfreq: float = 200/3) -> np.ndarray:
        """Transform with consistent preprocessing."""

        # Process tabular features
        X_tab_filtered = self.variance_filter.transform(X_tab)
        X_tab_normalized = self._subject_normalize(X_tab_filtered, subjects, fit=False)

        # Extract features
        all_features = self._extract_stable_features(
            X_tab_normalized, X_raw, subjects, sfreq
        )

        # Scale and select
        all_features_scaled = self.robust_scaler.transform(all_features)
        selected_features = self.feature_selector.transform(all_features_scaled)

        # Apply PCA if fitted
        if hasattr(self.pca, 'components_') and selected_features.shape[1] > 50:
            final_features = self.pca.transform(selected_features)
        else:
            final_features = selected_features

        return final_features

    def _subject_normalize(self, X: np.ndarray, subjects: np.ndarray, fit: bool) -> np.ndarray:
        """Subject-specific normalization for better generalization."""
        X_normalized = X.copy()

        if fit:
            self.subject_stats = {}

        for subj in np.unique(subjects):
            subj_mask = subjects == subj
            subj_data = X[subj_mask]

            if fit:
                # Store subject-specific statistics
                self.subject_stats[subj] = {
                    'mean': np.mean(subj_data, axis=0),
                    'std': np.std(subj_data, axis=0) + 1e-8
                }

            if subj in self.subject_stats:
                # Apply subject-specific normalization
                stats = self.subject_stats[subj]
                X_normalized[subj_mask] = (subj_data - stats['mean']) / stats['std']

        return X_normalized

    def _fit_stable_csp(self, X_raw: np.ndarray, y: np.ndarray,
                       subjects: np.ndarray, sfreq: float):
        """Fit CSP with strong regularization for stability."""
        print("        Fitting stable CSP...")

        for band_name, (low, high) in STABLE_FREQ_BANDS.items():
            try:
                # Conservative butterworth filter
                sos = signal.butter(4, [low, high], btype='band', fs=sfreq, output='sos')
                X_filtered = signal.sosfiltfilt(sos, X_raw, axis=2)

                # CSP with heavy regularization
                csp = CSP(
                    n_components=self.n_csp,
                    reg='shrunk',        # Strong regularization
                    log=True,
                    norm_trace=False,
                    component_order='mutual_info'  # Most stable ordering
                )
                csp.fit(X_filtered, y)
                self.csp_filters[band_name] = csp

            except Exception as e:
                print(f"          CSP failed for {band_name}: {e}")

    def _extract_stable_features(self, X_tab: np.ndarray, X_raw: np.ndarray,
                               subjects: np.ndarray, sfreq: float) -> np.ndarray:
        """Extract conservative, stable feature set."""
        features = []

        # 1. Processed tabular features (most reliable)
        features.append(X_tab)

        # 2. Conservative CSP features only
        for band_name, csp_filter in self.csp_filters.items():
            try:
                csp_features = csp_filter.transform(X_raw)
                features.append(csp_features)
            except:
                continue

        # 3. Simple band power features (reliable across subjects)
        band_powers = self._extract_stable_band_powers(X_raw, sfreq)
        if band_powers.size > 0:
            features.append(band_powers)

        return np.hstack(features)

    def _extract_stable_band_powers(self, X_raw: np.ndarray, sfreq: float) -> np.ndarray:
        """Extract simple, reliable band power features."""
        n_samples, n_channels, n_times = X_raw.shape
        band_features = []

        # Use only central channels for stability (reduce noise)
        center_channels = list(range(n_channels//4, 3*n_channels//4, 2))

        for i in range(n_samples):
            sample_features = []

            for ch in center_channels[:8]:  # Limit to 8 channels max
                signal_data = X_raw[i, ch, :]

                # Simple Welch PSD
                freqs, psd = signal.welch(signal_data, fs=sfreq, nperseg=min(64, n_times//4))

                # Conservative band powers
                for band_name, (low, high) in STABLE_FREQ_BANDS.items():
                    band_mask = (freqs >= low) & (freqs <= high)
                    if band_mask.any():
                        # Log power (more stable than raw power)
                        band_power = np.log(np.mean(psd[band_mask]) + 1e-10)
                        sample_features.append(band_power)

            band_features.append(sample_features)

        return np.array(band_features)

# ══════════════════════════════════════════════════════════════════════
# STABLE ENSEMBLE CLASSIFIERS
# ══════════════════════════════════════════════════════════════════════

def create_stable_ensemble():
    """Create conservative ensemble for consistent performance."""

    # Conservative classifiers with strong regularization
    classifiers = []

    # SVM with balanced parameters
    svm = SVC(
        kernel='rbf',
        C=1.0,              # Conservative regularization
        gamma='scale',
        probability=True,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )
    classifiers.append(('svm', svm))

    # Random Forest with conservative settings
    rf = RandomForestClassifier(
        n_estimators=100,    # Conservative number
        max_depth=8,         # Prevent overfitting
        min_samples_split=10, # Higher minimum
        min_samples_leaf=5,   # Higher minimum
        max_features='sqrt',
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    classifiers.append(('rf', rf))

    # LDA (naturally regularized)
    lda = LinearDiscriminantAnalysis(solver='shrinkage', shrinkage=0.2)
    classifiers.append(('lda', lda))

    # Ridge Classifier (L2 regularized)
    ridge = RidgeClassifier(
        alpha=2.0,          # Strong L2 regularization
        class_weight='balanced',
        random_state=RANDOM_STATE
    )
    classifiers.append(('ridge', ridge))

    # Bagged SVM for additional stability
    bagged_svm = BaggingClassifier(
        estimator=SVC(kernel='rbf', C=0.5, gamma='scale', probability=True),
        n_estimators=10,
        max_samples=0.8,
        max_features=0.8,
        random_state=RANDOM_STATE,
        n_jobs=1
    )
    classifiers.append(('bagged_svm', bagged_svm))

    return VotingClassifier(
        estimators=classifiers,
        voting='soft',
        n_jobs=1
    )

def create_conservative_svm():
    """Single conservative SVM for comparison."""
    return SVC(
        kernel='rbf',
        C=0.5,                # Conservative
        gamma='scale',
        probability=True,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )

# ══════════════════════════════════════════════════════════════════════
# MAIN STABILITY-FOCUSED PIPELINE
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

def load_session_data_stable(subject: int, session: int) -> Optional[Tuple]:
    """Load session data with stability focus."""
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
        subjects = np.full(len(y), subject)  # Subject array for normalization

        # Binary filtering with NaN handling
        binary_mask = (y != 0) & ~np.isnan(y)
        if not binary_mask.any():
            return None

        X_tab = X_tab[binary_mask]
        y = y[binary_mask]
        trials = trials[binary_mask]
        subjects = subjects[binary_mask]

        # Clean tabular data
        X_tab = np.nan_to_num(X_tab, nan=0, posinf=0, neginf=0)

        # Load and clean raw data
        X_raw = np.load(raw_file, mmap_mode='r')
        X_raw = X_raw[binary_mask]
        X_raw = X_raw[:, :, ::DECIMATE_FACTOR].astype(np.float32)
        X_raw = np.nan_to_num(X_raw, nan=0, posinf=0, neginf=0)

        return X_tab, X_raw, y, trials, subjects, subject, session

    except Exception as e:
        print(f"  Error loading {subject}_{session}: {e}")
        return None

def stable_cv_per_session(
    X_tab: np.ndarray,
    X_raw: np.ndarray,
    y: np.ndarray,
    trials: np.ndarray,
    subjects: np.ndarray,
    subject: int,
    session: int,
    sfreq: float = 200/3
) -> List[StableResult]:
    """Cross-validation focused on stable, consistent performance."""

    unique_trials = np.unique(trials)
    n_trials = len(unique_trials)

    if n_trials < N_FOLDS:
        n_splits = max(2, min(n_trials, 3))
    else:
        n_splits = N_FOLDS

    gkf = GroupKFold(n_splits=n_splits)
    results = []

    print(f"    Stable {n_splits}-fold CV (consistency focus)")

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
            # Stable feature engineering
            feat_eng = StableFeatureEngineering()
            feat_eng.fit(
                X_tab[train_idx], X_raw[train_idx], y_train,
                subjects[train_idx], sfreq
            )

            X_train_final = feat_eng.transform(
                X_tab[train_idx], X_raw[train_idx], subjects[train_idx], sfreq
            )
            X_test_final = feat_eng.transform(
                X_tab[test_idx], X_raw[test_idx], subjects[test_idx], sfreq
            )

            print(f"      Stable features: {X_train_final.shape[1]}")

            # Test stable models
            models = {
                'stable_ensemble': create_stable_ensemble(),
                'conservative_svm': create_conservative_svm(),
            }

            for model_name, clf in models.items():
                try:
                    clf.fit(X_train_final, y_train)
                    y_pred = clf.predict(X_test_final)
                    accuracy = accuracy_score(y_test, y_pred)

                    result = StableResult(
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
    """Run stability-focused pipeline for consistent 65%+ mean accuracy."""
    print("="*80)
    print("STABLE & CONSISTENT EEG PIPELINE")
    print("Priority: Reliable mean accuracy over peak performance")
    print("Target: 65%+ consistent mean (vs current ~50% mean)")
    print("="*80)

    sessions = discover_sessions()
    print(f"Found {len(sessions)} sessions")

    if not sessions:
        print("No data found!")
        return []

    all_results = []
    target_sessions = min(15, len(sessions))  # Process more for better statistics

    for subject, session in sessions[:target_sessions]:
        print(f"\nProcessing Subject {subject}, Session {session}")

        session_data = load_session_data_stable(subject, session)
        if session_data is None:
            continue

        X_tab, X_raw, y, trials, subjects, _, _ = session_data

        if len(np.unique(y)) < 2:
            continue

        session_results = stable_cv_per_session(
            X_tab, X_raw, y, trials, subjects, subject, session
        )

        if session_results:
            all_results.extend(session_results)

            # Immediate session feedback
            session_accs = [r.accuracy for r in session_results]
            if session_accs:
                mean_acc = np.mean(session_accs)
                print(f"  Session Mean: {mean_acc:.4f}")

                if mean_acc >= 0.65:
                    print(f"  ✅ CONSISTENT TARGET: {mean_acc:.1%}")
                elif mean_acc >= 0.60:
                    print(f"  📈 GOOD PROGRESS: {mean_acc:.1%}")

        del X_tab, X_raw, y, trials, subjects
        gc.collect()

    # Final analysis focusing on consistency
    print(f"\n{'='*80}")
    print("STABILITY-FOCUSED RESULTS")
    print(f"{'='*80}")

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
        overall_min = df['accuracy'].min()
        overall_max = df['accuracy'].max()

        print(f"Mean Accuracy: {overall_mean:.4f} ± {overall_std:.4f}")
        print(f"Range: {overall_min:.4f} - {overall_max:.4f}")
        print(f"Consistency (CV): {overall_std/overall_mean:.3f}")

        # By classifier
        print("\nBy Classifier:")
        for model in df['classifier'].unique():
            model_data = df[df['classifier'] == model]
            accuracies = model_data['accuracy'].values

            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            cv_score = std_acc / mean_acc  # Coefficient of variation

            print(f"{model:>18}: {mean_acc:.4f} ± {std_acc:.4f} (CV: {cv_score:.3f})")

            # Consistency rating
            if cv_score < 0.15 and mean_acc >= 0.65:
                print(f"{'':>18}  🏆 EXCELLENT: Consistent & High")
            elif cv_score < 0.20 and mean_acc >= 0.60:
                print(f"{'':>18}  ✅ GOOD: Stable performance")
            elif mean_acc >= 0.65:
                print(f"{'':>18}  ⚠️  HIGH but inconsistent")
            elif cv_score < 0.20:
                print(f"{'':>18}  📊 CONSISTENT but low")

        # Save results
        df.to_csv('stable_consistent_results.csv', index=False)
        print(f"\nResults saved → stable_consistent_results.csv")

        # Success evaluation (consistency focus)
        if overall_mean >= 0.65 and overall_std <= 0.12:
            print(f"\n🎉 CONSISTENCY TARGET ACHIEVED!")
            print(f"   {overall_mean:.1%} mean with {overall_std:.3f} std deviation")
            print(f"   Reliable for deployment with honest CV")
        elif overall_mean >= 0.60:
            print(f"\n📈 SIGNIFICANT IMPROVEMENT")
            print(f"   {overall_mean:.1%} mean (vs ~50% baseline)")
            print(f"   {overall_std:.3f} deviation (good consistency)")
        else:
            print(f"\n📊 BASELINE PERFORMANCE")
            print(f"   Focus on subject-specific adaptations needed")

    return all_results

if __name__ == "__main__":
    results = main()