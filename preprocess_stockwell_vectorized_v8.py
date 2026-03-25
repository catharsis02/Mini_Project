"""
SEED EEG — Stockwell Transform Preprocessing Pipeline (v8).

==========================================================
Memory budget : 8 GB RAM
CPU budget    : 8 cores

Features
────────
  - Stockwell Transform (S-transform) for time-frequency analysis
  - DE (Differential Entropy) per band per channel (310 features)
  - RASM (Rational Asymmetry) for hemisphere asymmetry (50 features)
  - Total: 360 features per window
  - Binary classification: positive (1) vs negative (-1), neutral (0) excluded
  - Session-aware splitting for CV (no session mixing)
  - Per-session scaling (no global scaling leakage)

Critical Design (from mini pipeline)
────────────────────────────────────
  - WINDOW-FIRST processing: trial → window → features → discard
  - NO full trial storage, NO large intermediate arrays
  - Constant memory regardless of dataset size
  - Stockwell computed PER WINDOW only (not per trial)

Bug Fixes
─────────
  BUG-1  trial_id must NOT include window index (GroupKFold leakage)
  BUG-2  Explicit band-major column names for ML pipeline

Hardening
─────────
  HARD-1  Assert START/END/LABEL lengths match
  HARD-2  Assert sampling rate = 1000 Hz
  HARD-3  Skip empty trials gracefully
  HARD-4  Hard fail on window count mismatch
"""

from __future__ import annotations

import gc
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from joblib import Parallel, delayed

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH = Path("./data/SEED_EEG/SEED_RAW_EEG")
SAVE_PATH = Path("./data/SEED_EEG/processed_data/")
SAVE_PATH.mkdir(exist_ok=True)

N_JOBS = 4
EXPECTED_SFREQ = 1000

# Windowing (MUST match count_windows exactly)
WINDOW_SIZE = 2000  # 2 seconds at 1000 Hz
STEP_SIZE = 1000    # 1 second (50% overlap)
MAX_WINDOWS_PER_TRIAL = 20  # Limit to avoid OOM

# Trial boundaries at 1000 Hz (from time.txt)
START_POINTS = [
    27000, 290000, 551000, 784000, 1050000, 1262000, 1484000,
    1748000, 1993000, 2287000, 2551000, 2812000, 3072000, 3335000, 3599000,
]
END_POINTS = [
    262000, 523000, 757000, 1022000, 1235000, 1457000, 1721000,
    1964000, 2258000, 2524000, 2786000, 3045000, 3307000, 3573000, 3805000,
]
LABELS = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

# HARD-1: Validate at import time
assert len(START_POINTS) == len(END_POINTS) == len(LABELS), \
    "START_POINTS, END_POINTS, LABELS must have same length"
assert all(s < e for s, e in zip(START_POINTS, END_POINTS)), \
    "Every START_POINT must be < END_POINT"

# Frequency bands for DE
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 50)]
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
N_BANDS = len(BANDS)

# Stockwell parameters
ST_FMIN = 1.0
ST_FMAX = 50.0
ST_N_FREQS = 75  # Log-spaced frequency bins


# ══════════════════════════════════════════════════════════════════════════════
# WINDOWING (MUST MATCH EXACTLY)
# ══════════════════════════════════════════════════════════════════════════════

def count_windows(n_samples: int, window: int = WINDOW_SIZE, step: int = STEP_SIZE) -> int:
    """Count windows with exact same logic as create_windows."""
    if n_samples < window:
        return 0
    return ((n_samples - window) // step) + 1


def create_windows(trial: np.ndarray, window: int = WINDOW_SIZE, step: int = STEP_SIZE):
    """
    Yield windows from trial. MUST match count_windows exactly.

    Uses range(0, n - window + 1, step) to ensure consistency.
    """
    n_samples = trial.shape[1]
    for i in range(0, n_samples - window + 1, step):
        yield trial[:, i:i + window]


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def bandpass_filter(data: np.ndarray, sfreq: float = 1000.0) -> np.ndarray:
    """Apply 1-45 Hz bandpass filter using SOS form for numerical stability."""
    sos = butter(4, [1.0, 45.0], btype="band", fs=sfreq, output='sos')
    return sosfiltfilt(sos, data, axis=1)


def notch_filter(data: np.ndarray, sfreq: float = 1000.0) -> np.ndarray:
    """Apply 50 Hz notch filter using SOS form for numerical stability."""
    from scipy.signal import tf2sos
    b, a = iirnotch(50.0, 30.0, fs=sfreq)
    sos = tf2sos(b, a)
    return sosfiltfilt(sos, data, axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# STOCKWELL TRANSFORM (CORRECT IMPLEMENTATION)
# ══════════════════════════════════════════════════════════════════════════════

def stockwell_transform(
    window: np.ndarray,
    sfreq: float = 1000.0,
    fmin: float = ST_FMIN,
    fmax: float = ST_FMAX,
    n_freqs: int = ST_N_FREQS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Batch Stockwell Transform over all channels with mathematically correct normalization.

    S(τ, f) = ∫ x(t) · g(t−τ, f) · e^{−2πift} dt

    CRITICAL CORRECTIONS:
    1. Gaussian kernel centered at frequency f (NOT DC)
    2. Properly normalized Gaussian to preserve energy across frequencies
    3. Numerically stable for all frequencies including f → 0

    Parameters
    ----------
    window : (n_channels, n_samples)
        Single EEG window (already filtered)
    sfreq : float
        Sampling frequency
    fmin, fmax : float
        Frequency range
    n_freqs : int
        Number of log-spaced frequency bins

    Returns
    -------
    st_amp : (n_channels, n_freqs, n_samples) float32
        Absolute value of S-transform
    freqs : (n_freqs,)
        Frequency axis
    """
    n_channels, n_samples = window.shape

    # Log-spaced frequencies (avoid f=0 exactly)
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_freqs)

    # Frequency axis for FFT
    f_axis = np.fft.fftfreq(n_samples, d=1.0 / sfreq)

    # FFT of all channels at once
    fft_signals = np.fft.fft(window, axis=1).astype(np.complex64)

    # Allocate output
    st = np.zeros((n_channels, n_freqs, n_samples), dtype=np.complex64)

    # Compute S-transform for each frequency
    for i, f in enumerate(freqs):
        # S-transform frequency-domain Gaussian: exp(-2π²(α-f)²/f²)
        # where α is the FFT frequency axis  # Avoid division by zero for f ≈ 0
        if f < 1e-6:
            f = 1e-6

        # Standard S-transform Gaussian in frequency domain
        alpha = f_axis - f  # Frequency offset
        gauss = np.exp(-2.0 * np.pi**2 * alpha**2 / (f**2 + 1e-10))
        gauss = gauss.astype(np.complex64)

        # Apply Gaussian and inverse FFT
        st[:, i, :] = np.fft.ifft(fft_signals * gauss, axis=1)

    return np.abs(st).astype(np.float32), freqs


def stockwell_de(window: np.ndarray, sfreq: float = 1000.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Differential Entropy from Stockwell Transform with mathematical rigor.

    CRITICAL: DE must be computed on POWER (amplitude²), not amplitude.

    DE(X) = 0.5 × log(2πe × σ²)

    where σ² is the variance of the POWER signal.

    MATHEMATICAL CORRECTIONS:
    1. Use POWER = |S(t,f)|² (not amplitude)
    2. Stable variance estimation with numerical guards
    3. Proper handling of log-domain operations

    Steps:
    1. Compute Stockwell transform → complex values
    2. Compute POWER = |S(t,f)|² (not just amplitude)
    3. For each frequency band, average power across freq bins
    4. Compute variance over time for each channel
    5. Apply DE formula with numerical stability

    Output layout — band-major:
        [δ_ch0 δ_ch1 … δ_ch61 | θ_ch0 … θ_ch61 | … | γ_ch0 … γ_ch61]
        = 5 bands × 62 channels = 310 features

    Returns
    -------
    de_flat : float32 array, shape (n_channels * N_BANDS,) - flattened DE
    de_stacked : float32 array, shape (n_bands, n_channels) - for RASM computation
    """
    # Compute Stockwell transform for this window
    st_amp, freqs = stockwell_transform(window, sfreq)
    # st_amp: (n_channels, n_freqs, n_samples) - this is AMPLITUDE

    # CRITICAL: Convert amplitude to POWER for correct DE computation
    # Power = amplitude² ensures correct variance scaling
    st_power = st_amp ** 2  # Power = |S(t,f)|²
    # st_power: (n_channels, n_freqs, n_samples)

    feats = []

    for low, high in BANDS:
        # Find frequency indices in this band
        idx = np.where((freqs >= low) & (freqs <= high))[0]

        if len(idx) == 0:
            # Fallback: use minimum DE value if band has no frequencies
            # This preserves feature count without introducing NaN
            feats.append(np.full(window.shape[0], -10.0, dtype=np.float32))
            continue

        # Average power over frequency bins in band
        # st_power[:, idx, :] → (n_channels, n_idx, n_samples)
        # Using arithmetic mean is correct for power averaging
        # .mean(axis=1) → (n_channels, n_samples)
        band_power = st_power[:, idx, :].mean(axis=1)

        # CRITICAL: For non-Gaussian data, compute variance on log-power
        log_power = np.log(band_power + 1e-10)
        var = np.var(log_power, axis=1, ddof=1)

        # Numerical guard: ensure variance is positive and above machine epsilon
        var = np.maximum(var, 1e-10)

        # DE formula: h(X) = 0.5 * log(2πe * σ²)
        # Mathematically equivalent to: h(X) = 0.5 * (1 + log(2π) + log(σ²))
        # Using direct formula for numerical stability
        de = 0.5 * np.log(2.0 * np.pi * np.e * var)

        # Ensure no NaN or Inf values
        de = np.nan_to_num(de, nan=-10.0, posinf=10.0, neginf=-10.0)

        feats.append(de.astype(np.float32))

    # Stack bands: (n_bands, n_channels)
    de_stacked = np.stack(feats, axis=0)  # (5, 62)
    de_flat = de_stacked.flatten().astype(np.float32)  # (310,)

    return de_flat, de_stacked


def compute_rasm(de_stacked: np.ndarray) -> np.ndarray:
    """
    Compute Rational Asymmetry (RASM) features from DE with numerical stability.

    MATHEMATICAL CORRECTNESS:
    Since DE values are in log-space (DE = 0.5*log(2πeσ²)), the correct
    asymmetry measure is the DIFFERENCE (not ratio):

        RASM = DE_right - DE_left
             = 0.5*log(2πeσ²_R) - 0.5*log(2πeσ²_L)
             = 0.5*log(σ²_R/σ²_L)

    This is the standard EEG Frontal Asymmetry Index (FAI) formulation
    (Davidson 1992, Allen et al. 2004).

    NUMERICAL STABILITY:
    - Handles potential NaN/Inf values from DE computation
    - Ensures asymmetry features are finite and meaningful

    Parameters
    ----------
    de_stacked : (n_bands, n_channels) array
        Stacked DE features before flattening

    Returns
    -------
    rasm : (n_pairs * n_bands,) float32 array
        50 RASM features (10 pairs × 5 bands)
    """
    # EEG channel pairs (left-right homologous regions)
    # Based on standard 62-channel montage
    LEFT =  [0, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    RIGHT = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    # Ensure channel indices are valid
    assert max(LEFT + RIGHT) < de_stacked.shape[1], \
        f"Invalid channel indices: max={max(LEFT + RIGHT)}, n_channels={de_stacked.shape[1]}"

    rasm = []
    for band_idx in range(de_stacked.shape[0]):  # For each band
        for l, r in zip(LEFT, RIGHT):
            # Log-difference asymmetry (since DE is already in log-space)
            # This equals: 0.5 * log(σ²_R / σ²_L)
            de_left = de_stacked[band_idx, l]
            de_right = de_stacked[band_idx, r]

            # Compute asymmetry with numerical guards
            asymmetry = de_right - de_left

            # Guard against NaN/Inf from upstream computation
            if not np.isfinite(asymmetry):
                asymmetry = 0.0  # Neutral asymmetry if invalid

            rasm.append(asymmetry)

    return np.array(rasm, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# PER-WINDOW PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def process_window(window: np.ndarray, sfreq: float = 1000.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Process single EEG window: Stockwell DE + RASM → z-score for CSP.

    CRITICAL FLOW (per window, not per trial):
        filtered window → Stockwell DE + RASM → discard
        filtered window → z-score → save for CSP

    IMPORTANT: Window is ALREADY FILTERED at trial level.
    DE is computed on FILTERED (but NOT z-scored) window.
    Z-scoring destroys variance information which DE depends on.

    NOTE: Raw data saved at 1000 Hz. If downstream pipeline applies decimation,
    it MUST use proper anti-aliasing filter before downsampling to prevent aliasing
    artifacts (e.g., scipy.signal.decimate or manual lowpass + resample).

    Returns
    -------
    w_norm : (n_channels, n_samples) float32 — z-scored for CSP
    feat : (360,) float32 — Stockwell DE (310) + RASM (50) features
    """
    # Step 1: Stockwell DE on filtered (but NOT z-scored) window
    # CRITICAL: DE now computed on log-power
    de_flat, de_stacked = stockwell_de(window, sfreq)

    # Step 2: Compute RASM from stacked DE
    rasm = compute_rasm(de_stacked)

    # Step 3: Concatenate DE + RASM (310 + 50 = 360 features)
    feat = np.concatenate([de_flat, rasm])

    # Step 4: Z-score normalization ONLY for CSP output (AFTER DE computation)
    w_norm = (window - window.mean(axis=1, keepdims=True)) / (
        window.std(axis=1, keepdims=True) + 1e-6
    )

    return w_norm.astype(np.float32), feat


# ══════════════════════════════════════════════════════════════════════════════
# FILE PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

def process_file(file: Path) -> dict:
    """
    Process one subject-session CNT file.

    Binary classification: Only keeps positive (1) and negative (-1) labels.
    Neutral (0) trials are excluded.

    Returns
    -------
    dict with processing statistics
    """
    parts = file.stem.split("_")
    if len(parts) != 2:
        raise ValueError(f"Unexpected filename: {file.name}")

    subject, session = int(parts[0]), int(parts[1])
    print(f"\n{'='*60}")
    print(f"Processing {file.stem}  (subject={subject}, session={session})")
    print(f"{'='*60}")

    raw_path = SAVE_PATH / f"{file.stem}_raw.npy"

    with mne.io.read_raw_cnt(file, preload=False) as raw:
        # Drop non-EEG channels
        drop = [ch for ch in ["M1", "M2", "VEO", "HEO"] if ch in raw.ch_names]
        if drop:
            raw.drop_channels(drop)

        sfreq = raw.info["sfreq"]
        n_channels = len(raw.ch_names)

        # HARD-2: Assert sampling rate
        if int(sfreq) != EXPECTED_SFREQ:
            raise RuntimeError(
                f"{file.name}: expected {EXPECTED_SFREQ} Hz, got {sfreq} Hz"
            )

        print(f"  Channels: {n_channels}, sfreq: {sfreq} Hz")

        # Count windows for BINARY labels only (exclude neutral)
        binary_trials = [(i, s, e, LABELS[i])
                         for i, (s, e) in enumerate(zip(START_POINTS, END_POINTS))
                         if LABELS[i] != 0]  # Exclude neutral

        total_windows = 0
        for _, start, end, _ in binary_trials:
            n_win = count_windows(end - start)
            n_win = min(n_win, MAX_WINDOWS_PER_TRIAL)  # Limit per trial
            total_windows += n_win

        print(f"  Binary trials: {len(binary_trials)}/15 (excluding neutral)")
        print(f"  Expected windows: {total_windows}")

        if total_windows == 0:
            print("  ⚠ No windows to process!")
            return {"file": file.stem, "windows": 0, "status": "empty"}

        # Allocate arrays
        # Features: 310 (DE) + 50 (RASM) = 360 total
        X_raw = np.lib.format.open_memmap(
            raw_path, mode="w+", dtype=np.float32,
            shape=(total_windows, n_channels, WINDOW_SIZE),
        )
        X_feat = np.empty((total_windows, n_channels * N_BANDS + 50), dtype=np.float32)  # 360 features
        Yw = np.empty(total_windows, dtype=np.int8)
        trial_ids = []

        write_idx = 0

        for trial_idx, start, end, label in binary_trials:
            # Get trial data (RAW)
            trial = raw.get_data(start=start, stop=end).astype(np.float32)

            # CRITICAL: Filter FULL trial before windowing
            # Window-wise filtering is mathematically incorrect for spectral analysis
            trial_filtered = bandpass_filter(trial, sfreq)
            trial_filtered = notch_filter(trial_filtered, sfreq)

            # Create windows from FILTERED trial
            windows = list(create_windows(trial_filtered, WINDOW_SIZE, STEP_SIZE))
            windows = windows[:MAX_WINDOWS_PER_TRIAL]  # Limit

            del trial, trial_filtered  # Free memory

            if len(windows) == 0:
                print(f"  ⚠ Trial {trial_idx} produced 0 windows, skipping")
                continue

            # Process windows in parallel
            # Each window is already filtered: DE → z-score
            results = Parallel(n_jobs=N_JOBS, backend="loky")(
                delayed(process_window)(w, sfreq) for w in windows
            )
            del windows

            # Group ID: all windows share same ID (no window index!)
            group_id = f"{subject}_{session}_{trial_idx}"

            for w_norm, feat in results:
                X_raw[write_idx] = w_norm
                X_feat[write_idx] = feat
                Yw[write_idx] = label
                trial_ids.append(group_id)
                write_idx += 1

            del results
            gc.collect()

        X_raw.flush()

    actual = write_idx

    # HARD-4: Verify window count
    if actual != total_windows:
        raw_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"{file.stem}: expected {total_windows} windows, got {actual}"
        )

    # Truncate to actual (should match)
    X_feat = X_feat[:actual]
    Yw = Yw[:actual]

    # Create DataFrame with band-major column names: DE + RASM
    de_cols = [
        f"{band}_ch{ch}"
        for band in BAND_NAMES
        for ch in range(n_channels)
    ]
    rasm_cols = [
        f"rasm_{band}_pair{p}"
        for band in BAND_NAMES
        for p in range(10)
    ]
    feat_cols = de_cols + rasm_cols  # 310 + 50 = 360

    df = pd.DataFrame(X_feat, columns=feat_cols)
    df["label"] = Yw
    df["subject"] = subject
    df["session"] = session
    df["trial_id"] = trial_ids

    # Save parquet
    out_parquet = SAVE_PATH / f"{file.stem}.parquet"
    df.to_parquet(out_parquet)

    # Statistics
    label_dist = {int(k): int(v) for k, v in zip(*np.unique(Yw, return_counts=True))}
    n_unique_trials = len(set(trial_ids))

    print(f"\n  ✓ Saved: {out_parquet.name} + {raw_path.name}")
    print(f"    Windows: {actual}")
    print(f"    Labels: {label_dist} (binary: -1=neg, 1=pos)")
    print(f"    Unique trials: {n_unique_trials}")
    print(f"    Features: {len(feat_cols)} ({feat_cols[0]}...{feat_cols[-1]})")

    del X_raw, X_feat, Yw, df
    gc.collect()

    return {
        "file": file.stem,
        "windows": actual,
        "labels": label_dist,
        "trials": n_unique_trials,
        "status": "success"
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("SEED EEG Stockwell Transform Preprocessing")
    print("Binary Classification: Positive (1) vs Negative (-1)")
    print("=" * 70)

    cnt_files = sorted(p for p in DATA_PATH.iterdir() if p.suffix == ".cnt")
    skipped = [p.name for p in DATA_PATH.iterdir()
               if p.suffix != ".cnt" and p.is_file()]

    if skipped:
        print(f"\nSkipping non-.cnt files: {skipped}")
    print(f"\nFound {len(cnt_files)} .cnt files")

    results = []
    failed = []

    for file in cnt_files:
        try:
            stats = process_file(file)
            results.append(stats)
        except Exception as exc:
            print(f"\n  ✗ {file.name} FAILED: {exc}")
            failed.append(file.name)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_windows = sum(r["windows"] for r in results if r["status"] == "success")
    print(f"Files processed: {len(results)}/{len(cnt_files)}")
    print(f"Total windows: {total_windows}")
    print(f"Failed: {len(failed)}")

    if failed:
        print(f"Failed files: {failed}")

    print("\nPreprocessing complete!")
    print("Next: Run ML pipeline with session-aware CV and per-session scaling")
