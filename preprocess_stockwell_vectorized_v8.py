"""
SEED EEG — Stockwell Transform Preprocessing Pipeline (v8)
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
from scipy.signal import butter, filtfilt, iirnotch
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
    """Apply 1-50 Hz bandpass filter."""
    nyq = 0.5 * sfreq
    b, a = butter(4, [1.0 / nyq, 50.0 / nyq], btype="band")
    return filtfilt(b, a, data, axis=1)


def notch_filter(data: np.ndarray, sfreq: float = 1000.0) -> np.ndarray:
    """Apply 50 Hz notch filter."""
    nyq = 0.5 * sfreq
    b, a = iirnotch(50.0 / nyq, 30.0)
    return filtfilt(b, a, data, axis=1)


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
    Batch Stockwell Transform over all channels.

    S(τ, f) = ∫ x(t) · g(t−τ, f) · e^{−2πift} dt

    CRITICAL: Gaussian kernel centered at frequency f (NOT DC).
    Processes all channels together (vectorized, no Python loops).

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

    # Log-spaced frequencies
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_freqs)

    # Frequency axis for FFT
    f_axis = np.fft.fftfreq(n_samples, d=1.0 / sfreq)

    # FFT of all channels at once
    fft_signals = np.fft.fft(window, axis=1).astype(np.complex64)

    # Allocate output
    st = np.zeros((n_channels, n_freqs, n_samples), dtype=np.complex64)

    # Compute S-transform for each frequency
    for i, f in enumerate(freqs):
        # Gaussian CENTERED AT f (not DC) - this is the critical fix
        sigma_f = f / (2 * np.pi)  # Standard S-transform scaling
        gauss = np.exp(-0.5 * ((f_axis - f) / (sigma_f + 1e-8)) ** 2)
        gauss = gauss.astype(np.complex64)

        # Apply Gaussian and inverse FFT
        st[:, i, :] = np.fft.ifft(fft_signals * gauss, axis=1)

    return np.abs(st).astype(np.float32), freqs


def stockwell_de(window: np.ndarray, sfreq: float = 1000.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Differential Entropy from Stockwell Transform.

    DE(X) = 0.5 × log(2πe × σ²)

    Steps:
    1. Compute Stockwell transform
    2. For each frequency band, average amplitude across freq bins
    3. Compute variance over time for each channel
    4. Apply DE formula

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
    # st_amp: (n_channels, n_freqs, n_samples)

    feats = []

    for low, high in BANDS:
        # Find frequency indices in this band
        idx = np.where((freqs >= low) & (freqs <= high))[0]

        if len(idx) == 0:
            # Fallback: zeros if no frequencies (shouldn't happen)
            feats.append(np.zeros(window.shape[0], dtype=np.float32))
            continue

        # Average S-transform amplitude over frequency bins in band
        # st_amp[:, idx, :] → (n_channels, n_idx, n_samples)
        # .mean(axis=1) → (n_channels, n_samples)
        band_amp = st_amp[:, idx, :].mean(axis=1)

        # Variance over time for each channel → (n_channels,)
        var = np.var(band_amp, axis=1) + 1e-10  # Floor prevents log(0)

        # DE formula
        de = 0.5 * np.log(2 * np.pi * np.e * var)
        feats.append(de.astype(np.float32))

    # Stack bands: (n_bands, n_channels)
    de_stacked = np.stack(feats, axis=0)  # (5, 62)
    de_flat = de_stacked.flatten().astype(np.float32)  # (310,)

    return de_flat, de_stacked


def compute_rasm(de_stacked: np.ndarray) -> np.ndarray:
    """
    Compute Rational Asymmetry (RASM) features from DE.

    RASM captures left-right hemisphere asymmetry by computing ratios
    of DE values at homologous electrode pairs.

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

    rasm = []
    for band_idx in range(de_stacked.shape[0]):  # For each band
        for l, r in zip(LEFT, RIGHT):
            # Ratio of left to right (asymmetry measure)
            ratio = de_stacked[band_idx, l] / (de_stacked[band_idx, r] + 1e-6)
            rasm.append(ratio)

    return np.array(rasm, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# PER-WINDOW PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def process_window(window: np.ndarray, sfreq: float = 1000.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Process single EEG window: filter → Stockwell DE + RASM → z-score for CSP.

    CRITICAL FLOW (per window, not per trial):
        raw window → bandpass → notch → Stockwell DE + RASM → discard
        raw window → bandpass → notch → z-score → save for CSP

    IMPORTANT: DE is computed on FILTERED (but NOT z-scored) window.
    Z-scoring destroys variance information which DE depends on.

    Returns
    -------
    w_norm : (n_channels, n_samples) float32 — z-scored for CSP
    feat : (360,) float32 — Stockwell DE (310) + RASM (50) features
    """
    # Step 1: Filter THIS window (not the whole trial)
    w_filtered = bandpass_filter(window, sfreq)
    w_filtered = notch_filter(w_filtered, sfreq)

    # Step 2: Stockwell DE on filtered (but NOT z-scored) window
    de_flat, de_stacked = stockwell_de(w_filtered, sfreq)

    # Step 3: Compute RASM from stacked DE
    rasm = compute_rasm(de_stacked)

    # Step 4: Concatenate DE + RASM
    feat = np.concatenate([de_flat, rasm])

    # Step 5: Z-score normalization ONLY for CSP output (AFTER DE)
    w_norm = (w_filtered - w_filtered.mean(axis=1, keepdims=True)) / (
        w_filtered.std(axis=1, keepdims=True) + 1e-6
    )

    return w_norm.astype(np.float32), feat
    w_norm = (w_filtered - w_filtered.mean(axis=1, keepdims=True)) / (
        w_filtered.std(axis=1, keepdims=True) + 1e-6
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
            # Get trial data (RAW - no filtering here!)
            trial = raw.get_data(start=start, stop=end).astype(np.float32)

            # Create windows from RAW trial (filtering happens per-window)
            windows = list(create_windows(trial, WINDOW_SIZE, STEP_SIZE))
            windows = windows[:MAX_WINDOWS_PER_TRIAL]  # Limit

            del trial  # Free memory

            if len(windows) == 0:
                print(f"  ⚠ Trial {trial_idx} produced 0 windows, skipping")
                continue

            # Process windows in parallel
            # Each window is: raw → filter → DE → z-score
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
