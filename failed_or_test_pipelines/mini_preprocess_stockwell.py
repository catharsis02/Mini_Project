"""
Mini Stockwell Preprocessing - Memory-safe version
Process ONE window at a time, no parallelization
"""

import gc
import numpy as np
import mne
from pathlib import Path
from scipy.signal import butter, filtfilt, iirnotch
import pandas as pd

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
DATA_PATH = Path("./data/SEED_EEG/SEED_RAW_EEG")
SAVE_PATH = Path("./mini_data_stockwell")
SAVE_PATH.mkdir(exist_ok=True)

SFREQ = 1000
WINDOW = 2000
STEP = 1000
MAX_WINDOWS_PER_TRIAL = 5  # Reduced for memory  # Reduced for memory

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 50)]
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
N_BANDS = 5

# Stockwell params - REDUCED for memory
ST_FMIN = 1.0
ST_FMAX = 50.0
ST_N_FREQS = 50  # Reduced from 75

START_POINTS = [
    27000, 290000, 551000, 784000, 1050000,
    1262000, 1484000, 1748000, 1993000, 2287000,
    2551000, 2812000, 3072000, 3335000, 3599000
]

END_POINTS = [
    262000, 523000, 757000, 1022000, 1235000,
    1457000, 1721000, 1964000, 2258000, 2524000,
    2786000, 3045000, 3307000, 3573000, 3805000
]

LABELS = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]


# ─────────────────────────────
# FILTERS
# ─────────────────────────────
def bandpass(data, sfreq=1000):
    nyq = 0.5 * sfreq
    b, a = butter(4, [1.0 / nyq, 50.0 / nyq], btype="band")
    return filtfilt(b, a, data, axis=1)


def notch(data, sfreq=1000):
    nyq = 0.5 * sfreq
    b, a = iirnotch(50.0 / nyq, 30.0)
    return filtfilt(b, a, data, axis=1)


# ─────────────────────────────
# STOCKWELL TRANSFORM (Memory-efficient)
# ─────────────────────────────
def stockwell_de(window, sfreq=1000):
    """
    Memory-efficient Stockwell DE.

    Computes one frequency at a time, accumulates band statistics.

    Flow: raw window → filter → S-transform → band variance → DE
    """
    n_channels, n_samples = window.shape
    freqs = np.logspace(np.log10(ST_FMIN), np.log10(ST_FMAX), ST_N_FREQS)
    f_axis = np.fft.fftfreq(n_samples, d=1.0 / sfreq)

    # FFT once
    fft_signals = np.fft.fft(window, axis=1).astype(np.complex64)

    # Accumulate band amplitudes
    band_sums = [np.zeros((n_channels, n_samples), dtype=np.float32) for _ in range(N_BANDS)]
    band_counts = [0] * N_BANDS

    for f in freqs:
        # Gaussian centered at f
        sigma_f = f / (2 * np.pi)
        gauss = np.exp(-0.5 * ((f_axis - f) / (sigma_f + 1e-8)) ** 2).astype(np.complex64)

        # S-transform for this frequency
        st_f = np.abs(np.fft.ifft(fft_signals * gauss, axis=1)).astype(np.float32)

        # Add to appropriate band
        for bi, (low, high) in enumerate(BANDS):
            if low <= f <= high:
                band_sums[bi] += st_f
                band_counts[bi] += 1

        del st_f

    del fft_signals

    # Compute DE for each band
    de_feats = []
    for bi in range(N_BANDS):
        if band_counts[bi] > 0:
            band_amp = band_sums[bi] / band_counts[bi]  # (n_channels, n_samples)
            var = np.var(band_amp, axis=1) + 1e-10  # variance over TIME
            de = 0.5 * np.log(2 * np.pi * np.e * var)
            de_feats.append(de.astype(np.float32))
        else:
            de_feats.append(np.zeros(n_channels, dtype=np.float32))

    del band_sums
    gc.collect()

    # Stack: (n_bands, n_channels) → flatten band-major
    return np.stack(de_feats, axis=0).flatten()


# ─────────────────────────────
# PROCESS ONE WINDOW
# ─────────────────────────────
def process_window(w_raw, sfreq=1000):
    """
    Process single window: filter → DE → z-score for CSP
    """
    # Filter this window
    w = bandpass(w_raw, sfreq)
    w = notch(w, sfreq)

    # DE on filtered (NOT z-scored)
    de = stockwell_de(w, sfreq)

    # Z-score for CSP (AFTER DE)
    w_norm = (w - w.mean(axis=1, keepdims=True)) / (w.std(axis=1, keepdims=True) + 1e-6)

    return w_norm.astype(np.float32), de.astype(np.float32)


# ─────────────────────────────
# PROCESS FILE (Memory-safe)
# ─────────────────────────────
def process_file(file):
    print(f"\nProcessing {file.name}")

    # Load with preload=False to save memory
    raw = mne.io.read_raw_cnt(file, preload=False, verbose=False)
    raw.drop_channels([ch for ch in ["M1", "M2", "VEO", "HEO"] if ch in raw.ch_names])

    sfreq = raw.info["sfreq"]
    n_channels = len(raw.ch_names)
    subject = int(file.stem.split("_")[0])
    session = int(file.stem.split("_")[1])

    # Binary trials only
    binary_trials = [(i, s, e, LABELS[i])
                     for i, (s, e) in enumerate(zip(START_POINTS, END_POINTS))
                     if LABELS[i] != 0]

    # Process and save incrementally
    all_raw = []
    all_feat = []
    all_y = []
    all_trials = []

    for trial_idx, start, end, label in binary_trials:
        # Load just this trial
        trial = raw.get_data(start=start, stop=end).astype(np.float32)

        count = 0
        for i in range(0, trial.shape[1] - WINDOW, STEP):
            if count >= MAX_WINDOWS_PER_TRIAL:
                break

            w_raw = trial[:, i:i + WINDOW]
            w_norm, de = process_window(w_raw, sfreq)

            all_raw.append(w_norm)
            all_feat.append(de)
            all_y.append(label)
            all_trials.append(f"{subject}_{session}_{trial_idx}")

            count += 1
            del w_raw, w_norm, de

        del trial
        gc.collect()

    raw.close()
    del raw
    gc.collect()

    if not all_raw:
        print(f"  ⚠ No windows produced")
        return 0

    # Stack and save
    X_raw = np.stack(all_raw)
    X_feat = np.stack(all_feat)
    y = np.array(all_y, dtype=np.int8)

    del all_raw, all_feat
    gc.collect()

    np.save(SAVE_PATH / f"{file.stem}_raw.npy", X_raw)

    feat_cols = [f"{band}_ch{ch}" for band in BAND_NAMES for ch in range(n_channels)]
    df = pd.DataFrame(X_feat, columns=feat_cols)
    df["label"] = y
    df["subject"] = subject
    df["session"] = session
    df["trial_id"] = all_trials
    df.to_parquet(SAVE_PATH / f"{file.stem}.parquet")

    label_dist = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    print(f"  → raw:{X_raw.shape} feat:{X_feat.shape} labels:{label_dist}")

    del X_raw, X_feat, df
    gc.collect()

    return len(y)


# ─────────────────────────────
# ENTRY
# ─────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Mini Stockwell Preprocessing (Memory-safe)")
    print("=" * 60)

    files = sorted(DATA_PATH.glob("*.cnt"))
    # Only 2 files to test
    selected = [f for f in files if f.stem.startswith("1_")][:2]

    print(f"Processing {len(selected)} files")

    total = 0
    for f in selected:
        try:
            n = process_file(f)
            total += n
            gc.collect()
        except Exception as e:
            print(f"  ✗ {f.name} failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nTotal windows: {total}")
    print("Done!")
