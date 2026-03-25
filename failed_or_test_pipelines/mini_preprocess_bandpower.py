"""
Mini Bandpower Preprocessing - Simple DE (known working approach)
Use this as baseline to verify pipeline before Stockwell
"""

import gc
import numpy as np
import mne
from pathlib import Path
from scipy.signal import butter, filtfilt
import pandas as pd

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
DATA_PATH = Path("./data/SEED_EEG/SEED_RAW_EEG")
SAVE_PATH = Path("./mini_data_bandpower")
SAVE_PATH.mkdir(exist_ok=True)

SFREQ = 1000
WINDOW = 2000
STEP = 1000
MAX_WINDOWS_PER_TRIAL = 10  # More windows

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 50)]
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
N_BANDS = 5

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
# SIMPLE BANDPOWER DE (known working)
# ─────────────────────────────
def compute_de_bandpower(window, sfreq=1000):
    """
    Simple bandpower DE - the approach that works in mini_preprocess.py

    For each band:
      1. Bandpass filter
      2. Variance over time
      3. DE = 0.5 * log(2*pi*e * var)

    Returns: (n_bands, n_channels) for RASM computation, then flatten
    """
    feats = []
    nyq = sfreq / 2

    for low, high in BANDS:
        b, a = butter(4, [low / nyq, high / nyq], btype="band")
        filtered = filtfilt(b, a, window, axis=1)

        # Variance over TIME axis (axis=1)
        var = np.var(filtered, axis=1) + 1e-10

        # DE formula
        de = 0.5 * np.log(2 * np.pi * np.e * var)
        feats.append(de)

    # Stack: (n_bands, n_channels)
    feats = np.stack(feats, axis=0)  # (5, 62)
    return feats


def compute_rasm(de_feats):
    """
    Compute Rational Asymmetry (RASM) features.
    Left-right hemisphere asymmetry ratios.

    Args:
        de_feats: (n_bands, n_channels) array

    Returns:
        (n_pairs * n_bands,) flattened RASM features
    """
    # EEG channel pairs (left-right homologous regions)
    LEFT =  [0, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    RIGHT = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    rasm = []
    for band_idx in range(de_feats.shape[0]):  # For each band
        for l, r in zip(LEFT, RIGHT):
            ratio = de_feats[band_idx, l] / (de_feats[band_idx, r] + 1e-6)
            rasm.append(ratio)

    return np.array(rasm, dtype=np.float32)


# ─────────────────────────────
# PROCESS FILE
# ─────────────────────────────
def process_file(file):
    print(f"\nProcessing {file.name}")

    # preload=False to save memory
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

    all_raw = []
    all_feat = []
    all_y = []
    all_trials = []

    for trial_idx, start, end, label in binary_trials:
        trial = raw.get_data(start=start, stop=end)

        count = 0
        for i in range(0, trial.shape[1] - WINDOW, STEP):
            if count >= MAX_WINDOWS_PER_TRIAL:
                break

            w = trial[:, i:i + WINDOW]

            # Compute DE features (n_bands, n_channels)
            de = compute_de_bandpower(w, sfreq)

            # Flatten DE: (n_bands, n_channels) → (n_bands * n_channels,)
            de_flat = de.flatten()

            # Compute RASM features
            rasm = compute_rasm(de)

            # Combine DE + RASM
            feat = np.concatenate([de_flat, rasm])

            # Z-score for CSP
            w_norm = (w - w.mean(axis=1, keepdims=True)) / (
                w.std(axis=1, keepdims=True) + 1e-6
            )

            all_raw.append(w_norm.astype(np.float32))
            all_feat.append(feat)
            all_y.append(label)
            all_trials.append(f"{subject}_{session}_{trial_idx}")

            count += 1

        del trial
        gc.collect()

    raw.close()
    del raw
    gc.collect()

    if not all_raw:
        print(f"  ⚠ No windows")
        return 0

    X_raw = np.stack(all_raw)
    X_feat = np.stack(all_feat)
    y = np.array(all_y, dtype=np.int8)

    # Check features are valid
    print(f"  Feature stats: min={X_feat.min():.2f} max={X_feat.max():.2f} mean={X_feat.mean():.2f}")
    print(f"  Features: {X_feat.shape[1]} (DE: 310, RASM: 50)")

    np.save(SAVE_PATH / f"{file.stem}_raw.npy", X_raw)

    # Column names: DE features + RASM features
    de_cols = [f"{band}_ch{ch}" for band in BAND_NAMES for ch in range(n_channels)]
    rasm_cols = [f"rasm_{band}_pair{p}" for band in BAND_NAMES for p in range(10)]
    feat_cols = de_cols + rasm_cols

    df = pd.DataFrame(X_feat, columns=feat_cols)
    df["label"] = y
    df["subject"] = subject
    df["session"] = session
    df["trial_id"] = all_trials
    df.to_parquet(SAVE_PATH / f"{file.stem}.parquet")

    label_dist = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    print(f"  → raw:{X_raw.shape} feat:{X_feat.shape} labels:{label_dist}")

    del X_raw, X_feat
    gc.collect()

    return len(y)


# ─────────────────────────────
# ENTRY
# ─────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Mini Bandpower Preprocessing (Simple DE)")
    print("=" * 60)

    files = sorted(DATA_PATH.glob("*.cnt"))
    # Subject 1 all sessions + subject 2 session 1
    selected = [f for f in files if f.stem.startswith("1_")] + \
               [f for f in files if f.stem == "2_1"]

    print(f"Processing {len(selected)} files")

    total = 0
    for f in selected:
        try:
            n = process_file(f)
            total += n
        except Exception as e:
            print(f"  ✗ {f.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nTotal: {total}")
