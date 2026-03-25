"""
SEED EEG — Preprocessing Pipeline
===================================
Memory budget : 8 GB RAM
CPU budget    : 8 cores

Memory fixes applied vs original
──────────────────────────────────
1. n_jobs=4 (was -1)
   loky copies the window array into every worker process.
   8 workers × window_bytes = OOM.  4 workers keeps peak < ~2 GB.

2. Trial deleted before Parallel() call
   Without `del trial`, each worker inherits a copy of the full trial
   in addition to its own window slice.

3. X_raw stays as a memmap — no second np.save()
   The original did np.save(... X_raw) after the memmap, which
   materialises the whole array in RAM a second time.
   The memmap file IS the final .npy; just flush() and move on.

4. gc.collect() after every trial
   Ensures Python/OS returns freed pages before the next trial load.
"""

from __future__ import annotations

import gc
from pathlib import Path

import mne
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from joblib import Parallel, delayed

# ── CONFIG ─────────────────────────────────────────────────────────────────

DATA_PATH = Path("./data/SEED_EEG/SEED_RAW_EEG")
SAVE_PATH = Path("./data/SEED_EEG/processed_data/")
SAVE_PATH.mkdir(exist_ok=True)

# FIX 1: cap at 4 — see module docstring
N_JOBS = 4

START_POINTS = [
    27000, 290000, 551000, 784000, 1050000, 1262000, 1484000,
    1748000, 1993000, 2287000, 2551000, 2812000, 3072000,
    3335000, 3599000,
]
END_POINTS = [
    262000, 523000, 757000, 1022000, 1235000, 1457000, 1721000,
    1964000, 2258000, 2524000, 2786000, 3045000, 3307000,
    3573000, 3805000,
]
LABEL = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]


# ══════════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSING
# ══════════════════════════════════════════════════════════════════════════

def count_windows(n_samples: int, window: int, step: int) -> int:
    if n_samples < window:
        return 0
    return ((n_samples - window) // step) + 1


def bandpass_filter(data: np.ndarray, sfreq: float, low=1.0, high=50.0, order=4) -> np.ndarray:
    nyq = 0.5 * sfreq
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data, axis=1)


def notch_filter(data: np.ndarray, sfreq: float, freq=50.0, q=30.0) -> np.ndarray:
    b, a = iirnotch(freq / (0.5 * sfreq), q)
    return filtfilt(b, a, data, axis=1)


def create_windows(trial: NDArray[np.floating], sfreq: float, window_sec=2.0, overlap=0.5):
    window = int(window_sec * sfreq)
    step   = int(window * (1 - overlap))
    for i in range(0, trial.shape[1] - window + 1, step):
        yield trial[:, i : i + window]


# ══════════════════════════════════════════════════════════════════════════
# STOCKWELL TRANSFORM  (corrected kernel)
# ══════════════════════════════════════════════════════════════════════════

def stockwell_transform_batch(
    signals: np.ndarray,
    sfreq: float,
    fmin: float = 1.0,
    fmax: float = 50.0,
    n_freqs: int = 75,          # was 20 → too few bins for narrow bands
) -> tuple[np.ndarray, np.ndarray]:
    n_channels, n_samples = signals.shape
    freqs  = np.logspace(np.log10(fmin), np.log10(fmax), n_freqs)
    f_axis = np.fft.fftfreq(n_samples, d=1.0 / sfreq)

    fft_signals = np.fft.fft(signals, axis=1).astype(np.complex64)
    st = np.zeros((n_channels, n_freqs, n_samples), dtype=np.complex64)

    for i, f in enumerate(freqs):
        # Gaussian centred at `f`, not at DC (was (f_axis**2 / f**2))
        gauss = np.exp(-2.0 * np.pi**2 * (f_axis - f)**2 / (f**2 + 1e-8)).astype(np.complex64)
        st[:, i, :] = np.fft.ifft(fft_signals * gauss, axis=1)

    return np.abs(st), freqs


def stockwell_de(window: np.ndarray, sfreq: float) -> list[float]:
    """DE = 0.5 * log(2πe * σ²) per band per channel."""
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 50)]
    st, freqs = stockwell_transform_batch(window, sfreq)
    feats: list[float] = []
    for low, high in bands:
        idx = np.where((freqs >= low) & (freqs <= high))[0]
        if len(idx) == 0:
            feats.extend([0.0] * window.shape[0])
            continue
        band_amp = st[:, idx, :].mean(axis=1)
        sigma2   = np.var(band_amp, axis=1) + 1e-10
        feats.extend((0.5 * np.log(2 * np.pi * np.e * sigma2)).tolist())
    return feats


# ══════════════════════════════════════════════════════════════════════════
# PER-WINDOW WORKER
# ══════════════════════════════════════════════════════════════════════════

def process_window(window_raw: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    w = bandpass_filter(window_raw, sfreq)
    w = notch_filter(w, sfreq)
    w = (w - w.mean(axis=1, keepdims=True)) / (w.std(axis=1, keepdims=True) + 1e-6)
    feat = stockwell_de(w, sfreq)
    return w.astype(np.float32), np.array(feat, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════
# FILE PROCESSOR
# ══════════════════════════════════════════════════════════════════════════

def process_file(file: Path) -> None:
    subject, session = map(int, file.stem.split("_"))
    print(f"\nProcessing {file.stem}  (subject={subject} session={session})")

    trial_ids: list[str] = []

    with mne.io.read_raw_cnt(file, preload=False) as raw:
        drop = [ch for ch in ["M1", "M2", "VEO", "HEO"] if ch in raw.ch_names]
        if drop:
            raw.drop_channels(drop)

        sfreq       = raw.info["sfreq"]
        window_size = int(2 * sfreq)
        step_size   = int(window_size * 0.5)
        n_channels  = len(raw.ch_names)

        total_windows = sum(
            count_windows(end - start, window_size, step_size)
            for start, end in zip(START_POINTS, END_POINTS, strict=False)
        )
        print(f"  Expected windows : {total_windows}")

        raw_path = SAVE_PATH / f"{file.stem}_raw.npy"

        # FIX 3: open_memmap is the final file — no np.save() at the end.
        # Only written pages are resident; the full array is never in RAM.
        X_raw  = np.lib.format.open_memmap(
            raw_path, mode="w+", dtype=np.float32,
            shape=(total_windows, n_channels, window_size),
        )
        X_feat = np.empty((total_windows, n_channels * 5), dtype=np.float32)
        Yw     = np.empty(total_windows, dtype=np.int8)

        write_idx = 0

        for trial_idx, (start, end) in enumerate(zip(START_POINTS, END_POINTS, strict=False)):
            trial   = raw.get_data(start=start, stop=end).astype(np.float32, copy=False)
            windows = list(create_windows(trial, sfreq))

            # FIX 2: delete trial before spawning workers so workers
            # don't inherit a redundant copy of the full trial array.
            del trial

            # FIX 1: n_jobs=4, not -1
            results = Parallel(n_jobs=N_JOBS, backend="loky")(
                delayed(process_window)(w, sfreq) for w in windows
            )
            del windows

            for win_local_idx, (window, feat) in enumerate(results):
                X_raw[write_idx]  = window
                X_feat[write_idx] = feat
                Yw[write_idx]     = LABEL[trial_idx]
                trial_ids.append(f"{subject}_{session}_{trial_idx}_{win_local_idx}")
                write_idx += 1

            del results
            # FIX 4: prompt page release between trials
            gc.collect()

        X_raw.flush()

    actual = write_idx
    if actual != total_windows:
        print(f"  ⚠ Trimming {total_windows} → {actual}")
        X_feat    = X_feat[:actual]
        Yw        = Yw[:actual]
        trial_ids = trial_ids[:actual]

    df = pd.DataFrame(X_feat)
    df["label"]    = Yw
    df["subject"]  = subject
    df["session"]  = session
    df["trial_id"] = trial_ids
    df.to_parquet(SAVE_PATH / f"{file.stem}.parquet")

    print(f"  ✓ {actual} windows → {file.stem}.parquet + {file.stem}_raw.npy")

    # FIX 4: release memmap and feature array before next file
    del X_raw, X_feat, Yw, df
    gc.collect()


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cnt_files = sorted(p for p in DATA_PATH.iterdir() if p.suffix == ".cnt")
    skipped   = [p.name for p in DATA_PATH.iterdir() if p.suffix != ".cnt" and p.is_file()]

    if skipped:
        print(f"Skipping: {skipped}")

    print(f"Found {len(cnt_files)} .cnt files")

    for file in cnt_files:
        process_file(file)

    print("\nDone.")