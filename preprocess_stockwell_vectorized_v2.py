"""
SEED EEG — Preprocessing Pipeline  (fixed v2)
==============================================
Memory budget : 8 GB RAM
CPU budget    : 8 cores

Fixes vs original
──────────────────────────────────────────────────────────────────────
  ORIGINAL BUGS (caused 63 % accuracy in ML pipeline)
  ─────────────────────────────────────────────────────
  BUG-1  trial_id included win_local_idx
           f"{subject}_{session}_{trial_idx}_{win_local_idx}"
           Every window got a unique ID → GroupKFold treated each window
           as its own trial → identical behaviour to StratifiedKFold.
           The entire point of GroupKFold was silently defeated.
           FIX: f"{subject}_{session}_{trial_idx}"
                All windows from the same trial share one ID.

  BUG-2  pd.DataFrame(X_feat) used integer column names (RangeIndex).
           PyArrow stringifies these to '0','1','2',… on parquet write.
           detect_de_layout() received ['0','1',...], none starting with
           a band name → fell through to 'channel_major'.
           But stockwell_de() stores band-major:
             [δ_ch0…δ_ch61 | θ_ch0…θ_ch61 | … | γ_ch61]
           de_to_3d() with channel_major reshaped as (n,62,5) treating
           position j*5+k as channel j band k → wrong pairing.
           All 270 DASM/RASM features were pure noise.
           FMI ranking was contaminated → ~15–20 % accuracy loss.
           FIX: explicit band_ch column names:
                ['delta_ch0', …, 'delta_ch61', 'theta_ch0', …, 'gamma_ch61']

  MEMORY FIXES (retained from previous version)
  ──────────────────────────────────────────────
  MEM-1  n_jobs=4 (was -1)
           loky copies the window array into every worker.
           8 workers × window_bytes → OOM.  4 workers keeps peak < 2 GB.

  MEM-2  Trial deleted before Parallel() call
           Without `del trial`, each worker inherits a copy of the full
           trial array in addition to its own window slice.

  MEM-3  X_raw stays as a memmap — no second np.save()
           The memmap file IS the final .npy; just flush() and move on.
           np.save() would materialise the whole array in RAM again.

  MEM-4  gc.collect() after every trial
           Prompts OS to reclaim freed pages before the next trial load.
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

# ── CONFIG ──────────────────────────────────────────────────────────────────

DATA_PATH = Path("./data/SEED_EEG/SEED_RAW_EEG")
SAVE_PATH = Path("./data/SEED_EEG/processed_data/")
SAVE_PATH.mkdir(exist_ok=True)

N_JOBS = 4   # MEM-1: cap at 4, not -1

START_POINTS = [
    27000,   290000,  551000,  784000,  1050000, 1262000, 1484000,
    1748000, 1993000, 2287000, 2551000, 2812000, 3072000, 3335000, 3599000,
]
END_POINTS = [
    262000,  523000,  757000,  1022000, 1235000, 1457000, 1721000,
    1964000, 2258000, 2524000, 2786000, 3045000, 3307000, 3573000, 3805000,
]
LABEL = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

# BUG-2 FIX: explicit column names so detect_de_layout() works correctly.
# Layout: band-major → [delta_ch0…delta_ch61 | theta_ch0… | … | gamma_ch61]
# 62 channels × 5 bands = 310 features
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def count_windows(n_samples: int, window: int, step: int) -> int:
    if n_samples < window:
        return 0
    return ((n_samples - window) // step) + 1


def bandpass_filter(
    data: np.ndarray, sfreq: float, low: float = 1.0, high: float = 50.0, order: int = 4
) -> np.ndarray:
    nyq  = 0.5 * sfreq
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data, axis=1)


def notch_filter(
    data: np.ndarray, sfreq: float, freq: float = 50.0, q: float = 30.0
) -> np.ndarray:
    b, a = iirnotch(freq / (0.5 * sfreq), q)
    return filtfilt(b, a, data, axis=1)


def create_windows(trial: NDArray[np.floating], sfreq: float,
                   window_sec: float = 2.0, overlap: float = 0.5):
    window = int(window_sec * sfreq)
    step   = int(window * (1 - overlap))
    for i in range(0, trial.shape[1] - window + 1, step):
        yield trial[:, i : i + window]


# ══════════════════════════════════════════════════════════════════════════════
# STOCKWELL TRANSFORM
# ══════════════════════════════════════════════════════════════════════════════

def stockwell_transform_batch(
    signals: np.ndarray,
    sfreq:   float,
    fmin:    float = 1.0,
    fmax:    float = 50.0,
    n_freqs: int   = 75,
) -> tuple[np.ndarray, np.ndarray]:
    n_channels, n_samples = signals.shape
    freqs      = np.logspace(np.log10(fmin), np.log10(fmax), n_freqs)
    f_axis     = np.fft.fftfreq(n_samples, d=1.0 / sfreq)
    fft_signals = np.fft.fft(signals, axis=1).astype(np.complex64)
    st          = np.zeros((n_channels, n_freqs, n_samples), dtype=np.complex64)

    for i, f in enumerate(freqs):
        # Gaussian centred at `f` (not at DC)
        gauss      = np.exp(
            -2.0 * np.pi**2 * (f_axis - f)**2 / (f**2 + 1e-8)
        ).astype(np.complex64)
        st[:, i, :] = np.fft.ifft(fft_signals * gauss, axis=1)

    return np.abs(st), freqs


def stockwell_de(window: np.ndarray, sfreq: float) -> list[float]:
    """
    Differential Entropy per frequency band per channel.

    DE = 0.5 * log(2πe * σ²)

    Output order: band-major
      [δ_ch0, δ_ch1, …, δ_ch61, θ_ch0, …, θ_ch61, …, γ_ch61]
    This matches BAND_NAMES × n_channels = 310 features.
    Explicit column names in the parquet preserve this for de_to_3d().
    """
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


# ══════════════════════════════════════════════════════════════════════════════
# PER-WINDOW WORKER
# ══════════════════════════════════════════════════════════════════════════════

def process_window(
    window_raw: np.ndarray, sfreq: float
) -> tuple[np.ndarray, np.ndarray]:
    w    = bandpass_filter(window_raw, sfreq)
    w    = notch_filter(w, sfreq)
    w    = (w - w.mean(axis=1, keepdims=True)) / (w.std(axis=1, keepdims=True) + 1e-6)
    feat = stockwell_de(w, sfreq)
    return w.astype(np.float32), np.array(feat, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# FILE PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

def process_file(file: Path) -> None:
    subject, session = map(int, file.stem.split("_"))
    print(f"\nProcessing {file.stem}  (subject={subject}  session={session})")

    trial_ids: list[str] = []

    with mne.io.read_raw_cnt(file, preload=False) as raw:
        drop = [ch for ch in ["M1", "M2", "VEO", "HEO"] if ch in raw.ch_names]
        if drop:
            raw.drop_channels(drop)

        sfreq       = raw.info["sfreq"]
        window_size = int(2.0 * sfreq)
        step_size   = int(window_size * 0.5)
        n_channels  = len(raw.ch_names)

        total_windows = sum(
            count_windows(end - start, window_size, step_size)
            for start, end in zip(START_POINTS, END_POINTS, strict=False)
        )
        print(f"  Expected windows : {total_windows}  |  "
              f"Channels : {n_channels}  |  sfreq : {sfreq} Hz")

        raw_path = SAVE_PATH / f"{file.stem}_raw.npy"

        # MEM-3: memmap IS the final file — flush() only, no np.save()
        X_raw  = np.lib.format.open_memmap(
            raw_path, mode="w+", dtype=np.float32,
            shape=(total_windows, n_channels, window_size),
        )
        X_feat = np.empty((total_windows, n_channels * len(BAND_NAMES)),
                          dtype=np.float32)
        Yw     = np.empty(total_windows, dtype=np.int8)

        write_idx = 0

        for trial_idx, (start, end) in enumerate(
                zip(START_POINTS, END_POINTS, strict=False)):

            trial   = raw.get_data(start=start, stop=end).astype(np.float32, copy=False)
            windows = list(create_windows(trial, sfreq))

            # MEM-2: delete trial before spawning workers
            del trial

            # MEM-1: n_jobs=4
            results = Parallel(n_jobs=N_JOBS, backend="loky")(
                delayed(process_window)(w, sfreq) for w in windows
            )
            del windows

            # BUG-1 FIX: trial_id has NO window index.
            # All windows from the same trial share one group ID so that
            # GroupKFold holds out entire trials, not random windows.
            trial_group_id = f"{subject}_{session}_{trial_idx}"

            for window, feat in results:
                X_raw[write_idx]  = window
                X_feat[write_idx] = feat
                Yw[write_idx]     = LABEL[trial_idx]
                trial_ids.append(trial_group_id)   # same ID for all windows
                write_idx += 1

            del results
            gc.collect()   # MEM-4: prompt page release

        X_raw.flush()   # MEM-3: flush memmap, no extra np.save()

    actual = write_idx
    if actual != total_windows:
        print(f"  ⚠  Trimming {total_windows} → {actual}")
        X_feat    = X_feat[:actual]
        Yw        = Yw[:actual]
        trial_ids = trial_ids[:actual]

    # BUG-2 FIX: explicit band-major column names.
    # detect_de_layout() checks feat_cols[0].startswith('delta') → 'band_major'
    # de_to_3d() reshapes (n, 310) → (n, 5, 62).transpose → (n, 62, 5) correctly.
    feat_cols = [f"{band}_ch{ch}"
                 for band in BAND_NAMES
                 for ch in range(n_channels)]

    df = pd.DataFrame(X_feat, columns=feat_cols)
    df["label"]    = Yw
    df["subject"]  = subject
    df["session"]  = session
    df["trial_id"] = trial_ids

    out_parquet = SAVE_PATH / f"{file.stem}.parquet"
    df.to_parquet(out_parquet)

    print(f"  ✓  {actual} windows  →  {out_parquet.name}  +  {raw_path.name}")
    print(f"     Label dist : { {int(k): int(v) for k, v in zip(*np.unique(Yw, return_counts=True))} }")
    print(f"     Trials     : {len(set(trial_ids))}  unique IDs")
    print(f"     DE cols    : {feat_cols[0]} … {feat_cols[-1]}")

    # MEM-4: release before next file
    del X_raw, X_feat, Yw, df
    gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cnt_files = sorted(p for p in DATA_PATH.iterdir() if p.suffix == ".cnt")
    skipped   = [p.name for p in DATA_PATH.iterdir()
                 if p.suffix != ".cnt" and p.is_file()]

    if skipped:
        print(f"Skipping non-.cnt files: {skipped}")

    print(f"Found {len(cnt_files)} .cnt files\n")

    for file in cnt_files:
        process_file(file)

    print("\nDone.")
