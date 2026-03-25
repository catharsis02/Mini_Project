"""
SEED EEG — Preprocessing Pipeline  (hardened v2)
=================================================
Memory budget : 8 GB RAM
CPU budget    : 8 cores

Bug fixes vs original
──────────────────────────────────────────────────────────────────────
  BUG-1  trial_id included win_local_idx → every window got a unique
         ID → GroupKFold treated each window as its own trial.
         FIX: f"{subject}_{session}_{trial_idx}" — shared by all
              windows of the same trial.

  BUG-2  pd.DataFrame(X_feat) produced integer column names ('0','1'…)
         → detect_de_layout() fell through to 'channel_major' →
         de_to_3d() scrambled channel-band pairings → DASM/RASM noise.
         FIX: explicit band-major column names 'delta_ch0'…'gamma_ch61'

Hardening additions
──────────────────────────────────────────────────────────────────────
  HARD-1  Assert START_POINTS / END_POINTS / LABEL have equal length.
           A copy-paste error here would silently corrupt all windows.

  HARD-2  Empty-windows guard.
           If a trial has fewer samples than window_size (shouldn't
           happen with SEED's fixed boundaries, but guards re-runs on
           partial files), the trial is skipped with a warning instead
           of passing an empty list to Parallel() which would produce
           zero results and desync the write_idx counter.

  HARD-3  Verify actual == total_windows after writing.
           If there's a mismatch (impossible in normal operation but
           catches future changes to START/END points), the parquet
           is still saved correctly for 'actual' rows.  The raw memmap
           is NOT trimmed here because open_memmap has already flushed
           it at size total_windows.  A mismatch print makes it visible.

Memory fixes (retained from previous version)
──────────────────────────────────────────────
  MEM-1  n_jobs=4 (not -1)
  MEM-2  del trial before Parallel()
  MEM-3  memmap IS the final .npy — flush() only, no np.save()
  MEM-4  gc.collect() after every trial
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

# ── CONFIG ───────────────────────────────────────────────────────────────────

DATA_PATH = Path("./data/SEED_EEG/SEED_RAW_EEG")
SAVE_PATH = Path("./data/SEED_EEG/processed_data/")
SAVE_PATH.mkdir(exist_ok=True)

N_JOBS = 4

START_POINTS = [
    27000,   290000,  551000,  784000,  1050000, 1262000, 1484000,
    1748000, 1993000, 2287000, 2551000, 2812000, 3072000, 3335000, 3599000,
]
END_POINTS = [
    262000,  523000,  757000,  1022000, 1235000, 1457000, 1721000,
    1964000, 2258000, 2524000, 2786000, 3045000, 3307000, 3573000, 3805000,
]
LABEL = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

# HARD-1: catch copy-paste errors in the trial boundary tables
assert len(START_POINTS) == len(END_POINTS) == len(LABEL), (
    f"START_POINTS({len(START_POINTS)}), END_POINTS({len(END_POINTS)}), "
    f"LABEL({len(LABEL)}) must all have the same length"
)

BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
N_BANDS    = len(BAND_NAMES)   # 5


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def count_windows(n_samples: int, window: int, step: int) -> int:
    if n_samples < window:
        return 0
    return ((n_samples - window) // step) + 1


def bandpass_filter(data: np.ndarray, sfreq: float,
                    low: float = 1.0, high: float = 50.0, order: int = 4) -> np.ndarray:
    nyq  = 0.5 * sfreq
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data, axis=1)


def notch_filter(data: np.ndarray, sfreq: float,
                 freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    b, a = iirnotch(freq / (0.5 * sfreq), q)
    return filtfilt(b, a, data, axis=1)


def create_windows(
    trial: NDArray[np.floating],
    window: int,
    step: int,
):
    """
    Yield (window, step)-strided slices of trial along axis 1.

    Both `window` and `step` are in samples.  Passing them explicitly
    (rather than deriving from sfreq inside the function) keeps
    create_windows and count_windows using identical values.
    """
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
    """
    Batch Stockwell (S-transform) over all channels.

    Returns
    -------
    st    : |S(τ, f)|  shape (n_channels, n_freqs, n_samples)
    freqs : 1-D float array, shape (n_freqs,)
    """
    n_channels, n_samples = signals.shape
    freqs       = np.logspace(np.log10(fmin), np.log10(fmax), n_freqs)
    f_axis      = np.fft.fftfreq(n_samples, d=1.0 / sfreq)
    fft_signals = np.fft.fft(signals, axis=1).astype(np.complex64)
    st          = np.zeros((n_channels, n_freqs, n_samples), dtype=np.complex64)

    for i, f in enumerate(freqs):
        # Gaussian centred at frequency f (corrected kernel vs original)
        gauss       = np.exp(
            -2.0 * np.pi**2 * (f_axis - f)**2 / (f**2 + 1e-8)
        ).astype(np.complex64)
        st[:, i, :] = np.fft.ifft(fft_signals * gauss, axis=1)

    return np.abs(st), freqs


def stockwell_de(window: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Differential Entropy per frequency band per channel.

    DE = 0.5 * log(2πe * σ²)

    Output layout is band-major (matches BAND_NAMES order):
        [δ_ch0 … δ_ch61 | θ_ch0 … θ_ch61 | … | γ_ch0 … γ_ch61]
    = N_BANDS × n_channels = 5 × 62 = 310 values.

    The explicit column names written to the parquet ('delta_ch0' …
    'gamma_ch61') preserve this layout so detect_de_layout() works.

    Returns
    -------
    1-D float32 array of length n_channels * N_BANDS
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
        sigma2   = np.var(band_amp, axis=1) + 1e-10   # floor > 0
        feats.extend((0.5 * np.log(2 * np.pi * np.e * sigma2)).tolist())
    return np.array(feats, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# PER-WINDOW WORKER
# ══════════════════════════════════════════════════════════════════════════════

def process_window(
    window_raw: np.ndarray, sfreq: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter, normalise, and extract DE features from one EEG window.

    Returns
    -------
    w    : (n_channels, n_samples) float32 — cleaned raw signal
    feat : (n_channels * N_BANDS,) float32 — DE features
    """
    w = bandpass_filter(window_raw, sfreq)
    w = notch_filter(w, sfreq)
    # Per-channel z-score (within window)
    w = (w - w.mean(axis=1, keepdims=True)) / (w.std(axis=1, keepdims=True) + 1e-6)
    feat = stockwell_de(w, sfreq)
    return w.astype(np.float32), feat


# ══════════════════════════════════════════════════════════════════════════════
# FILE PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

def process_file(file: Path) -> None:
    parts = file.stem.split("_")
    if len(parts) != 2:
        print(f"  ✗  Unexpected filename format: {file.name}  (expected subject_session.cnt)")
        return
    subject, session = int(parts[0]), int(parts[1])
    print(f"\nProcessing {file.stem}  (subject={subject}  session={session})")

    trial_ids: list[str] = []

    with mne.io.read_raw_cnt(file, preload=False) as raw:
        drop = [ch for ch in ["M1", "M2", "VEO", "HEO"] if ch in raw.ch_names]
        if drop:
            raw.drop_channels(drop)

        sfreq       = raw.info["sfreq"]
        window_size = int(2.0 * sfreq)
        step_size   = int(window_size * 0.5)    # 50 % overlap
        n_channels  = len(raw.ch_names)

        # Both count_windows and create_windows use window_size + step_size,
        # so their window counts are guaranteed to agree.
        total_windows = sum(
            count_windows(end - start, window_size, step_size)
            for start, end in zip(START_POINTS, END_POINTS)
        )
        print(f"  Expected windows : {total_windows}  |  "
              f"Channels : {n_channels}  |  sfreq : {sfreq} Hz")

        raw_path = SAVE_PATH / f"{file.stem}_raw.npy"

        # MEM-3: open_memmap IS the final .npy — flush() only, no np.save()
        X_raw  = np.lib.format.open_memmap(
            raw_path, mode="w+", dtype=np.float32,
            shape=(total_windows, n_channels, window_size),
        )
        X_feat = np.empty((total_windows, n_channels * N_BANDS), dtype=np.float32)
        Yw     = np.empty(total_windows, dtype=np.int8)

        write_idx = 0

        for trial_idx, (start, end) in enumerate(zip(START_POINTS, END_POINTS)):
            trial   = raw.get_data(start=start, stop=end).astype(np.float32, copy=False)
            windows = list(create_windows(trial, window_size, step_size))
            del trial   # MEM-2: free before spawning workers

            # HARD-2: guard for empty windows (e.g. truncated files)
            if len(windows) == 0:
                print(f"  ⚠  Trial {trial_idx}: 0 windows produced, skipping")
                gc.collect()
                continue

            # MEM-1: n_jobs=4, not -1
            results = Parallel(n_jobs=N_JOBS, backend="loky")(
                delayed(process_window)(w, sfreq) for w in windows
            )
            del windows

            # BUG-1 FIX: trial_group_id has NO window index.
            # All windows from the same trial share one ID so GroupKFold
            # holds out entire trials rather than random windows.
            trial_group_id = f"{subject}_{session}_{trial_idx}"

            for window, feat in results:
                X_raw[write_idx]  = window
                X_feat[write_idx] = feat
                Yw[write_idx]     = LABEL[trial_idx]
                trial_ids.append(trial_group_id)
                write_idx += 1

            del results
            gc.collect()   # MEM-4: prompt OS page release

        X_raw.flush()   # MEM-3: persist memmap to disk

    actual = write_idx

    # HARD-3: detect window count mismatch
    if actual != total_windows:
        print(
            f"  ⚠  Window count mismatch: expected {total_windows}, got {actual}.\n"
            f"     Raw memmap has {total_windows} rows but parquet will have {actual}.\n"
            f"     This will cause a crash in the ML pipeline.  "
            f"Check START/END_POINTS vs the file length."
        )
        X_feat    = X_feat[:actual]
        Yw        = Yw[:actual]
        trial_ids = trial_ids[:actual]
        # Note: X_raw is NOT trimmed here — it was flushed at total_windows size.
        # If you hit this branch, re-run preprocessing after fixing START/END_POINTS.

    # BUG-2 FIX: explicit band-major column names.
    # detect_de_layout() checks feat_cols[0].startswith('delta') → 'band_major'
    # de_to_3d() can then correctly reshape (n, 310) → (n, 5, 62) → transpose → (n, 62, 5)
    feat_cols = [
        f"{band}_ch{ch}"
        for band in BAND_NAMES
        for ch in range(n_channels)
    ]

    df = pd.DataFrame(X_feat, columns=feat_cols)
    df["label"]    = Yw
    df["subject"]  = subject
    df["session"]  = session
    df["trial_id"] = trial_ids

    out_parquet = SAVE_PATH / f"{file.stem}.parquet"
    df.to_parquet(out_parquet)

    # Verification prints — check these before running the ML pipeline
    label_dist = {int(k): int(v)
                  for k, v in zip(*np.unique(Yw[:actual], return_counts=True))}
    n_unique_trials = len(set(trial_ids))
    print(f"  ✓  {actual} windows  →  {out_parquet.name}  +  {raw_path.name}")
    print(f"     Label dist    : {label_dist}  (expect 3 classes)")
    print(f"     Unique trials : {n_unique_trials}  (expect 15)")
    print(f"     DE cols       : {feat_cols[0]} … {feat_cols[-1]}")
    print(f"     DE col count  : {len(feat_cols)}  (expect {n_channels * N_BANDS})")

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
        try:
            process_file(file)
        except Exception as exc:
            print(f"  ✗  {file.name} failed: {exc}")

    print("\nDone.")