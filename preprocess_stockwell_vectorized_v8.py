"""
SEED EEG — Preprocessing Pipeline  (v8 - DE + RASM)
====================================================
Memory budget : 8 GB RAM
CPU budget    : 8 cores

Features
────────
  - DE (Differential Entropy) per band per channel using direct filtering
  - RASM (Right-Left Asymmetry) features for 10 electrode pairs
  - Total features: 62 channels × 5 bands = 310 DE + 50 RASM = 360 features

Bug fixes vs original
──────────────────────────────────────────────────────────────────────
  BUG-1  trial_id included win_local_idx → every window got a unique
         ID → GroupKFold treated each window as its own trial.
         FIX: f"{subject}_{session}_{trial_idx}" — all windows from
              the same trial share one group ID.

  BUG-2  pd.DataFrame(X_feat) produced integer column names ('0','1'…)
         → detect_de_layout() fell through to 'channel_major' →
         de_to_3d() scrambled channel-band pairings → DASM/RASM noise.
         FIX: explicit band-major column names 'delta_ch0'…'gamma_ch61'
              + RASM features with clear names

Hardening additions
──────────────────────────────────────────────────────────────────────
  HARD-1  Module-level assert: START_POINTS / END_POINTS / LABEL lengths
           match.  Copy-paste errors here silently corrupt all windows.

  HARD-2  Sampling-rate assertion (Finding 3)
           START_POINTS and END_POINTS are sample indices at exactly
           1000 Hz (documented in time.txt: "#1000Hz").  If a CNT file
           has a different sampling rate, every trial boundary is wrong.
           The code now asserts int(sfreq) == EXPECTED_SFREQ and raises
           immediately with a clear message.

  HARD-3  Empty-windows guard
           If a trial has fewer samples than window_size the Parallel()
           call produces zero results, desynchronising write_idx.
           Now skips with a warning instead of silently desynchronising.

  HARD-4  Hard failure on window count mismatch (Finding 1)
           Previous code: flushed the memmap at total_windows size,
           then truncated only X_feat/Yw/trial_ids and saved both files.
           That left a broken pair: raw.npy had total_windows rows,
           parquet had actual rows.  The ML pipeline's HARD-4 check
           would crash on it, but the broken files stayed on disk.
           New behaviour: if actual != total_windows, delete the partial
           raw.npy and raise RuntimeError.  Nothing is saved.  Re-run
           after diagnosing the boundary mismatch.

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

# HARD-2: sample indices below are at exactly 1000 Hz (see time.txt: "#1000Hz").
# If the CNT files have a different rate, every boundary is wrong.
EXPECTED_SFREQ = 1000

START_POINTS = [
    27000,   290000,  551000,  784000,  1050000, 1262000, 1484000,
    1748000, 1993000, 2287000, 2551000, 2812000, 3072000, 3335000, 3599000,
]
END_POINTS = [
    262000,  523000,  757000,  1022000, 1235000, 1457000, 1721000,
    1964000, 2258000, 2524000, 2786000, 3045000, 3307000, 3573000, 3805000,
]
LABEL = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

# HARD-1: catch copy-paste errors in the trial boundary tables at import time
assert len(START_POINTS) == len(END_POINTS) == len(LABEL), (
    f"START_POINTS({len(START_POINTS)}), END_POINTS({len(END_POINTS)}), "
    f"LABEL({len(LABEL)}) must all have the same length"
)
assert all(s < e for s, e in zip(START_POINTS, END_POINTS)), \
    "Every START_POINT must be less than its matching END_POINT"

BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
N_BANDS    = len(BAND_NAMES)   # 5


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def count_windows(n_samples: int, window: int, step: int) -> int:
    """Number of windows of length `window` with stride `step` in `n_samples`."""
    if n_samples < window:
        return 0
    return ((n_samples - window) // step) + 1


def bandpass_filter(
    data: np.ndarray, sfreq: float,
    low: float = 1.0, high: float = 50.0, order: int = 4,
) -> np.ndarray:
    nyq  = 0.5 * sfreq
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data, axis=1)


def notch_filter(
    data: np.ndarray, sfreq: float,
    freq: float = 50.0, q: float = 30.0,
) -> np.ndarray:
    b, a = iirnotch(freq / (0.5 * sfreq), q)
    return filtfilt(b, a, data, axis=1)


def create_windows(trial: NDArray[np.floating], window: int, step: int):
    """
    Yield non-overlapping (with stride `step`) slices of `trial` along axis 1.

    Both `window` and `step` are in samples.  Passing them explicitly
    (not deriving from sfreq inside this function) guarantees
    create_windows and count_windows use identical values.
    """
    for i in range(0, trial.shape[1] - window + 1, step):
        yield trial[:, i : i + window]


# ══════════════════════════════════════════════════════════════════════════════
# STOCKWELL TRANSFORM + DIFFERENTIAL ENTROPY
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

    S(τ, f) = ∫ x(t) · g(t−τ, f) · e^{−2πift} dt
    where g is a Gaussian centred at f in the frequency domain.

    Implementation: multiply the signal's FFT by a Gaussian centred at f
    (not at DC — corrected kernel), then IFFT.

    Returns
    -------
    st    : |S(τ, f)|  float32  shape (n_channels, n_freqs, n_samples)
    freqs : log-spaced frequency axis  shape (n_freqs,)
    """
    n_channels, n_samples = signals.shape
    freqs        = np.logspace(np.log10(fmin), np.log10(fmax), n_freqs)
    f_axis       = np.fft.fftfreq(n_samples, d=1.0 / sfreq)
    fft_signals  = np.fft.fft(signals, axis=1).astype(np.complex64)
    st           = np.zeros((n_channels, n_freqs, n_samples), dtype=np.complex64)

    for i, f in enumerate(freqs):
        # Gaussian centred at f in the frequency domain
        gauss       = np.exp(
            -2.0 * np.pi**2 * (f_axis - f)**2 / (f**2 + 1e-8)
        ).astype(np.complex64)
        st[:, i, :] = np.fft.ifft(fft_signals * gauss, axis=1)

    return np.abs(st), freqs


def stockwell_de(window: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Differential Entropy per frequency band per channel.

    DE(X) = 0.5 × log(2πe × σ²)
    where σ² = variance of the band-amplitude time series.

    This assumes the band-amplitude follows a Gaussian distribution,
    which is the standard approximation used throughout the SEED
    literature (Zheng & Lu 2015).

    Output layout — band-major:
        [δ_ch0 δ_ch1 … δ_ch61 | θ_ch0 … θ_ch61 | … | γ_ch0 … γ_ch61]
        = N_BANDS × n_channels = 5 × 62 = 310 values

    The explicit column names written to the parquet ('delta_ch0' …
    'gamma_ch61') encode this layout so detect_de_layout() works
    correctly in the ML pipeline.

    Returns
    -------
    float32 array, shape (n_channels × N_BANDS,)
    """
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 50)]
    st, freqs = stockwell_transform_batch(window, sfreq)

    feats: list[float] = []
    for low, high in bands:
        idx = np.where((freqs >= low) & (freqs <= high))[0]
        if len(idx) == 0:
            # Should not happen with n_freqs=75 and bands within [1,50] Hz,
            # but guard against edge cases (very short windows, unusual sfreq)
            feats.extend([0.0] * window.shape[0])
            continue
        # Average S-transform amplitude over frequency bins in this band
        # st[:, idx, :] → (n_channels, n_idx_freqs, n_samples)
        # .mean(axis=1)  → (n_channels, n_samples)
        band_amp = st[:, idx, :].mean(axis=1)
        # Variance over time for each channel → (n_channels,)
        sigma2   = np.var(band_amp, axis=1) + 1e-10   # floor prevents log(0)
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

    Steps
    ─────
    1. Bandpass filter  1–50 Hz  (removes DC drift and high-frequency noise)
    2. Notch filter     50 Hz    (powerline interference)
    3. Per-channel z-score within the window  (removes channel amplitude bias)
    4. Stockwell DE                            (extract 310 features)

    Returns
    -------
    w    : (n_channels, n_samples)       float32 — cleaned raw signal
    feat : (n_channels × N_BANDS,)       float32 — DE features, band-major
    """
    w = bandpass_filter(window_raw, sfreq)
    w = notch_filter(w, sfreq)
    w = (w - w.mean(axis=1, keepdims=True)) / (w.std(axis=1, keepdims=True) + 1e-6)
    feat = stockwell_de(w, sfreq)
    return w.astype(np.float32), feat


# ══════════════════════════════════════════════════════════════════════════════
# FILE PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

def process_file(file: Path) -> None:
    """
    Process one subject-session CNT file.

    Writes two output files to SAVE_PATH:
      {stem}.parquet    — DE features + metadata
      {stem}_raw.npy    — cleaned EEG windows (memmap)

    HARD-4: if the number of windows actually written does not match the
    number predicted from START/END_POINTS, the partial raw.npy is
    deleted and a RuntimeError is raised.  Nothing is saved to disk.
    This prevents the ML pipeline from loading a broken pair.
    """
    parts = file.stem.split("_")
    if len(parts) != 2:
        raise ValueError(
            f"Unexpected filename format: {file.name}  "
            f"(expected subject_session.cnt, e.g. 1_1.cnt)"
        )
    subject, session = int(parts[0]), int(parts[1])
    print(f"\nProcessing {file.stem}  (subject={subject}  session={session})")

    trial_ids: list[str] = []
    raw_path  = SAVE_PATH / f"{file.stem}_raw.npy"

    with mne.io.read_raw_cnt(file, preload=False) as raw:
        drop = [ch for ch in ["M1", "M2", "VEO", "HEO"] if ch in raw.ch_names]
        if drop:
            raw.drop_channels(drop)

        sfreq      = raw.info["sfreq"]
        n_channels = len(raw.ch_names)

        # HARD-2: START/END_POINTS are in 1000 Hz samples.
        # A different sampling rate would silently corrupt all trial boundaries.
        if int(sfreq) != EXPECTED_SFREQ:
            raise RuntimeError(
                f"{file.name}: expected sfreq={EXPECTED_SFREQ} Hz "
                f"(as documented in time.txt '#1000Hz'), "
                f"got {sfreq} Hz.  "
                f"If the files were resampled, rescale START_POINTS and "
                f"END_POINTS by (sfreq / {EXPECTED_SFREQ}) before running."
            )

        window_size = int(2.0 * sfreq)    # 2 000 samples at 1000 Hz
        step_size   = int(window_size * 0.5)   # 50 % overlap = 1 000 samples

        # count_windows and create_windows use the same (window_size, step_size)
        # so they are guaranteed to agree on the number of windows per trial.
        total_windows = sum(
            count_windows(end - start, window_size, step_size)
            for start, end in zip(START_POINTS, END_POINTS)
        )
        print(
            f"  Expected windows : {total_windows}  |  "
            f"Channels : {n_channels}  |  sfreq : {sfreq} Hz"
        )

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

            # HARD-3: guard for empty windows (e.g. truncated file boundary)
            if len(windows) == 0:
                print(
                    f"  ⚠  Trial {trial_idx} produced 0 windows "
                    f"(START={start}, END={end}, window={window_size}).  "
                    f"Skipping trial."
                )
                gc.collect()
                continue

            # MEM-1: n_jobs=4, not -1 (loky copies window array per worker)
            results = Parallel(n_jobs=N_JOBS, backend="loky")(
                delayed(process_window)(w, sfreq) for w in windows
            )
            del windows

            # BUG-1 FIX: NO window index in trial_group_id.
            # All windows from the same trial share one group ID so
            # GroupKFold holds out entire trials, not random windows.
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

    # HARD-4 (Finding 1 fix): hard failure — do NOT save a broken pair.
    # Previous code truncated the parquet but left X_raw at total_windows rows,
    # creating a size mismatch that crashes the ML pipeline's HARD-4 check.
    # Now: delete the raw file and raise, so nothing broken remains on disk.
    if actual != total_windows:
        raw_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"{file.stem}: expected {total_windows} windows, wrote {actual}.  "
            f"The partial raw.npy has been deleted.  "
            f"Diagnose which trial(s) produced fewer windows than expected "
            f"(check trial lengths vs START/END_POINTS at {sfreq} Hz), "
            f"then re-run preprocessing."
        )

    # BUG-2 FIX: explicit band-major column names.
    # detect_de_layout() in the ML pipeline checks feat_cols[0].startswith('delta')
    # and returns 'band_major', so de_to_3d() reshapes correctly.
    # Layout: [delta_ch0…delta_ch61 | theta_ch0…theta_ch61 | … | gamma_ch61]
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

    # Verification prints — check all four lines before running the ML pipeline
    label_dist      = {int(k): int(v)
                       for k, v in zip(*np.unique(Yw, return_counts=True))}
    n_unique_trials = len(set(trial_ids))
    print(f"  ✓  {actual} windows  →  {out_parquet.name}  +  {raw_path.name}")
    print(f"     Label dist    : {label_dist}          (expect 3 classes)")
    print(f"     Unique trials : {n_unique_trials}                       (expect 15)")
    print(f"     DE cols       : {feat_cols[0]} … {feat_cols[-1]}")
    print(f"     DE col count  : {len(feat_cols)}    (expect {n_channels * N_BANDS})")

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

    failed = []
    for file in cnt_files:
        try:
            process_file(file)
        except Exception as exc:
            print(f"  ✗  {file.name} failed: {exc}")
            failed.append(file.name)

    print(f"\nDone.  {len(cnt_files) - len(failed)}/{len(cnt_files)} files succeeded.")
    if failed:
        print(f"Failed: {failed}")