from pathlib import Path
import mne
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from joblib import Parallel, delayed


DATA_PATH = Path("./data/SEED_EEG/SEED_RAW_EEG")
SAVE_PATH = Path("./SEED_EEG/processed_data/")
SAVE_PATH.mkdir(exist_ok=True)

# Used from ./data/SEED_EEG/SEED_RAW_EEG/time.txt

START_POINTS = [27000,290000,551000,784000,1050000,1262000,1484000,
                1748000,1993000,2287000,2551000,2812000,3072000,
                3335000,3599000]

END_POINTS = [262000,523000,757000,1022000,1235000,1457000,1721000,
              1964000,2258000,2524000,2786000,3045000,3307000,
              3573000,3805000]

LABEL = [ 1,  0, -1, -1,  0,  1, -1,  0,  1,  1,  0, -1,  0,  1, -1]

files = [path for path in DATA_PATH.iterdir() if path.is_file()]


def count_windows(n_samples: int, window: int, step: int) -> int:
    """Return the number of sliding windows for a given sample length."""
    if n_samples < window:
        return 0
    return ((n_samples - window) // step) + 1


def bandpass_filter(data, sfreq, low=1, high=50, order=4):
    nyq = 0.5 * sfreq
    low /= nyq
    high /= nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=1)


def notch_filter(data, sfreq, freq=50, q=30):
    nyq = 0.5 * sfreq
    w0 = freq / nyq
    b, a = iirnotch(w0, q)
    return filtfilt(b, a, data, axis=1)


def create_windows(
    trial: NDArray[np.floating],
    sfreq: float,
    window_sec: float = 2,
    overlap: float = 0.5,
):
    window = int(window_sec * sfreq)
    step = int(window * (1 - overlap))

    for i in range(0, trial.shape[1] - window + 1, step):
        yield trial[:, i:i + window]


# =========================
# 🚀 VECTORIZED STOCKWELL
# =========================
def stockwell_transform_batch(
    signals: np.ndarray,   # (channels, samples)
    sfreq: float,
    fmin: float = 1,
    fmax: float = 50,
    n_freqs: int = 50,
):
    """
    Vectorized Stockwell Transform for multiple channels.
    """
    n_channels, n_samples = signals.shape

    freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_freqs)
    f_axis = np.fft.fftfreq(n_samples, d=1.0 / sfreq)

    fft_signals = np.fft.fft(signals, axis=1).astype(np.complex64)

    st = np.zeros((n_channels, n_freqs, n_samples), dtype=np.complex64)

    for i, f in enumerate(freqs):
        gaussian_freq = np.exp(-2.0 * np.pi**2 * f_axis**2 / f**2)

        shift = int(round(f * n_samples / sfreq))
        shifted_fft = np.roll(fft_signals, -shift, axis=1)

        # FIX: zero out the wrapped-around region to prevent frequency aliasing
        if shift > 0:
            shifted_fft[:, -shift:] = 0

        st[:, i, :] = np.fft.ifft(shifted_fft * gaussian_freq, axis=1)

    return np.abs(st), freqs


def stockwell_de(window: np.ndarray, sfreq: float) -> list[float]:
    """
    Converts a raw multi-channel EEG segment into Differential Entropy (DE) features.
    (Vectorized version)
    """
    bands = [
        (1,  4),
        (4,  8),
        (8,  13),
        (13, 30),
        (30, 50),
    ]

    st, freqs = stockwell_transform_batch(window, sfreq)

    feats = []

    for low, high in bands:
        idx = np.where((freqs >= low) & (freqs <= high))[0]

        if len(idx) == 0:
            feats.extend([0.0] * window.shape[0])
            continue

        band_st = st[:, idx, :]
        band_energy = np.mean(band_st ** 2, axis=(1, 2))

        de = np.log(np.maximum(band_energy, 1e-10))
        feats.extend(de.tolist())

    return feats


# =========================
# 🚀 PARALLEL WINDOW PROCESSING
# =========================
def process_window(window_raw, sfreq):
    window = bandpass_filter(window_raw, sfreq, 1, 50)
    window = notch_filter(window, sfreq, 50)
    window = window - window.mean(axis=1, keepdims=True)

    feat = stockwell_de(window, sfreq)
    return window.astype(np.float32), np.array(feat, dtype=np.float32)


# =========================
# 🚀 MAIN LOOP
# =========================
for file in files:
    subject, session = map(int, file.stem.split("_"))
    print(f"Processing {file.stem}")

    trial_ids = []

    with mne.io.read_raw_cnt(file, preload=False) as raw:
        raw: mne.io.BaseRaw

        channels_to_drop = [ch for ch in ["M1", "M2", "VEO", "HEO"] if ch in raw.ch_names]
        if channels_to_drop:
            raw.drop_channels(channels_to_drop)

        sfreq = raw.info["sfreq"]

        window_size = int(2 * sfreq)
        step_size = int(window_size * (1 - 0.5))
        n_channels = len(raw.ch_names)

        total_windows = 0
        for start, end in zip(START_POINTS, END_POINTS, strict=False):
            total_windows += count_windows(end - start, window_size, step_size)

        raw_staging_path = SAVE_PATH / f"{file.stem}_raw_staging.npy"

        X_raw = np.lib.format.open_memmap(
            raw_staging_path,
            mode="w+",
            dtype=np.float32,
            shape=(total_windows, n_channels, window_size),
        )

        X_feat = np.empty((total_windows, n_channels * 5), dtype=np.float32)
        Yw = np.empty(total_windows, dtype=np.int8)

        write_idx = 0

        for idx, (start, end) in enumerate(zip(START_POINTS, END_POINTS, strict=False)):
            trial = raw.get_data(start=start, stop=end).astype(np.float32, copy=False)

            windows = list(create_windows(trial, sfreq))

            results = Parallel(n_jobs=-1, backend="loky")(
                delayed(process_window)(w, sfreq) for w in windows
            )

            for window, feat in results:
                X_raw[write_idx] = window
                X_feat[write_idx] = feat
                Yw[write_idx] = LABEL[idx]
                trial_ids.append(f"{subject}_{session}_{idx}_{write_idx}")
                write_idx += 1

        X_raw.flush()

    if write_idx != total_windows:
        X_feat = X_feat[:write_idx]
        Yw = Yw[:write_idx]
        trial_ids = trial_ids[:write_idx]

    np.save(SAVE_PATH / f"{file.stem}_raw.npy", X_raw)
    del X_raw
    raw_staging_path.unlink(missing_ok=True)

    df = pd.DataFrame(X_feat)
    df["label"] = Yw
    df["subject"] = subject
    df["session"] = session
    df["trial_id"] = trial_ids

    df.to_parquet(SAVE_PATH / f"{file.stem}.parquet")