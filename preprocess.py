from pathlib import Path
from collections.abc import Iterator
import mne
import numpy as np
from numpy.typing import NDArray
import pywt
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch

DATA_PATH = Path("./data/SEED_EEG/SEED_RAW_EEG")
SAVE_PATH = Path("./data/SEED_EEG/processed_data/")
SAVE_PATH.mkdir(parents=True, exist_ok=True)

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
    """Apply channel-wise Butterworth band-pass filtering."""
    nyq = 0.5 * sfreq
    low /= nyq
    high /= nyq

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=1)


def notch_filter(data, sfreq, freq=50, q=30):
    """Apply channel-wise notch filtering to suppress line noise."""
    nyq = 0.5 * sfreq
    w0 = freq / nyq

    b, a = iirnotch(w0, q)
    return filtfilt(b, a, data, axis=1)

def create_windows(
            trial: NDArray[np.floating],
            sfreq: float,
            window_sec: float = 2,
            overlap: float = 0.5,
    ) -> Iterator[NDArray[np.floating]]:
    """
    Creates Windows from trials.

    Without windowing:
        1 trial ~ 4 minutes
        You only get 15 samples per file.

        1 sample per trial -> (0,240)
        15 samples per file.

    With windowing:
        1 trial ~ 4 minutes -> 2 minutes
        if we take window_size as 2 seconds
        overlap -> 0.5

        we can divide 240 sec into
        (0,2),(1,3),(2,4),(3,5),......,(238,240)
        -> 238 samples per trial

        240 x 15 samples per file = 3600

        Why do we need to create windows :
        Brain Activity is not STABLE for entire
        periods of time, therefore we need to take small slices.

    Arguments:
        trial: EEG trial data shaped as (n_channels, n_samples).
        sfreq: Sampling frequency in Hz.
        window_sec: Window length in seconds.
        overlap: Fractional overlap between consecutive windows.

    Yields:
        NDArray[np.floating]: Windowed EEG segment.

    """
    window = int(window_sec * sfreq)
    step = int(window * (1 - overlap))

    for i in range(0, trial.shape[1] - window  + 1, step):
        yield trial[:, i:i+window]

def wavelet_de(window: NDArray[np.floating]) -> list[float]:
    """
    Converts raw EEG signal → frequency-based features (per channel).

    EEG Signals -> ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2', 'VEO', 'HEO']
    ["M1","M2","VEO","HEO"] are removed.

    Frontal -> FP1,FP2,F3,F4,FZ -> emotion, decision making
    Central -> C3,CZ,C4 -> motor activity
    Temporal -> T7,T8 -> auditory/emotion
    Parietal -> P3,PZ,P4 -> sensory integration
    Occipital -> O1,O2,OZ -> vision

    Wavelets split these signals(ex:FP1) into
    Delta   (1-4 Hz)
    Theta   (4-8 Hz)
    Alpha   (8-13 Hz)
    Beta    (13-30 Hz)
    Gamma   (30-50 Hz)

    Measuring Energy (ex:FP1):
    Delta energy  → how active low-frequency signals are
    Theta energy  → emotional processing
    Alpha energy  → relaxation
    Beta energy   → stress / focus
    Gamma energy  → high-level processing

    ``` coeffs = pywt.wavedec(window[ch],"db4",level = 5) ```

    coeffs = [cA5, cD5, cD4, cD3, cD2, cD1]
    cA5 -> Approximation (very low freq) -> <1 Hz
    cD5 -> Detail Level 5 -> Delta (1-4Hz)
    cD4 -> Detail Level 4 -> Theta (4-8 Hz)
    cD3 -> Detail Level 3 -> Alpha (8-13 Hz)
    cD2 -> Detail Level 2 -> Beta  (13-30 Hz)
    cD1 -> Detail Level 1 -> Gamma (30-50 Hz)

    db4 -> Daubechies wavelet (order 4)
    level 5 covers all important bands
    5 details


    var -> Measures energy

    Arguments:
    window: 2D EEG segment of shape (n_channels, n_samples), typically one sliding window from create_windows.

    Returns:
    list[float]: Feature vector where each value is log(var(band) + 1e-10) from wavedec coefficients for every channel. The length is n_channels multiplied by number of wavelet coefficient bands.

    """
    feats: list[float] = []

    for ch in range(window.shape[0]):
        coeffs = pywt.wavedec(window[ch], "db4", level=5)

        for band in coeffs:
            var = np.var(band)
            feats.append(np.log(var + 1e-10))

    return feats


for file in files:
    subject, session = map(int, file.stem.split("_"))
    print(f"Processing {file.stem}")

    trial_ids = []
    with mne.io.read_raw_cnt(file, preload=False) as raw:
        raw: mne.io.BaseRaw

        # M1, M2 -> reference electrodes, EEG does not measure absolute voltage, It measures difference between electrodes

        # VEO → vertical eye movement (blinks)
        # HEO → horizontal eye movement, do not capture brain activity
        channels_to_drop = [ch for ch in ["M1", "M2", "VEO", "HEO"] if ch in raw.ch_names]
        if channels_to_drop:
            raw.drop_channels(channels_to_drop)

        # Sampling Frequency = 1000Hz
        sfreq = raw.info["sfreq"]

        window_size = int(2 * sfreq)
        step_size = int(window_size * (1 - 0.5))
        n_channels = len(raw.ch_names)

        total_windows = 0
        for start, end in zip(START_POINTS, END_POINTS, strict=False):
            total_windows += count_windows(end - start, window_size, step_size)

        # Stage large raw arrays on disk so RAM usage stays low.
        raw_staging_path = SAVE_PATH / f"{file.stem}_raw_staging.npy"
        X_raw = np.lib.format.open_memmap(
            raw_staging_path,
            mode="w+",
            dtype=np.float32,
            shape=(total_windows, n_channels, window_size),
        )
        X_feat = np.empty((total_windows, n_channels * 6), dtype=np.float32)
        Yw = np.empty(total_windows, dtype=np.int8)

        write_idx = 0

        for idx, (start, end) in enumerate(zip(START_POINTS, END_POINTS, strict=False)):
            trial = raw.get_data(start=start, stop=end).astype(np.float32, copy=False)

            # Feature Extraction
            for window_raw in create_windows(trial, sfreq):
                # Apply filters (channel-wise)
                window = bandpass_filter(window_raw, sfreq, 1, 50)
                # Avoid Spikes at 50Hz -> drift + high frequency noise.
                window = notch_filter(window, sfreq, 50)
                # CSP-safe normalization: remove per-channel DC offset only.
                window = (window - window.mean(axis=1, keepdims=True)) / (
    window.std(axis=1, keepdims=True) + 1e-6
)

                X_raw[write_idx] = window
                X_feat[write_idx] = wavelet_de(window)
                Yw[write_idx] = LABEL[idx]
                trial_ids.append(f"{subject}_{session}_{idx}_{write_idx}")
                write_idx += 1

        X_raw.flush()

    # # CSP per file
    # csp = CSP(n_components=10)
    # X_csp = csp.fit_transform(Xw)

    if write_idx != total_windows:
        X_feat = X_feat[:write_idx]
        Yw = Yw[:write_idx]
        trial_ids = trial_ids[:write_idx]
        X_raw = X_raw[:write_idx]

    # Keep CSP input compressed for downstream loading.
    np.save(SAVE_PATH / f"{file.stem}_xraw.npy", X_raw)
    del X_raw
    raw_staging_path.unlink(missing_ok=True)

    df = pd.DataFrame(X_feat)
    df["label"] = Yw
    df["subject"] = subject
    df["session"] = session
    df["trial_id"] = trial_ids

    df.to_parquet(SAVE_PATH / f"{file.stem}.parquet")
