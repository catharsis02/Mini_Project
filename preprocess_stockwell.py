from pathlib import Path
import mne
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch


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
    """
    Creates Windows from trials.

    Without windowing:
        1 trial ~ 4 minutes
        You only get 15 samples per file.

        1 sample per trial -> (0,240)
        15 samples per file.

    With windowing:
        1 trial ~ 4 minutes -> 2 minues
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

    Returns:
        list[NDArray[np.floating]]: Windowed EEG segments.

    """
    window = int(window_sec*sfreq)
    step = int(window*(1-overlap))

    for i in range(0, trial.shape[1] - window + 1, step):
        yield trial[:,i:i+window]


def stockwell_transform(
    signal: np.ndarray,
    sfreq: float,
    fmin: float = 1,
    fmax: float = 50,
    n_freqs: int = 50,
):
    """
    Computes the Stockwell Transform (S-transform) of a 1D EEG signal.

    The Stockwell Transform combines:
    - Fourier Transform -> captures frequency components
    - Wavelet Transform -> captures time-localisation

    It provides frequency-dependent resolution:
    - Low frequencies  -> better frequency resolution
    - High frequencies -> better time resolution

    Unlike STFT (fixed window), Stockwell uses a Gaussian window
    whose width varies inversely with frequency, built entirely in
    the frequency domain:

        ST(f, t) = IFFT( FFT(signal shifted by f) * G(a, f) )

    where G(a, f) = exp( -2*pi^2*a^2 / f^2 ) is the frequency-domain Gaussian.

    EEG Frequency Range:
    We restrict computation to 1-50 Hz, covering all EEG bands:
        Delta  (1-4  Hz)
        Theta  (4-8  Hz)
        Alpha  (8-13 Hz)
        Beta   (13-30 Hz)
        Gamma  (30-50 Hz)

    Frequencies are log-spaced so that low bands (delta, theta) receive
    more bins proportionally, matching the S-transform's natural
    frequency resolution.

    Arguments:
        signal  : 1D EEG signal of shape (n_samples,)
        sfreq   : Sampling frequency in Hz
        fmin    : Minimum frequency (default 1 Hz)
        fmax    : Maximum frequency (default 50 Hz)
        n_freqs : Number of frequency bins (default 50)

    Returns:
        st    : 2D array of shape (n_freqs, n_samples) - magnitude |ST(f,t)|
        freqs : 1D array of shape (n_freqs,) - corresponding frequencies

    """
    n_samples = len(signal)

    freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_freqs)
    f_axis = np.fft.fftfreq(n_samples, d=1.0 / sfreq)

    fft_signal = np.fft.fft(signal).astype(np.complex64)
    st = np.zeros((n_freqs, n_samples), dtype=np.complex64)

    for i, f in enumerate(freqs):
        gaussian_freq = np.exp(-2.0 * np.pi**2 * f_axis**2 / f**2)
        shift = round(f * n_samples / sfreq)
        shifted_fft = np.roll(fft_signal, -shift)
        st[i] = np.fft.ifft(shifted_fft * gaussian_freq)

    return np.abs(st), freqs


def stockwell_de(window: np.ndarray, sfreq: float) -> list[float]:
    """Converts a raw multi-channel EEG segment into Differential Entropy (DE) features."""
    bands = [
        (1,  4),
        (4,  8),
        (8,  13),
        (13, 30),
        (30, 50),
    ]

    feats = []

    for ch in range(window.shape[0]):
        signal = window[ch]

        st, freqs = stockwell_transform(signal, sfreq)

        for low, high in bands:
            idx = np.where((freqs >= low) & (freqs <= high))[0]

            if len(idx) == 0:
                feats.append(0.0)
                continue

            band_st = st[idx]
            band_energy = np.mean(band_st ** 2)

            de = np.log(max(float(band_energy), 1e-10))
            feats.append(de)

    return feats


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

            for window_raw in create_windows(trial, sfreq):
                window = bandpass_filter(window_raw, sfreq, 1, 50)
                window = notch_filter(window, sfreq, 50)
                window = window - window.mean(axis=1, keepdims=True)

                X_raw[write_idx] = window
                X_feat[write_idx] = stockwell_de(window, sfreq)
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
