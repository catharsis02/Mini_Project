import numpy as np
import mne
from pathlib import Path
from scipy.signal import butter, filtfilt
import pandas as pd

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
DATA_PATH = Path("./data/SEED_EEG/SEED_RAW_EEG")
SAVE_PATH = Path("./mini_data")
SAVE_PATH.mkdir(exist_ok=True)

SFREQ = 1000
WINDOW = 2000
STEP = 1000
MAX_WINDOWS_PER_TRIAL = 10

BANDS = [(1,4),(4,8),(8,13),(13,30),(30,50)]

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

LABEL = [1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1]


# ─────────────────────────────
# FILTER
# ─────────────────────────────
def bandpass(data):
    b, a = butter(4, [1/500, 50/500], btype="band")
    return filtfilt(b, a, data, axis=1)


# ─────────────────────────────
# DE FEATURE (bandpower)
# ─────────────────────────────
def compute_de(window):
    feats = []

    for low, high in BANDS:
        b, a = butter(4, [low/(SFREQ/2), high/(SFREQ/2)], btype="band")
        filtered = filtfilt(b, a, window, axis=1)

        var = np.var(filtered, axis=1) + 1e-10
        de = 0.5 * np.log(2 * np.pi * np.e * var)

        feats.append(de)

    feats = np.stack(feats, axis=1)  # (channels, bands)
    return feats


# ─────────────────────────────
# RASM (LEFT-RIGHT ASYMMETRY)
# ─────────────────────────────
def compute_rasm(de_feats):
    LEFT =  [0, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    RIGHT = [2, 4, 6, 8,10, 12, 14, 16, 18, 20]

    rasm = []

    for l, r in zip(LEFT, RIGHT):
        rasm.append(de_feats[l] / (de_feats[r] + 1e-6))

    return np.array(rasm).flatten()

# ─────────────────────────────
# MAIN
# ─────────────────────────────
def process_file(file):

    print(f"\nProcessing {file.name}")

    raw = mne.io.read_raw_cnt(file, preload=True)
    raw.drop_channels([ch for ch in ["M1","M2","VEO","HEO"] if ch in raw.ch_names])

    X_raw, X_feat, y, trials = [], [], [], []

    for t, (s, e) in enumerate(zip(START_POINTS, END_POINTS)):

        trial = raw.get_data(start=s, stop=e)
        trial = bandpass(trial)

        count = 0
        for i in range(0, trial.shape[1] - WINDOW, STEP):

            if count >= MAX_WINDOWS_PER_TRIAL:
                break

            w = trial[:, i:i+WINDOW]

            # ── DE
            de = compute_de(w)   # (channels, bands)

            # flatten DE
            de_flat = de.flatten()

            # ── RASM
            rasm = compute_rasm(de)

            # combine
            feat = np.concatenate([de_flat, rasm])

            # normalize raw (for CSP)
            w_norm = (w - w.mean(axis=1, keepdims=True)) / (
                w.std(axis=1, keepdims=True) + 1e-6
            )

            X_raw.append(w_norm.astype(np.float32))
            X_feat.append(feat)
            y.append(LABEL[t])
            trials.append(f"{file.stem}_{t}")

            count += 1

    X_raw = np.stack(X_raw)
    X_feat = np.stack(X_feat)

    np.save(SAVE_PATH / f"{file.stem}_raw.npy", X_raw)

    df = pd.DataFrame(X_feat, columns=[f"f{i}" for i in range(X_feat.shape[1])])
    df["label"] = y
    df["trial"] = trials

    df.to_parquet(SAVE_PATH / f"{file.stem}.parquet")

    print(f"{file.name} → raw:{X_raw.shape} feat:{X_feat.shape}")


# ─────────────────────────────
# ENTRY
# ─────────────────────────────
if __name__ == "__main__":
    files = sorted(DATA_PATH.glob("*.cnt"))
    selected = [f for f in files if f.stem.startswith(("1_", "2_"))]

    for f in selected:
        process_file(f)
