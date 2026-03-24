import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from mne.decoding import CSP

DATA_PATH = Path("./mini_data")


# ─────────────────────────────
# LOAD (BINARY)
# ─────────────────────────────
def load():
    raws, tabs, ys, trials, sessions = [], [], [], [], []

    files = sorted(DATA_PATH.glob("*.parquet"))

    for f in files:
        raw_file = DATA_PATH / f"{f.stem}_raw.npy"

        raw = np.load(raw_file)
        df = pd.read_parquet(f)

        X_tab = df.drop(columns=["label", "trial"]).values
        y = df["label"].values
        trial = df["trial"].values

        # 🔥 REMOVE NEUTRAL
        mask = y != 0

        raw = raw[mask]
        X_tab = X_tab[mask]
        y = y[mask]
        trial = trial[mask]

        # 🔥 MAP TO BINARY
        y = (y == 1).astype(int)

        session_id = f.stem

        raws.append(raw)
        tabs.append(X_tab)
        ys.append(y)
        trials.append(trial)
        sessions.append(np.full(len(y), session_id))

    return (
        np.concatenate(raws),
        np.concatenate(tabs),
        np.concatenate(ys),
        np.concatenate(trials),
        np.concatenate(sessions),
    )


# ─────────────────────────────
# CSP (BINARY)
# ─────────────────────────────
class BinaryCSP:
    def __init__(self):
        self.csp = CSP(
            n_components=6,
            log=True,
            reg=0.1
        )

    def fit(self, X, y):
        self.csp.fit(X, y)

    def transform(self, X):
        return self.csp.transform(X)


# ─────────────────────────────
# TRAIN
# ─────────────────────────────
def run():

    X_raw, X_tab, y, trials, sessions = load()

    print("Data shape:", X_raw.shape, X_tab.shape)
    print("Class balance:", np.bincount(y))

    sgkf = StratifiedGroupKFold(n_splits=3)

    scores = []

    for fold, (tr, te) in enumerate(
        sgkf.split(X_raw, y, groups=sessions), 1
    ):

        print(f"\nFOLD {fold}")

        # ── CSP
        csp = BinaryCSP()
        csp.fit(X_raw[tr], y[tr])

        X_tr = np.concatenate([X_tab[tr], csp.transform(X_raw[tr])], axis=1)
        X_te = np.concatenate([X_tab[te], csp.transform(X_raw[te])], axis=1)

        print("Feature shape:", X_tr.shape)

        # ── SCALE
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        # ── 🔥 SVM (KEY UPGRADE)
        clf = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale"
        )

        clf.fit(X_tr, y[tr])

        acc = clf.score(X_te, y[te])
        print("Fold acc:", acc)

        scores.append(acc)

    print("\nMEAN:", np.mean(scores))


# ─────────────────────────────
# ENTRY
# ─────────────────────────────
if __name__ == "__main__":
    run()
