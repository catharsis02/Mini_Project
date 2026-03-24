"""
SEED EEG — CSP-ONLY DEBUG (OOM-SAFE)
===================================

Goal:
- Verify if CSP pipeline works
- If YES → preprocessing (DE) is broken
- If NO  → CSP/data pipeline broken
"""

import numpy as np
import pandas as pd
from pathlib import Path
import gc

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from mne.decoding import CSP


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

PROCESSED_PATH = Path("./data/SEED_EEG/processed_data/")
DECIMATE = 4
CSP_FIT_MAX_PER_CLASS = 300
CHUNK_SIZE = 512


# ─────────────────────────────────────────────
# MEMMAP COLLECTION
# ─────────────────────────────────────────────

class SessionMemmapCollection:
    def __init__(self, arrays):
        self.arrays = arrays
        self.lengths = np.array([a.shape[0] for a in arrays])
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)])
        self.shape = (int(self.offsets[-1]),) + arrays[0].shape[1:]

    def __len__(self):
        return int(self.offsets[-1])

    def load(self, idx, decimate=1):
        idx = np.asarray(idx)
        T = self.shape[2] // decimate
        out = np.empty((len(idx), self.shape[1], T), dtype=np.float32)

        for s, arr in enumerate(self.arrays):
            lo, hi = self.offsets[s], self.offsets[s+1]
            mask = (idx >= lo) & (idx < hi)
            if mask.any():
                out[np.where(mask)[0]] = arr[idx[mask]-lo][:, :, ::decimate]

        return out


# ─────────────────────────────────────────────
# LOAD SUBJECT
# ─────────────────────────────────────────────

def load_subject(subject_id):
    files = sorted(PROCESSED_PATH.glob(f"{subject_id}_*.parquet"))

    raw_memmaps = []
    y_all = []
    trials_all = []

    for pf in files:
        df = pd.read_parquet(pf)

        y = df["label"].values
        trials = df["trial_id"].values

        raw_path = PROCESSED_PATH / f"{pf.stem}_raw.npy"
        raw_memmaps.append(np.load(raw_path, mmap_mode="r"))

        y_all.append(y)
        trials_all.append(trials)

        print(f"{pf.name} → y:{y.shape}")

    return (
        SessionMemmapCollection(raw_memmaps),
        np.concatenate(y_all),
        np.concatenate(trials_all),
    )


# ─────────────────────────────────────────────
# CSP (OOM SAFE)
# ─────────────────────────────────────────────

class MulticlassCSP:
    def __init__(self, n_components=6):
        self.n_components = n_components

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.csps_ = {}

        for cls in self.classes_:
            csp = CSP(n_components=self.n_components, log=True)
            csp.fit(X, (y == cls).astype(int))
            self.csps_[cls] = csp

        return self

    def transform_chunked(self, collection, idx):
        feats = []

        for i in range(0, len(idx), CHUNK_SIZE):
            batch_idx = idx[i:i+CHUNK_SIZE]
            raw = collection.load(batch_idx, decimate=DECIMATE)

            chunk = np.concatenate(
                [self.csps_[c].transform(raw) for c in self.classes_],
                axis=1
            )

            feats.append(chunk)
            del raw, chunk
            gc.collect()

        return np.vstack(feats)


def stratified_subsample(idx, y):
    out = []
    for cls in np.unique(y[idx]):
        ci = idx[y[idx] == cls]
        if len(ci) > CSP_FIT_MAX_PER_CLASS:
            ci = np.random.choice(ci, CSP_FIT_MAX_PER_CLASS, replace=False)
        out.append(ci)
    return np.concatenate(out)


# ─────────────────────────────────────────────
# DEBUG CV (CSP ONLY)
# ─────────────────────────────────────────────

def cross_validate_csp_only(collection, y, trials):

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    idx = np.arange(len(y))
    scores = []

    for fold, (tr, te) in enumerate(sgkf.split(idx, y, groups=trials), 1):

        print("\n" + "="*60)
        print(f"FOLD {fold}")
        print("="*60)

        y_tr, y_te = y[tr], y[te]

        print("Train label dist:", np.unique(y_tr, return_counts=True))
        print("Test  label dist:", np.unique(y_te, return_counts=True))
        print("Trial overlap:", set(trials[tr]) & set(trials[te]))

        # ── CSP FIT (SUBSAMPLED)
        sub_idx = stratified_subsample(tr, y)
        raw_sub = collection.load(sub_idx, decimate=DECIMATE)

        csp = MulticlassCSP()
        csp.fit(raw_sub, y[sub_idx])

        del raw_sub
        gc.collect()

        # ── CSP TRANSFORM (CHUNKED)
        csp_tr = csp.transform_chunked(collection, tr)
        csp_te = csp.transform_chunked(collection, te)

        print("CSP stats:", np.mean(csp_tr), np.std(csp_tr))
        print("Feature shape:", csp_tr.shape)

        # ── SIMPLE CLASSIFIER
        clf = LogisticRegression(max_iter=1000)
        clf.fit(csp_tr, y_tr)

        y_pred = clf.predict(csp_te)

        print("Pred dist:", np.unique(y_pred, return_counts=True))

        acc = (y_pred == y_te).mean()
        print("Accuracy:", acc)

        scores.append(acc)

        del csp_tr, csp_te
        gc.collect()

    return np.array(scores)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    subject_id = 1

    collection, y, trials = load_subject(subject_id)

    scores = cross_validate_csp_only(collection, y, trials)

    print("\nFINAL:", scores)
    print("MEAN:", scores.mean())
