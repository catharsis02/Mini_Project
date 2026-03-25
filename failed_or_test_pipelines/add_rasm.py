"""
Add RASM features to existing bandpower data
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path("./mini_data_stockwell")
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]


def compute_rasm_from_de(de_array, n_channels=62, n_bands=5):
    """
    Compute RASM from flattened DE features.

    Args:
        de_array: (n_bands * n_channels,) flat array
        n_channels: number of EEG channels
        n_bands: number of frequency bands

    Returns:
        (n_pairs * n_bands,) RASM features
    """
    # Reshape to (n_bands, n_channels)
    de = de_array.reshape(n_bands, n_channels)

    # EEG channel pairs (left-right homologous regions)
    LEFT =  [0, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    RIGHT = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    rasm = []
    for band_idx in range(n_bands):
        for l, r in zip(LEFT, RIGHT):
            ratio = de[band_idx, l] / (de[band_idx, r] + 1e-6)
            rasm.append(ratio)

    return np.array(rasm, dtype=np.float32)


def add_rasm_to_file(parquet_file):
    """Add RASM features to a single parquet file."""
    print(f"\nProcessing {parquet_file.name}")

    # Load existing data
    df = pd.read_parquet(parquet_file)

    # Get DE feature columns
    feat_cols = [c for c in df.columns if c not in ('label', 'subject', 'session', 'trial_id')]
    print(f"  Current features: {len(feat_cols)}")

    # Check if RASM already exists
    if any('rasm' in c for c in feat_cols):
        print(f"  ⚠ RASM already exists, skipping")
        return

    X_de = df[feat_cols].values

    # Compute RASM for all samples
    rasm_features = []
    for i in range(len(X_de)):
        rasm = compute_rasm_from_de(X_de[i])
        rasm_features.append(rasm)

    rasm_features = np.array(rasm_features)

    # Create RASM column names
    rasm_cols = [f"rasm_{band}_pair{p}" for band in BAND_NAMES for p in range(10)]

    # Add RASM features to dataframe
    for i, col in enumerate(rasm_cols):
        df[col] = rasm_features[:, i]

    # Save back
    df.to_parquet(parquet_file)

    print(f"  ✓ Added {len(rasm_cols)} RASM features")
    print(f"  Total features: {len(feat_cols) + len(rasm_cols)}")


def main():
    print("=" * 60)
    print("Adding RASM Features to Bandpower Data")
    print("=" * 60)

    parquet_files = sorted(DATA_PATH.glob("*.parquet"))
    print(f"Found {len(parquet_files)} files")

    for pf in parquet_files:
        try:
            add_rasm_to_file(pf)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
