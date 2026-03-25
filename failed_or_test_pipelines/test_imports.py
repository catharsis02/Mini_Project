#!/usr/bin/env python3
print("Testing simple pipeline imports...")

try:
    import numpy as np
    print("✓ NumPy imported")
    import pandas as pd
    print("✓ Pandas imported")
    from sklearn.preprocessing import StandardScaler
    print("✓ Scikit-learn imported")
    from mne.decoding import CSP
    print("✓ MNE imported")
    print("All imports successful!")
except ImportError as e:
    print(f"Import error: {e}")