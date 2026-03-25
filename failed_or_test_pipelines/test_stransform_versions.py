"""
Rigorous testing of S-transform normalization variants.
Tests frequency detection, energy conservation, and time-frequency localization.
"""

import numpy as np
from scipy.signal import hilbert

def stockwell_transform_v1(signal, sfreq, fmin=1.0, fmax=50.0, n_freqs=75):
    """Version 1: WITH |f| scaling (USER'S FIX 4)"""
    n_samples = signal.shape[1]
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_freqs)
    f_axis = np.fft.fftfreq(n_samples, d=1.0/sfreq)
    fft_signal = np.fft.fft(signal, axis=1).astype(np.complex64)
    st = np.zeros((signal.shape[0], n_freqs, n_samples), dtype=np.complex64)

    for i, f in enumerate(freqs):
        sigma_f = 1.0 / (2.0 * np.pi * np.abs(f) + 1e-10)
        sigma_max = n_samples / (4.0 * sfreq)
        sigma_f = np.clip(sigma_f, 1e-10, sigma_max)

        gauss = np.exp(-0.5 * ((f_axis - f) / sigma_f) ** 2)
        gauss *= np.abs(f)  # USER'S FIX 4
        gauss = gauss.astype(np.complex64)

        st[:, i, :] = np.fft.ifft(fft_signal * gauss, axis=1)

    return np.abs(st).astype(np.float32), freqs

def stockwell_transform_v2(signal, sfreq, fmin=1.0, fmax=50.0, n_freqs=75):
    """Version 2: NO scaling (REVERTED FIX 4)"""
    n_samples = signal.shape[1]
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_freqs)
    f_axis = np.fft.fftfreq(n_samples, d=1.0/sfreq)
    fft_signal = np.fft.fft(signal, axis=1).astype(np.complex64)
    st = np.zeros((signal.shape[0], n_freqs, n_samples), dtype=np.complex64)

    for i, f in enumerate(freqs):
        sigma_f = 1.0 / (2.0 * np.pi * np.abs(f) + 1e-10)
        sigma_max = n_samples / (4.0 * sfreq)
        sigma_f = np.clip(sigma_f, 1e-10, sigma_max)

        gauss = np.exp(-0.5 * ((f_axis - f) / sigma_f) ** 2)
        gauss = gauss.astype(np.complex64)

        st[:, i, :] = np.fft.ifft(fft_signal * gauss, axis=1)

    return np.abs(st).astype(np.float32), freqs

def stockwell_transform_v3(signal, sfreq, fmin=1.0, fmax=50.0, n_freqs=75):
    """Version 3: WITH normalization (BEFORE FIX 5)"""
    n_samples = signal.shape[1]
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_freqs)
    f_axis = np.fft.fftfreq(n_samples, d=1.0/sfreq)
    fft_signal = np.fft.fft(signal, axis=1).astype(np.complex64)
    st = np.zeros((signal.shape[0], n_freqs, n_samples), dtype=np.complex64)

    for i, f in enumerate(freqs):
        sigma_f = 1.0 / (2.0 * np.pi * np.abs(f) + 1e-10)
        sigma_max = n_samples / (4.0 * sfreq)
        sigma_f = np.clip(sigma_f, 1e-10, sigma_max)

        gauss = np.exp(-0.5 * ((f_axis - f) / sigma_f) ** 2)
        gauss_norm = gauss / (np.sqrt(2.0 * np.pi) * sigma_f)
        gauss_norm = gauss_norm.astype(np.complex64)

        st[:, i, :] = np.fft.ifft(fft_signal * gauss_norm, axis=1)

    return np.abs(st).astype(np.float32), freqs

print("="*80)
print("RIGOROUS S-TRANSFORM VALIDATION")
print("="*80)

# Test parameters
sfreq = 1000.0
n_samples = 2000
t = np.arange(n_samples) / sfreq

# TEST 1: Frequency Detection Accuracy
print("\nTEST 1: FREQUENCY DETECTION ACCURACY")
print("-"*80)

test_freqs = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
errors_v1 = []
errors_v2 = []
errors_v3 = []

for target_freq in test_freqs:
    signal = np.sin(2 * np.pi * target_freq * t).reshape(1, -1).astype(np.float32)

    # Version 1
    st1, freqs1 = stockwell_transform_v1(signal, sfreq, fmin=1.0, fmax=45.0, n_freqs=75)
    energy1 = (st1[0] ** 2).mean(axis=1)
    detected1 = freqs1[np.argmax(energy1)]
    err1 = abs(detected1 - target_freq)
    errors_v1.append(err1)

    # Version 2
    st2, freqs2 = stockwell_transform_v2(signal, sfreq, fmin=1.0, fmax=45.0, n_freqs=75)
    energy2 = (st2[0] ** 2).mean(axis=1)
    detected2 = freqs2[np.argmax(energy2)]
    err2 = abs(detected2 - target_freq)
    errors_v2.append(err2)

    # Version 3
    st3, freqs3 = stockwell_transform_v3(signal, sfreq, fmin=1.0, fmax=45.0, n_freqs=75)
    energy3 = (st3[0] ** 2).mean(axis=1)
    detected3 = freqs3[np.argmax(energy3)]
    err3 = abs(detected3 - target_freq)
    errors_v3.append(err3)

print(f"{'Target':>7} | {'V1 (|f| scale)':>20} | {'V2 (no scale)':>20} | {'V3 (normalized)':>20}")
print(f"{'(Hz)':>7} | {'Detected':>9} {'Error':>9} | {'Detected':>9} {'Error':>9} | {'Detected':>9} {'Error':>9}")
print("-"*80)

for i, target_freq in enumerate(test_freqs):
    signal = np.sin(2 * np.pi * target_freq * t).reshape(1, -1).astype(np.float32)

    st1, freqs1 = stockwell_transform_v1(signal, sfreq, fmin=1.0, fmax=45.0, n_freqs=75)
    detected1 = freqs1[np.argmax((st1[0] ** 2).mean(axis=1))]

    st2, freqs2 = stockwell_transform_v2(signal, sfreq, fmin=1.0, fmax=45.0, n_freqs=75)
    detected2 = freqs2[np.argmax((st2[0] ** 2).mean(axis=1))]

    st3, freqs3 = stockwell_transform_v3(signal, sfreq, fmin=1.0, fmax=45.0, n_freqs=75)
    detected3 = freqs3[np.argmax((st3[0] ** 2).mean(axis=1))]

    print(f"{target_freq:7.1f} | {detected1:9.2f} {errors_v1[i]:9.2f} | "
          f"{detected2:9.2f} {errors_v2[i]:9.2f} | {detected3:9.2f} {errors_v3[i]:9.2f}")

print(f"\nMean Error: | {np.mean(errors_v1):18.2f} | {np.mean(errors_v2):18.2f} | {np.mean(errors_v3):18.2f}")

# TEST 2: Energy Conservation
print("\n\nTEST 2: ENERGY CONSERVATION")
print("-"*80)

target_freq = 15.0
signal = np.sin(2 * np.pi * target_freq * t).reshape(1, -1).astype(np.float32)

st1, _ = stockwell_transform_v1(signal, sfreq)
st2, _ = stockwell_transform_v2(signal, sfreq)
st3, _ = stockwell_transform_v3(signal, sfreq)

energy_orig = np.mean(signal ** 2)
energy_v1 = np.mean(st1 ** 2)
energy_v2 = np.mean(st2 ** 2)
energy_v3 = np.mean(st3 ** 2)

print(f"Original signal energy: {energy_orig:.6f}")
print(f"V1 (|f| scale) energy:  {energy_v1:.6f} (ratio: {energy_v1/energy_orig:.2f})")
print(f"V2 (no scale) energy:   {energy_v2:.6f} (ratio: {energy_v2/energy_orig:.2f})")
print(f"V3 (normalized) energy: {energy_v3:.6f} (ratio: {energy_v3/energy_orig:.2f})")

# TEST 3: Amplitude Accuracy
print("\n\nTEST 3: AMPLITUDE ACCURACY FOR SINE WAVE")
print("-"*80)

target_amp = 1.0
target_freq = 15.0
signal = target_amp * np.sin(2 * np.pi * target_freq * t).reshape(1, -1).astype(np.float32)

st1, freqs1 = stockwell_transform_v1(signal, sfreq)
st2, freqs2 = stockwell_transform_v2(signal, sfreq)
st3, freqs3 = stockwell_transform_v3(signal, sfreq)

# Get amplitude at target frequency
idx1 = np.argmin(np.abs(freqs1 - target_freq))
amp_v1 = st1[0, idx1, :].mean()

idx2 = np.argmin(np.abs(freqs2 - target_freq))
amp_v2 = st2[0, idx2, :].mean()

idx3 = np.argmin(np.abs(freqs3 - target_freq))
amp_v3 = st3[0, idx3, :].mean()

expected_amp = target_amp / np.sqrt(2)  # RMS of sine wave

print(f"Original amplitude: {target_amp:.6f}")
print(f"Expected ST amplitude (RMS): {expected_amp:.6f}")
print(f"V1 (|f| scale):  {amp_v1:.6f} (error: {abs(amp_v1 - expected_amp):.6f})")
print(f"V2 (no scale):   {amp_v2:.6f} (error: {abs(amp_v2 - expected_amp):.6f})")
print(f"V3 (normalized): {amp_v3:.6f} (error: {abs(amp_v3 - expected_amp):.6f})")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

# Determine winner
freq_winner = "V2" if np.mean(errors_v2) < min(np.mean(errors_v1), np.mean(errors_v3)) else ("V1" if np.mean(errors_v1) < np.mean(errors_v3) else "V3")
energy_winner = "V2" if abs(energy_v2/energy_orig - 1.0) < min(abs(energy_v1/energy_orig - 1.0), abs(energy_v3/energy_orig - 1.0)) else ("V1" if abs(energy_v1/energy_orig - 1.0) < abs(energy_v3/energy_orig - 1.0) else "V3")
amp_winner = "V2" if abs(amp_v2 - expected_amp) < min(abs(amp_v1 - expected_amp), abs(amp_v3 - expected_amp)) else ("V1" if abs(amp_v1 - expected_amp) < abs(amp_v3 - expected_amp) else "V3")

print(f"\nFrequency Detection: {freq_winner} wins (lowest mean error)")
print(f"Energy Conservation: {energy_winner} wins (closest to original)")
print(f"Amplitude Accuracy:  {amp_winner} wins (closest to expected)")

if freq_winner == "V2" and energy_winner == "V2" and amp_winner == "V2":
    print("\n✓ V2 (NO SCALING) IS CORRECT")
    print("  FIX 4 (gauss *= |f|) is WRONG and should be REVERTED")
elif freq_winner == "V1" and energy_winner == "V1" and amp_winner == "V1":
    print("\n✓ V1 (WITH |f| SCALING) IS CORRECT")
    print("  FIX 4 (gauss *= |f|) is CORRECT")
else:
    print("\n⚠ RESULTS ARE MIXED - need further investigation")
