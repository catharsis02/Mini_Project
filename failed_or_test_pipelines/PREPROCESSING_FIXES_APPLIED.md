# Mathematical Corrections Applied to Preprocessing Pipeline
## Date: 2025-03-25

---

## ✅ CRITICAL FIXES APPLIED

### 1. **Differential Entropy: Power vs Amplitude (CRITICAL)**

**Issue:** DE was computed on S-transform **amplitude** instead of **power**.

**Location:** `stockwell_de()` function, line ~220-248

**Mathematical Error:**
```python
# WRONG (before):
band_amp = st_amp[:, idx, :].mean(axis=1)  # Using amplitude
var = np.var(band_amp, axis=1) + 1e-10
de = 0.5 * np.log(2 * np.pi * np.e * var)
```

**Problem:**
- S-transform returns complex values: S(t, f)
- `np.abs(S)` gives amplitude |S(t, f)|
- DE formula requires variance of **power**, not amplitude
- Power = amplitude² = |S(t, f)|²
- Variance of amplitude ≠ Variance of power

**Correct Formula:**
```python
# CORRECT (after):
st_power = st_amp ** 2  # Convert amplitude to power
band_power = st_power[:, idx, :].mean(axis=1)  # Use power
var = np.var(band_power, axis=1, ddof=1) + 1e-10
de = 0.5 * np.log(2 * np.pi * np.e * var)
```

**Impact:** This was causing systematic underestimation of variance, leading to incorrect DE values. All 310 DE features were affected.

**Additional Improvement:** Changed to `ddof=1` for unbiased variance estimator.

---

### 2. **Notch Filter: Numerical Stability**

**Issue:** Notch filter used transfer function (b, a) form, which is numerically unstable.

**Location:** `notch_filter()` function, line ~126-129

**Before:**
```python
nyq = 0.5 * sfreq
b, a = iirnotch(50.0 / nyq, 30.0)
return filtfilt(b, a, data, axis=1)
```

**After:**
```python
sos = iirnotch(50.0, 30.0, fs=sfreq, output='sos')
return sosfiltfilt(sos, data, axis=1)
```

**Impact:** Improved numerical stability, especially important for high Q-factor (30) notch filters.

---

### 3. **Code Cleanup: Removed Duplicate Lines**

**Issue:** Lines 323-327 were duplicated (copy-paste error).

**Location:** `process_window()` function

**Impact:** No functional change, but cleaner code.

---

### 4. **Added Anti-Aliasing Warning**

**Issue:** Raw data is saved at 1000 Hz, but ML pipeline decimates to 500 Hz without anti-aliasing filter.

**Location:** Added note in `process_window()` docstring

**Note Added:**
```
NOTE: Raw data saved at 1000 Hz. If downstream pipeline applies decimation,
it MUST use proper anti-aliasing filter before downsampling to prevent aliasing
artifacts (e.g., scipy.signal.decimate or manual lowpass + resample).
```

**Impact:** Documentation for downstream users. Actual fix should be in ML pipeline (`ml_pipeline_v10.py` line 647).

---

## ✅ ALREADY CORRECT (verified)

### 1. **S-transform Gaussian Kernel**
- ✅ Correctly uses: `σ_f = 1/(2πf)`
- ✅ Gaussian centered at frequency f, not DC
- ✅ Numerically stable with epsilon guards

### 2. **RASM Asymmetry Formula**
- ✅ Uses log-domain difference: `DE_right - DE_left`
- ✅ Mathematically correct for log-space features

### 3. **Bandpass Filter**
- ✅ Uses SOS form for stability
- ✅ Applied once per window (correct)

### 4. **Z-scoring Order**
- ✅ DE computed BEFORE z-scoring
- ✅ Z-scoring only applied for CSP output
- ✅ Correct separation of concerns

### 5. **Window-First Design**
- ✅ No full-trial storage
- ✅ Constant memory usage
- ✅ Parallel processing per window

### 6. **Output Format**
- ✅ 310 DE features (5 bands × 62 channels)
- ✅ 50 RASM features (10 pairs × 5 bands)
- ✅ Total: 360 features (preserved)

---

## 📊 VALIDATION CHECKLIST

- [x] DE uses power (amplitude²), not amplitude
- [x] Filters use numerically stable SOS form
- [x] No normalization before DE computation
- [x] S-transform mathematically correct
- [x] RASM uses log-domain difference
- [x] Output feature dimensions unchanged (360 features)
- [x] Window-first design preserved
- [x] Memory constraints respected

---

## 🔬 EXPECTED IMPACT

### Quantitative Changes:
1. **DE values will be different** (correct now) - all 310 features affected
2. **DE will have larger dynamic range** (power has higher variance than amplitude)
3. **RASM will have better discriminative power** (derived from corrected DE)

### Why This Matters:
- **Before:** DE was computed on amplitude → variance of amplitude signal
- **After:** DE computed on power → variance of power signal
- **Power signal has higher variance** → DE values will be quantitatively larger
- **This is the CORRECT behavior** per information theory

### Model Performance:
- Expect **improved classification accuracy** because features now correctly represent signal entropy
- Better separation between classes due to correct variance estimation
- More stable cross-subject generalization

---

## 🎯 NEXT STEPS

### For Preprocessing:
- ✅ All mathematical fixes applied
- ✅ No reprocessing needed unless old data exists

### For ML Pipeline:
- ⚠️ **Add anti-aliasing before decimation** (line 647 in `ml_pipeline_v10.py`)
- Current: `X_raw = X_raw[:, :, ::DECIMATE_FACTOR]` (naive downsampling)
- Should be: Apply lowpass filter before decimation

**Recommended fix for ML pipeline:**
```python
from scipy.signal import decimate

# Instead of:
# X_raw = X_raw[:, :, ::DECIMATE_FACTOR]

# Use proper decimation with anti-aliasing:
X_raw_decimated = []
for trial in X_raw:
    trial_decimated = decimate(trial, DECIMATE_FACTOR, axis=1, ftype='fir')
    X_raw_decimated.append(trial_decimated)
X_raw = np.array(X_raw_decimated, dtype=np.float32)
```

---

## 📝 FORMULA REFERENCE

### Differential Entropy (Corrected):
```
Given: S-transform S(t, f) (complex)
Power: P(t, f) = |S(t, f)|²
Band power: P_band(t) = mean_f(P(t, f)) for f in [f_low, f_high]
Variance: σ² = Var_t(P_band(t))
DE: h = 0.5 × log(2πe × σ²)
```

### RASM (Already Correct):
```
RASM = DE_right - DE_left
     = 0.5 × log(σ²_R) - 0.5 × log(σ²_L)
     = 0.5 × log(σ²_R / σ²_L)
```

This is the standard Frontal Asymmetry Index (FAI) formulation.

---

## ✅ SUMMARY

All critical mathematical errors have been corrected. The preprocessing pipeline now:
- Computes DE on power (correct)
- Uses numerically stable filters
- Preserves all design constraints
- Maintains output format compatibility

**Status: READY FOR REPROCESSING**
