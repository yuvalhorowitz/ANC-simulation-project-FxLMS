# Phase 1: Adaptive Step Size Selection - Learning Journey

**Author**: Development Team
**Date**: January 2025
**Status**: In Progress (Current: +0.48 dB, Target: +1.0 dB)

---

## Executive Summary

This document chronicles the complete trial-and-error process of implementing Phase 1 of the ML-enhanced FxLMS ANC system: learning to predict optimal step size (μ) from signal characteristics. Over multiple iterations, we went from a completely failed 5-class model to a working binary classifier achieving statistically significant improvement (+0.48 dB, p=0.0021).

**Key lessons learned:**
1. **Train/test distribution mismatch** was the critical failure point - features must match deployment environment
2. **Simpler is better** - binary classification (Low/High μ) outperformed 3-class and 5-class approaches
3. **Domain knowledge matters** - understanding signal stationarity explained why IDLE needs high μ
4. **Incremental optimization** - small gains (+0.02 dB from mapping tuning) accumulate

---

## Table of Contents

1. [Initial State: The Complete Failure](#1-initial-state-the-complete-failure)
2. [Attempt 1: Simplify to Binary Classification](#2-attempt-1-simplify-to-binary-classification)
3. [Discovery: The Train/Test Distribution Mismatch](#3-discovery-the-traintest-distribution-mismatch)
4. [The Fix: Regenerating All Training Data](#4-the-fix-regenerating-all-training-data)
5. [Attempt 2: Testing 3-Class Classification](#5-attempt-2-testing-3-class-classification)
6. [Optimization: Fine-Tuning Step Size Mapping](#6-optimization-fine-tuning-step-size-mapping)
7. [Current Status and Next Steps](#7-current-status-and-next-steps)
8. [Lessons Learned](#8-lessons-learned)

---

## 1. Initial State: The Complete Failure

### The Problem

We started with a 5-class step size selector that had collapsed completely:

**Training data:**
- 600 samples (200 per scenario: idle, city, highway)
- 5 step size classes: 0.003, 0.005, 0.007, 0.010, 0.015
- 12 features per sample (variance, RMS, spectral features, etc.)

**Training results:**
- Validation accuracy: **49.2%** (barely better than random)
- Loss plateaued immediately - model wasn't learning

**Deployment results (evaluation):**
```
IDLE:     NR = 4.89 dB (baseline = 4.89 dB) → +0.00 dB
CITY:     NR = 13.57 dB (baseline = 13.57 dB) → +0.00 dB
HIGHWAY:  NR = 4.36 dB (baseline = 4.36 dB) → +0.00 dB

Mean improvement: +0.00 dB
```

**Root cause hypothesis at the time:**
- Too many classes (5) for limited data (600 samples)
- Class imbalance (some step sizes rarely optimal)
- Model capacity issues

### Why This Failed

Looking back, we had **two fundamental problems**:
1. **Too much complexity** - 5 classes with only 120 samples per class
2. **Distribution mismatch** (discovered later) - training features didn't match deployment

---

## 2. Attempt 1: Simplify to Binary Classification

### The Hypothesis

"If the model can't learn 5 classes, test if it can learn ANY pattern at all by reducing to 2 classes."

### Implementation

**Binary classification design:**
- **Class 0 (Low μ)**: 0.003, 0.005, 0.007 → use μ=0.005
- **Class 1 (High μ)**: 0.010, 0.015 → use μ=0.012

**Rationale:**
- CITY/HIGHWAY (non-stationary, high-frequency) → need low μ for stability
- IDLE (stationary, low-frequency) → can use high μ for fast convergence

**Training process:**
```python
# Binary mapping function
def step_size_to_binary(mu: float) -> int:
    if mu <= 0.007:
        return 0  # Low μ
    else:
        return 1  # High μ

# Class weights to handle imbalance
class_counts = [438, 162]  # 73% low, 27% high
total = sum(class_counts)
class_weights = torch.tensor([total / (len(class_counts) * c) for c in class_counts])

# Loss with class weighting
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### Results

**Training:**
- Validation accuracy: **89.2%** ✅ (huge improvement!)
- Loss converged smoothly
- Clear learning happening

**Deployment (first evaluation):**
```
IDLE:     μ=0.005 (predicted Low instead of High!)
CITY:     μ=0.005 (correct)
HIGHWAY:  μ=0.005 (correct)

ALL scenarios predicted Low μ = 0.005
Mean improvement: +0.06 dB
```

### The Mystery

**Model had 89.2% validation accuracy but predicted the SAME class for ALL scenarios in deployment.**

This made no sense. The model clearly learned something during training, but completely failed in the real world.

---

## 3. Discovery: The Train/Test Distribution Mismatch

### Debugging Process

We tested the model with different inputs to isolate the issue:

**Test 1: Raw noise features (same as training)**
```python
# Generate raw noise
noise_signal = mixer.generate_scenario(1.0, 'idle')
features = extract_features(noise_signal, fs=16000)
predicted_class = model.predict(features)
# Result: Varied predictions ✅
```

**Test 2: Reference-filtered features (same as deployment)**
```python
# Filter through reference path (matching deployment)
reference_path = FIRPath(paths['reference'])
ref_signal = np.zeros(fs)
for i in range(len(noise_signal)):
    ref_signal[i] = reference_path.filter_sample(noise_signal[i])

features = extract_features(ref_signal, fs=16000)
predicted_class = model.predict(features)
# Result: ALL predictions = Class 0 ❌
```

### The Root Cause

**CRITICAL DISCOVERY**: Training and deployment used completely different feature distributions!

**Training data collection (`collect_training_data.py` line 264 - WRONG):**
```python
# Extract features from RAW noise signal
features = extract_features(noise_signal, FS)
```

**Deployment (`evaluate_binary.py` lines 164-169 - CORRECT):**
```python
# Extract features from REFERENCE-FILTERED signal
ref_signal = np.zeros(FS)
temp_ref_path = FIRPath(paths['reference'])
for i in range(min(FS, n_samples)):
    ref_signal[i] = temp_ref_path.filter_sample(noise_signal[i])

features = extract_features(ref_signal, FS)
```

### Why This Matters

The **reference path is an acoustic filter** (room impulse response from noise source to reference microphone). It completely changes the signal characteristics:

| Feature | Raw Noise | Reference-Filtered | Change |
|---------|-----------|-------------------|--------|
| Spectral Centroid | 150 Hz | 200 Hz | +33% |
| RMS Amplitude | 1.0 | 0.7 | -30% |
| Spectral Rolloff | 280 Hz | 320 Hz | +14% |
| Variance | 0.45 | 0.25 | -44% |

**The model learned to classify raw noise**, but in deployment it received **filtered noise** - a completely different distribution. The model had never seen this distribution, so it collapsed to always predicting the majority class (Low μ).

### The Analogy

Imagine training a dog to recognize cats vs. dogs using **color photos**, then testing it with **black-and-white photos**. The model would fail even though it learned the task correctly on its training distribution.

---

## 4. The Fix: Regenerating All Training Data

### The Solution

**Fix the training data to match deployment**: Extract features from reference-filtered signals, not raw noise.

**Code change in `collect_training_data.py` (lines 263-272):**

```python
# BEFORE (WRONG):
features = extract_features(noise_signal, FS)

# AFTER (CORRECT):
# Filter first second through reference path
reference_path = FIRPath(paths['reference'])
ref_signal = np.zeros(FS)
for i in range(min(FS, len(noise_signal))):
    ref_signal[i] = reference_path.filter_sample(noise_signal[i])

# Extract features from reference-filtered signal (matching deployment)
features = extract_features(ref_signal, FS)
```

### Regeneration Process

1. Deleted old `step_size_training_data.json` (600 samples with wrong features)
2. Ran `collect_training_data.py` (1,040 simulations, 35 minutes)
3. Generated new dataset with 600 samples using **reference-filtered features**
4. Retrained binary classifier from scratch

### Results After Fix

**Training:**
- Validation accuracy: **95.8%** ✅ (up from 89.2%!)
- Better generalization with correct distribution

**Deployment:**
```
IDLE:     μ=0.0106 (High μ) → NR = 6.17 dB → +1.27 dB ✅
CITY:     μ=0.0050 (Low μ)  → NR = 13.65 dB → +0.08 dB ✅
HIGHWAY:  μ=0.0050 (Low μ)  → NR = 4.38 dB → +0.02 dB ✅

Mean improvement: +0.46 dB
Effect size (Cohen's d): 0.637
p-value: 0.0018 → Statistically significant ✅
Win rate: 73.3%
```

**Model now correctly predicts:**
- IDLE → High μ (fast convergence for stationary signal)
- CITY/HIGHWAY → Low μ (stability for non-stationary signals)

### Why IDLE Needs High μ and CITY/HIGHWAY Need Low μ

**Signal stationarity theory:**

| Scenario | Characteristics | Optimal μ | Reason |
|----------|----------------|-----------|--------|
| **IDLE** | Stationary, periodic (engine harmonics), narrow-band (20-150 Hz), high autocorrelation | **High μ (0.010-0.015)** | Signal is predictable → large steps converge faster without causing instability |
| **CITY** | Non-stationary, transient events (bumps, acceleration), broad-band (20-800 Hz), low autocorrelation | **Low μ (0.003-0.005)** | Signal changes rapidly → large steps would diverge, small steps track variations |
| **HIGHWAY** | Non-stationary, turbulent (wind noise, road rumble), broad-band (50-1000 Hz), medium autocorrelation | **Low μ (0.003-0.005)** | High-frequency content → large steps would overshoot, causing instability |

**Adaptive filter convergence equation:**
```
Optimal μ ∝ 1 / (λ_max * ||x||²)
```
Where:
- λ_max = largest eigenvalue of autocorrelation matrix (higher for stationary signals)
- ||x||² = signal power

For IDLE: signal is predictable → can use μ closer to theoretical maximum
For CITY/HIGHWAY: signal is unpredictable → must use conservative μ for stability

---

## 5. Attempt 2: Testing 3-Class Classification

### The Hypothesis

"Binary classifier works (+0.46 dB), but maybe we can do better with more granular control."

### Implementation

**3-class design:**
- **Class 0 (Low μ)**: 0.003, 0.005 → use μ=0.004
- **Class 1 (Medium μ)**: 0.007, 0.010 → use μ=0.0085
- **Class 2 (High μ)**: 0.015 → use μ=0.015

**Training process:**
```python
def step_size_to_3class(mu: float) -> int:
    if mu <= 0.005:
        return 0  # Low μ
    elif mu <= 0.010:
        return 1  # Medium μ
    else:
        return 2  # High μ

# Class distribution from training data
class_counts = [327, 89, 184]  # 54.5%, 14.8%, 30.7%
```

### Results

**Training:**
- Overall validation accuracy: **75.0%** (lower than binary 95.8%)
- Per-class accuracy:
  - Low μ: 82.9% ✅
  - Medium μ: **25.9%** ❌ (most confused with Low)
  - High μ: 86.7% ✅

**Deployment:**
```
IDLE:     μ=0.0126 → NR = 6.19 dB → +1.30 dB ✅
CITY:     μ=0.0040 → NR = 13.62 dB → +0.05 dB ✅
HIGHWAY:  μ=0.00445 → NR = 4.24 dB → -0.12 dB ❌

Mean improvement: +0.41 dB (WORSE than binary's +0.46 dB)
```

### Why 3-Class Failed

**The Medium class problem:**
- Medium μ (0.007-0.010) is a **transition zone** between Low and High
- No scenario clearly belongs to Medium - it's an ambiguous middle ground
- Model confused Medium with Low (25.9% accuracy)
- HIGHWAY sometimes got μ=0.0085 (Medium) instead of μ=0.004 (Low)
  - Medium μ is too aggressive for HIGHWAY's non-stationary signals
  - Caused -0.12 dB degradation (negative improvement!)

**Decision: Stick with binary classifier**
- Binary: clear Low vs High boundary → 95.8% accuracy
- 3-class: ambiguous Medium class → 75% accuracy, worse results

---

## 6. Optimization: Fine-Tuning Step Size Mapping

### The Goal

Binary classifier achieves +0.46 dB, but Phase 1 success criteria requires **≥+1.0 dB**.

**Gap: +0.54 dB needed**

### Strategy 1: Step Size Mapping Tuning

**Hypothesis:** The classifier correctly identifies Low vs High scenarios, but the actual μ values (0.005, 0.012) might not be optimal.

**Test 3 mapping options without retraining:**

| Option | Low μ | High μ | Rationale |
|--------|-------|--------|-----------|
| Current | 0.005 | 0.012 | Baseline from training data |
| **A** | 0.003 | 0.015 | Lower low, higher high (more aggressive separation) |
| **B** | 0.004 | 0.015 | Slightly lower low, higher high |
| **C** | 0.005 | 0.015 | Keep low same, increase high only |

### Results

**Option A: (0.003, 0.015)**
```
IDLE:     +1.21 dB
CITY:     -0.20 dB ❌
HIGHWAY:  -0.44 dB ❌

Mean: +0.19 dB (WORSE)
p-value: 0.2942 (not significant)
Win rate: 36.7%
```
**Failure reason:** μ=0.003 too low for CITY/HIGHWAY → sluggish convergence, poor tracking

---

**Option B: (0.004, 0.015)**
```
IDLE:     +1.28 dB
CITY:     +0.05 dB
HIGHWAY:  -0.15 dB ❌

Mean: +0.39 dB (WORSE)
p-value: 0.0153 (significant)
Win rate: 53.3%
```
**Failure reason:** μ=0.004 still too conservative for CITY/HIGHWAY

---

**Option C: (0.005, 0.015) ✅ WINNER**
```
IDLE:     +1.35 dB ✅ (best IDLE performance!)
CITY:     +0.08 dB ✅ (same as current)
HIGHWAY:  +0.02 dB ✅ (same as current)

Mean: +0.48 dB (BEST)
p-value: 0.0021 (statistically significant)
Win rate: 73.3%
```
**Success reason:**
- Keep Low μ = 0.005 (optimal for CITY/HIGHWAY)
- Increase High μ from 0.012 → 0.015 (better IDLE convergence)
- No degradation to CITY/HIGHWAY

### Optimization Summary

| Configuration | IDLE | CITY | HIGHWAY | **Mean** | Status |
|--------------|------|------|---------|----------|--------|
| Current (0.005, 0.012) | +1.27 | +0.08 | +0.02 | **+0.46** | Baseline |
| Option A (0.003, 0.015) | +1.21 | -0.20 | -0.44 | **+0.19** | ❌ Failed |
| Option B (0.004, 0.015) | +1.28 | +0.05 | -0.15 | **+0.39** | ❌ Failed |
| **Option C (0.005, 0.015)** | +1.35 | +0.08 | +0.02 | **+0.48** | ✅ **Best** |

**Gain from optimization:** +0.02 dB (4% relative improvement)

### Key Insights

1. **Low μ = 0.005 is optimal** for CITY/HIGHWAY - lowering it degraded performance
2. **Increasing High μ helped IDLE** without affecting other scenarios
3. **Small gains matter** - every +0.02 dB brings us closer to the +1.0 dB target
4. **No retraining needed** - mapping changes are instantaneous to test

---

## 7. Current Status and Next Steps

### Phase 1 Success Criteria

| Criterion | Target | **Current** | Status |
|-----------|--------|-------------|--------|
| Mean NR Improvement | ≥1.0 dB | **+0.48 dB** | ❌ FAIL (48% of goal) |
| Worst-Case Improvement | ≥-0.5 dB | **+0.02 dB** | ✅ PASS |
| Stability Rate | ≥99% | **73.3%** | ❌ FAIL |
| Convergence Speedup | ≥1.1× | **0.00×** | ❌ FAIL |

**Overall Phase 1 Status:** 1/4 criteria passed

### What We've Achieved

✅ **Proof of concept successful:**
- ML-based step size selection works
- Statistically significant improvement (p=0.0021)
- Correct scenario differentiation (IDLE→High μ, CITY/HIGHWAY→Low μ)
- No negative scenarios (all ≥ -0.02 dB)

❌ **But improvement insufficient:**
- Need **+0.52 dB more** to reach +1.0 dB target
- Stability and convergence metrics not addressed yet

### Remaining Optimization Strategies

**Strategy 3: Feature Engineering** (60 min, expected +0.1-0.2 dB)
- Add 8 new features: autocorrelation, spectral flux, kurtosis, harmonic-to-noise ratio, etc.
- Re-collect 600 samples with 20-feature vectors
- Retrain binary classifier
- Better scenario discrimination → better class separation

**Strategy 5: Hierarchical Model** (2 hours, expected +0.2-0.4 dB)
- Two-stage: (1) Classify scenario (idle/city/highway), (2) Use scenario-specific μ
- More interpretable and targeted
- Potential for scenario-specific optimization

**Strategy 4: More Training Data** (90 min, marginal gain)
- Double from 600 to 1,200 samples
- Better generalization but likely saturated already

---

## 8. Lessons Learned

### Technical Lessons

1. **Distribution matching is critical**
   - Train/test features must come from the SAME distribution
   - In ANC systems, account for all acoustic paths in the signal chain
   - **Cost of mistake:** 2 weeks of debugging the train/test mismatch

2. **Simpler models often work better**
   - Binary (95.8% accuracy, +0.48 dB) beat 3-class (75% accuracy, +0.41 dB) and 5-class (49% accuracy, +0.00 dB)
   - Ambiguous classes (Medium μ) hurt more than they help
   - **Occam's Razor applies to ML**

3. **Domain knowledge accelerates debugging**
   - Understanding signal stationarity explained why IDLE needs high μ
   - Knowing FxLMS convergence theory guided step size selection
   - **Physics > black-box ML**

4. **Incremental testing saves time**
   - Testing mapping options (15 min each) before retraining (60 min) was efficient
   - Quick experiments revealed Option A/B failed, Option C succeeded
   - **Fail fast, iterate quickly**

5. **Class imbalance matters**
   - 73% low μ vs 27% high μ in training data
   - Used class weighting in loss function to compensate
   - Without weighting, model biased toward Low μ (majority class)

### Process Lessons

1. **Debug methodically**
   - When model had 89.2% train accuracy but failed deployment, we isolated:
     - Test 1: Raw features → predictions varied ✓
     - Test 2: Filtered features → all predictions same ✗
   - Pinpointed exact issue: feature distribution mismatch

2. **Validate intermediate results**
   - After fixing distribution mismatch, immediately tested deployment
   - Confirmed model now predicted correctly before proceeding
   - **Don't stack changes - validate incrementally**

3. **Document everything**
   - This file exists because we tracked every attempt, failure, and insight
   - Future work can avoid repeating mistakes (e.g., 3-class Medium confusion)
   - **Institutional memory > starting from scratch**

4. **Use statistical rigor**
   - p-values and effect sizes validated improvements were real
   - Win rate showed consistency across scenarios
   - **Anecdotes ≠ evidence**

### What Worked

- ✅ Simplifying from 5-class → binary classification
- ✅ Fixing train/test distribution mismatch
- ✅ Using class weights to handle imbalance
- ✅ Testing multiple mapping options quickly
- ✅ Grounding design in domain knowledge (stationarity theory)

### What Didn't Work

- ❌ 5-class classification (too many classes for data available)
- ❌ 3-class classification (Medium class caused confusion)
- ❌ Using raw noise features (didn't match deployment)
- ❌ Lowering Low μ below 0.005 (degraded CITY/HIGHWAY)

---

## Timeline Summary

| Date | Event | Result |
|------|-------|--------|
| Initial | 5-class model trained | 49.2% accuracy, +0.00 dB |
| Iteration 1 | Simplify to binary classification | 89.2% accuracy, +0.06 dB (all scenarios predicted same class) |
| Discovery | Identified train/test distribution mismatch | Root cause found |
| Iteration 2 | Regenerated data with reference-filtered features | 95.8% accuracy, **+0.46 dB**, p=0.0018 ✅ |
| Iteration 3 | Tested 3-class model | 75% accuracy, +0.41 dB (worse than binary) |
| Iteration 4 | Optimized step size mapping (Option C) | **+0.48 dB**, p=0.0021 ✅ |
| **Current** | Binary classifier with (0.005, 0.015) mapping | **+0.48 dB (48% of +1.0 dB goal)** |

**Total development time:** ~4 weeks (including debugging, data collection, training, evaluation)

---

## Conclusion

Phase 1 has been a journey from complete failure (+0.00 dB) to partial success (+0.48 dB). The most critical lesson was the train/test distribution mismatch - a subtle bug that completely broke deployment despite good training metrics.

**Current state:**
- ✅ Proof of concept: ML step size selection works
- ✅ Statistically significant: p=0.0021, 73.3% win rate
- ✅ Theoretically sound: matches signal stationarity principles
- ❌ Insufficient improvement: +0.48 dB vs ≥1.0 dB target

**Path forward:**
- Strategy 3 (feature engineering) or Strategy 5 (hierarchical model) needed
- Estimated +0.2-0.4 dB additional gain possible
- Phase 1 completion feasible but not guaranteed

The learning process has been invaluable - these lessons will inform Phase 2 (noise classification) and Phase 3 (neural ANC).

---

**Document Status:** Living document - will be updated as Phase 1 optimization continues

**Next Update:** After completing Strategy 3 or Strategy 5
