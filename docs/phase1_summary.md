# Phase 1: Adaptive Step Size Selection - Summary & Future Plans

## Overview

**Goal:** Use machine learning to intelligently select FxLMS step size (μ) based on noise characteristics to improve noise reduction performance across different driving scenarios.

**Hypothesis:** Different noise types (idle engine, highway road noise, etc.) require different step sizes for optimal performance. A trained classifier can detect the scenario and select the best μ.

---

## Phase 1A: Single-Channel Configuration (Initial Attempt)

### Configuration
- **Setup:** 1 speaker, 1 reference mic, 1 error mic (SISO)
- **Scenarios:** IDLE, CITY, HIGHWAY, ACCELERATION
- **Noise positions:** Single position [0.3, 0.92, 0.4] for ALL scenarios
- **Step size range tested:** 0.001, 0.003, 0.005, 0.007, 0.010, 0.015, 0.020
- **Regularization:** 1e-6
- **Training samples:** 600 (150 per scenario)

### Model Architecture
**Binary Classifier (IDLE vs Non-IDLE):**
```
Input: 12 features (4 time-domain + 8 spectral)
Architecture: 12 → 32 → 16 → 2 (softmax)
Dropout: 0.2
Output: IDLE → μ=0.015, Non-IDLE → μ=0.005
```

### Results
- **Training accuracy:** 100%
- **Validation accuracy:** 98%
- **Mean improvement:** +0.37 dB ✅
- **IDLE improvement:** +1.47 dB ✅
- **Win rate:** 25% ✅

**Success:** Single-channel Phase 1 met all success criteria.

### Key Findings
- IDLE scenarios benefit significantly from higher μ=0.015 (faster convergence)
- Other scenarios prefer conservative μ=0.005 (better steady-state NR)
- Binary classification (IDLE vs Non-IDLE) sufficient for meaningful improvement

---

## Phase 1B: Multi-Channel Configuration (Current)

### Why Retrain?
User requirement: **Default ML configuration must be 4 speakers + 4 reference mics** to match playground setup exactly. Single-channel model doesn't generalize to multi-channel acoustics.

### Configuration Changes
- **Setup:** 4 speakers (quad stereo), 4 reference mics (averaged), 1 error mic (MIMO)
- **Scenarios:** Same 4 scenarios
- **Noise positions:** **Scenario-specific** (matching reality):
  - IDLE: [0.15, 0.92, 0.5] (Engine/Firewall)
  - CITY: [0.5, 0.92, 0.5] (Combined/Dashboard)
  - HIGHWAY: [2.0, 0.92, 0.12] (Road/Floor)
  - ACCELERATION: [0.15, 0.92, 0.5] (Engine/Firewall)
- **Step size range:** Same as Phase 1A
- **Regularization:** **1e-4** (increased for stability with 4 speakers)
- **Training samples:** 600 (150 per scenario)

### Speaker Positions
```python
FOUR_SPEAKERS = {
    'door_L': [2.0, 0.1, 0.4],
    'door_R': [2.0, 1.75, 0.4],
    'dash_L': [0.8, 0.25, 0.9],
    'dash_R': [0.8, 1.60, 0.9],
}
```

### Reference Mic Positions
```python
FOUR_REF_MICS = {
    'firewall': [0.3, 0.92, 0.5],    # Engine noise detection
    'floor': [2.0, 0.55, 0.15],      # Road/tire noise
    'a_pillar': [0.5, 0.15, 1.0],    # Wind noise
    'dashboard': [0.9, 0.92, 0.8],   # General
}
```

### Signal Processing
**Reference signal:** Average of 4 microphone signals
```python
x(n) = mean([x_firewall(n), x_floor(n), x_a_pillar(n), x_dashboard(n)])
```

**Features:** 12 features extracted from averaged signal (same as Phase 1A)

**Anti-noise:** Single FxLMS controller drives all 4 speakers with identical signal y(n)

### Model Architecture
**Binary Classifier (IDLE vs Non-IDLE):**
```
Input: 12 features from averaged 4-mic signal
Architecture: 12 → 32 → 16 → 2 (softmax)
Dropout: 0.2
Output: IDLE → μ=0.007, Non-IDLE → μ=0.005
```

**Note:** μ values changed from Phase 1A (0.015 → 0.007 for IDLE)

### Training Results
- **Training accuracy:** 100%
- **Training samples:** 600 (150 per scenario)
- **Duration:** ~2 hours data collection
- **Model size:** 8 KB

### Evaluation Results

| Scenario | Baseline NR | Adaptive NR | Improvement | Selected μ | Accuracy |
|----------|-------------|-------------|-------------|------------|----------|
| IDLE | 4.31 ± 0.38 dB | 4.28 ± 0.27 dB | **-0.03 dB** | 0.007 | 100% |
| CITY | 6.64 ± 0.35 dB | 6.71 ± 0.41 dB | **+0.07 dB** | 0.005 | 100% |
| HIGHWAY | 2.91 ± 0.54 dB | 2.93 ± 0.51 dB | **+0.02 dB** | 0.005 | 100% |
| ACCELERATION | 22.16 ± 1.75 dB | 22.34 ± 1.96 dB | **+0.18 dB** | 0.005 | 100% |

**Overall Metrics:**
- **Mean improvement:** +0.061 dB (target: +0.30 dB) ❌ **FAILED**
- **Worst case:** -0.469 dB (target: -0.10 dB) ❌ **FAILED**
- **Win rate:** 62.5% (target: 25%) ✅ **PASSED**

### Success Criteria Assessment

| Criterion | Target | Achieved | Status |
|-----------|--------|----------|--------|
| Mean improvement | ≥ +0.30 dB | +0.061 dB | ❌ FAIL |
| Worst case | ≥ -0.10 dB | -0.469 dB | ❌ FAIL |
| Win rate | ≥ 25% | 62.5% | ✅ PASS |

**Conclusion:** Multi-channel Phase 1 does NOT meet success criteria.

---

## Key Findings: Single-Channel vs Multi-Channel

### Optimal Step Sizes Changed Dramatically

| Scenario | Single-Channel Best μ | Multi-Channel Best μ | Change |
|----------|----------------------|---------------------|---------|
| IDLE | 0.015 | 0.005 | -67% |
| CITY | 0.005 | 0.015 | +200% |
| HIGHWAY | 0.005 | 0.005 | No change |
| ACCELERATION | 0.005 | 0.005 | No change |

### Why Multi-Channel Changes Optimal Step Sizes

**Acoustic Summing:**
- 4 speakers sum their outputs at error mic: `y_total = y_speaker1 + y_speaker2 + y_speaker3 + y_speaker4`
- Since all speakers emit identical signal y(n), effective gain is 4× at error mic
- This amplifies the effect of step size μ on weight updates

**Impact:**
- Aggressive step sizes (μ=0.015) that worked for IDLE in single-channel now cause instability
- Even IDLE now prefers conservative μ=0.005 for stability
- Only CITY benefits from higher μ=0.015 (+0.61 dB improvement potential)

**Result:** Limited opportunity for step size optimization in multi-channel configuration.

---

## Current Status

### What We Have
✅ Multi-channel training data (600 samples, scenario-specific positions)
✅ Trained binary model (100% classification accuracy)
✅ Evaluation framework for multi-channel setup
✅ Understanding of why multi-channel limits adaptive step size benefits

### What We Learned
1. **Multi-channel fundamentally changes optimal step sizes** due to acoustic summing
2. **Static classification at t=0 provides minimal benefit** (+0.061 dB average)
3. **Perfect classification would only achieve +0.16 dB** (not enough to meet +0.30 dB target)
4. **62.5% win rate** suggests adaptive selection helps more often than it hurts

### Open Question
Can we do better with **dynamic adaptation** during runtime instead of static classification?

---

## Future Plans

### Option 1: Skip Phase 1 (Move to Phase 2 & 3)
**Rationale:** Static step size selection provides minimal benefit in multi-channel. Focus on filter length and max order optimization instead.

**Next steps:**
- Phase 2: Filter length optimization (64, 128, 256, 512)
- Phase 3: Max order optimization (1, 2, 3, 5 reflections)

---

### Option 2: Implement Dynamic Adaptive Step Size (RECOMMENDED)

#### Strategy: Sliding Window Re-Classification

**Concept:** Instead of classifying once at t=0, continuously re-classify noise type every 0.5 seconds during simulation.

**How it works:**
```
t=0.0s:   Extract features from [0.0-1.0s] → Classify → Set μ
t=0.5s:   Extract features from [0.5-1.5s] → Re-classify → Update μ
t=1.0s:   Extract features from [1.0-2.0s] → Re-classify → Update μ
...
```

**Implementation:**
1. Maintain sliding window buffer (1 second = 16000 samples)
2. Every 0.5 seconds:
   - Extract 12 features from last 1 second
   - Run classifier to predict current scenario
   - Update FxLMS step size if scenario changed
3. FxLMS uses current μ for all weight updates

**Expected benefits:**
- **Dynamic scenario adaptation:** Responds to transitions (idle→acceleration, city→highway)
- **Better "Dynamic Ride" performance:** Tracks changing noise characteristics
- **Faster reconvergence:** Can boost μ temporarily during transitions
- **Natural extension of Phase 1:** Reuses trained model and feature extraction

**Computational cost:**
- Feature extraction: ~1ms every 0.5s
- Classification: <0.1ms every 0.5s
- **Negligible overhead** (0.2% of runtime)

**Training:**
- **No retraining needed!** Use existing Phase 1B binary model
- Already trained on 600 samples across 4 scenarios
- Features proven to distinguish scenarios

**Evaluation approach:**
Test on dynamic scenarios:
1. **Smooth transitions:** idle→city→highway→acceleration (2s each)
2. **Sudden transitions:** idle→acceleration (instant RPM jump)
3. **Noisy transitions:** highway→city with overlapping noise

Compare:
- **Fixed μ=0.005 (baseline)**
- **Static classification (Phase 1B current)**
- **Dynamic sliding window (new strategy)**

Metrics:
- Overall NR across entire dynamic scenario
- Transition detection accuracy
- Reconvergence time after transitions

---

### Option 3: Alternative ML Strategies

#### Strategy A: Transition Detection
Train binary classifier to detect "stable" vs "transitioning" states:
- Stable → μ=0.005 (fine-tuning)
- Transitioning → μ=0.015 (fast adaptation for 2 seconds)

**Challenge:** Need to generate labeled transition data

#### Strategy B: Continuous Step Size Regression
Train regression model to predict optimal μ continuously:
- Input: 12 features + error magnitude + weight gradient
- Output: Continuous μ value (0.003 to 0.020)

**Challenge:** Expensive training (test many μ values per sample)

#### Strategy C: Dual-Filter Approach
Run two parallel FxLMS filters:
- Filter A: μ=0.005 (conservative)
- Filter B: μ=0.015 (aggressive)
- Select output with lower error

**Challenge:** 2× computational cost

---

## Recommendation

**Implement Option 2: Sliding Window Re-Classification**

**Why:**
1. ✅ Addresses main limitation (static classification)
2. ✅ No retraining needed (reuse Phase 1B model)
3. ✅ Minimal computational overhead
4. ✅ Natural extension of existing work
5. ✅ Interpretable (can visualize scenario tracking)
6. ✅ Likely to improve dynamic scenario performance significantly

**If successful:**
- Proceed to Phase 2 (filter length) with dynamic adaptive step size enabled
- Evaluate combined benefit of adaptive μ + optimized filter length

**If unsuccessful:**
- Skip Phase 1 entirely, use fixed μ=0.005
- Proceed to Phase 2 and Phase 3 optimizations

---

## Files and Locations

### Training Data
```
output/data/phase1/step_size_training_data.json
```
- 600 samples (150 per scenario)
- 12 features + 5 step sizes + optimal μ label
- Scenario-specific noise positions
- Multi-channel configuration (4 speakers + 4 ref mics)

### Models
```
output/models/phase1/step_selector_binary.pt
```
- Binary classifier (IDLE vs Non-IDLE)
- Input: 12 features
- Output: μ=0.007 (IDLE) or μ=0.005 (Non-IDLE)

### Evaluation Results
```
output/data/phase1/evaluation_results_binary.json
```
- Per-scenario NR (baseline and adaptive)
- Overall metrics (mean improvement, worst case, win rate)
- Classification accuracy per scenario

### Code
```
src/ml/phase1_step_size/
├── feature_extractor.py              # Extract 12 features from audio
├── step_size_selector_binary.py      # Binary classifier model
└── adaptive_fxlms_binary.py          # Integration with FxLMS

simulations_pyroom/phase1_step_size/
├── collect_training_data.py          # Generate training data
├── train_step_selector_binary.py     # Train binary model
└── evaluate_step_selector_binary.py  # Evaluate on test set
```

---

## Next Steps

1. **Implement sliding window re-classification**
   - Modify `src/ml/phase1_step_size/adaptive_fxlms_binary.py`
   - Add sliding window buffer and periodic re-classification
   - Test on dynamic scenarios

2. **Evaluate dynamic adaptation**
   - Create dynamic test scenarios (transitions)
   - Compare fixed μ vs static vs dynamic classification
   - Measure improvement in reconvergence and overall NR

3. **Decision point:**
   - If dynamic improves performance → integrate into Phase 2/3
   - If dynamic doesn't help → skip Phase 1, proceed to Phase 2/3

4. **Phase 2: Filter Length Optimization**
   - Test L ∈ {64, 128, 256, 512}
   - Multi-channel configuration
   - With or without dynamic step size (TBD)

5. **Phase 3: Max Order Optimization**
   - Test max_order ∈ {1, 2, 3, 5}
   - Use optimized filter length from Phase 2
   - Multi-channel configuration

---

## Success Criteria for Dynamic Adaptation

| Metric | Target | Notes |
|--------|--------|-------|
| Mean improvement (static scenarios) | ≥ +0.30 dB | Same as Phase 1B |
| Dynamic scenario NR | ≥ baseline + 0.5 dB | Better tracking of transitions |
| Transition detection latency | ≤ 1.0 second | Fast response to changes |
| False positive rate | ≤ 10% | Stable during steady-state |
| Computational overhead | ≤ 1% | Real-time feasible |

---

## Lessons Learned

1. **Multi-channel acoustics are fundamentally different** from single-channel
   - Can't simply apply single-channel findings to multi-channel systems
   - Need to retrain and re-evaluate for target configuration

2. **Static classification has limited benefit** when noise is stationary
   - All scenarios prefer similar step sizes in multi-channel
   - Opportunity is in dynamic adaptation, not static selection

3. **Regularization matters** for stability with multiple speakers
   - 1e-6 → 1e-4 prevented divergence with 4 speakers
   - Critical parameter often overlooked

4. **Scenario-specific noise positions** matter for realism
   - Engine noise at firewall, road noise at floor, etc.
   - Training data should match physical reality

5. **Feature extraction from averaged signals** works well
   - 4 mics → averaged signal → 12 features
   - Maintains interpretability and model architecture

6. **Perfect classification accuracy doesn't guarantee performance**
   - 100% accuracy but only +0.061 dB improvement
   - Limitation is physics (optimal step sizes are similar), not ML

---

## References

- Binary model architecture: `src/ml/phase1_step_size/step_size_selector_binary.py`
- Feature definitions: `src/ml/phase1_step_size/feature_extractor.py`
- Evaluation methodology: `simulations_pyroom/phase1_step_size/evaluate_step_selector_binary.py`
- Multi-channel configuration: `playground/presets.py` (FOUR_SPEAKERS, FOUR_REF_MICS)
