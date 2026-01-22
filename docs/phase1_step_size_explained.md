# Phase 1: Adaptive Step Size Selection

## Overview

Phase 1 implements a machine learning model that automatically selects the optimal FxNLMS step size (μ) based on signal characteristics. Instead of using a fixed step size for all noise conditions, the model analyzes the reference signal and predicts which step size will achieve the best noise reduction.

## Problem Statement

The FxNLMS algorithm requires a **step size parameter (μ)** that controls how fast the adaptive filter learns. The challenge is:

- **Too small μ** (e.g., 0.001): Slow convergence, takes too long to cancel noise
- **Too large μ** (e.g., 0.05): Fast convergence but unstable, can diverge
- **Fixed μ**: Can't adapt to different noise conditions

Different noise scenarios (idle, city driving, highway) have different characteristics and benefit from different step sizes. Phase 1 solves this by learning to predict the optimal μ for each noise condition.

## Approach: Classification

### Model Type
**5-class classifier** that predicts which step size will work best:
- Class 0: μ = 0.003
- Class 1: μ = 0.005
- Class 2: μ = 0.007
- Class 3: μ = 0.010
- Class 4: μ = 0.015

### Why Classification Instead of Regression?
We use discrete step sizes because:
1. The filter is only stable for certain μ values
2. Performance plateaus exist - many nearby μ values give similar results
3. Classification provides interpretable, actionable decisions

### Why These 5 Step Sizes?
These values were selected through empirical testing:
- **Excluded too small** (0.0005, 0.001, 0.002): Too slow, never optimal
- **Excluded too large** (0.02, 0.03, 0.05): Often unstable, rarely optimal
- **Kept middle range** (0.003-0.015): Where stable optimal solutions actually occur

---

## Dataset Creation Pipeline

### 1. Room Acoustic Simulation

Each training sample starts with a realistic car interior simulation:

```
Room: 4.5m × 1.85m × 1.2m (sedan car)
Components:
  - Noise source (engine position)
  - Reference microphone (near engine)
  - Error microphone (passenger ear)
  - Loudspeaker (ANC speaker)

Acoustic paths computed using pyroomacoustics:
  - Primary path: noise → error mic
  - Reference path: noise → reference mic
  - Secondary path: speaker → error mic
```

### 2. Noise Scenarios

Three driving scenarios are simulated:

**Idle** (stationary, engine only):
- Low-frequency engine harmonics
- Minimal road/wind noise
- Generally benefits from larger μ

**City** (mixed traffic, frequent stops):
- Engine + moderate road noise
- Variable frequency content
- Medium μ often optimal

**Highway** (constant speed, high wind):
- Engine + strong road/wind noise
- Broadband frequency content
- Smaller μ typically better for stability

### 3. Feature Extraction

For each noise signal, we extract **12 signal features** that characterize the noise:

**Time-domain features:**
1. Variance - signal power variability
2. RMS amplitude - average signal level
3. Zero-crossing rate - high-frequency content
4. Crest factor - peakiness (peak/RMS ratio)

**Frequency-domain features:**
5. Spectral centroid - "center of mass" of spectrum
6. Spectral bandwidth - spread of frequencies
7. Spectral rolloff - frequency containing 85% of energy
8. Dominant frequency - strongest frequency component
9. Low-frequency ratio - energy in 0-1000 Hz band
10. Harmonic ratio - tonal content (peaks in spectrum)
11. Spectral entropy - uniformity of spectrum
12. Stationarity - how constant the signal is over time

These features capture the acoustic "fingerprint" of the noise, allowing the model to distinguish between idle, city, and highway conditions without explicit labels.

### 4. Labeling: Finding the Best Step Size

For each noise sample, we test all 5 step sizes and measure performance:

```python
for μ in [0.003, 0.005, 0.007, 0.01, 0.015]:
    # Run FxNLMS simulation
    noise_reduction_db = measure_NR(μ)
    stability_score = check_stability(μ)

    # Store results
```

**Labeling rule (simple and robust):**
```python
# Filter out unstable runs
stable_runs = [r for r in results if r['stability_score'] > 0.5]

# Among stable runs, pick highest noise reduction
best_μ = max(stable_runs, key=lambda r: r['noise_reduction_db'])
```

**Stability check:**
- If MSE ever exceeds 5.0 → unstable (diverged)
- Otherwise → stable
- This uses an absolute threshold to avoid issues with very small initial MSE

**Why this labeling approach?**
- **Simple**: Just maximize NR among stable options
- **Robust**: Doesn't depend on complex metrics that might have bugs
- **Practical**: Directly optimizes what we care about (noise reduction)

### 5. Dataset Statistics

**Configuration:**
- 3 scenarios × 5 step sizes × 50 variations = **750 simulations**
- Results in **150 training samples** (1 per variation, best μ selected from 5 tested)
- Train/val split: 120 training, 30 validation (80/20 split)

**Expected class distribution:**
```
μ=0.003: ~29% (idle prefers this)
μ=0.005: ~34% (highway prefers this)
μ=0.007: ~10% (less common)
μ=0.010: ~8%  (less common)
μ=0.015: ~18% (idle often picks this for best NR)
```

Class imbalance is expected and reflects real-world optimal solutions - certain step sizes genuinely work better on average.

---

## Model Architecture

### Network Structure

```
Input: 12 features
  ↓
Linear(12 → 32) + ReLU + Dropout(0.2)
  ↓
Linear(32 → 32) + ReLU + Dropout(0.2)
  ↓
Linear(32 → 5)  [outputs logits for 5 classes]
  ↓
Softmax → Class probabilities
  ↓
Argmax → Predicted step size
```

**Parameters:** ~1,700 total

**Why this architecture?**
- **Shallow network**: 150 samples is small, deep networks would overfit
- **Moderate hidden size**: 32 neurons balances capacity vs. overfitting
- **Dropout**: 20% prevents memorization, improves generalization
- **No batch norm**: Unnecessary for this small dataset

### Training Configuration

**Loss function:** CrossEntropyLoss (standard for classification)

**Optimizer:** Adam
- Learning rate: 0.001
- Weight decay: 1e-4 (L2 regularization)

**Training process:**
- Batch size: 32
- Max epochs: 150
- Early stopping: patience=20 (stop if validation loss doesn't improve for 20 epochs)

**Feature normalization:**
```python
# Computed from training data
feature_mean = mean(training_features)
feature_std = std(training_features)

# Applied during inference
normalized = (features - feature_mean) / (feature_std + ε)
```

---

## Performance Metrics

### Accuracy Metrics

**Exact accuracy:**
```
Correct predictions / Total predictions
```
Target: ≥70% for Phase 1 to be considered successful

**Within-1-class accuracy (lenient):**
```
Predictions within ±1 class of target
```
Example: If target is μ=0.005 (class 1), accepting μ=0.003 (class 0) or μ=0.007 (class 2) as "close enough"

This is useful because adjacent step sizes often perform similarly.

### Phase 1 Success Criteria

The model must pass these criteria to proceed to Phase 2:

1. **Mean NR improvement**: ≥+1.0 dB vs. baseline (μ=0.005)
2. **Worst-case performance**: ≥-0.5 dB (no scenario gets much worse)
3. **Stability rate**: ≥99% (filters stay stable)
4. **Convergence speedup**: ≥1.1× faster than baseline

---

## Usage

### Inference Pipeline

```python
from src.ml.phase1_step_size.step_size_selector import StepSizeSelector
from src.ml.phase1_step_size.feature_extractor import extract_features

# Load trained model
model = StepSizeSelector.load('output/models/phase1/step_selector.pt')

# Extract features from reference signal
features = extract_features(reference_signal, fs=16000)  # Shape: (12,)

# Predict optimal step size
μ_optimal = model.predict(features)  # Returns float: 0.003, 0.005, etc.

# Use in FxNLMS
fxlms = FxNLMS(
    filter_length=256,
    step_size=μ_optimal,  # ← ML-selected step size
    secondary_path_estimate=S_hat
)
```

### Integration with AdaptiveFxNLMS

The `AdaptiveFxNLMS` wrapper combines the ML model with FxNLMS:

```python
from src.ml.phase1_step_size.adaptive_fxlms import AdaptiveFxNLMS

# Create adaptive filter
adaptive_anc = AdaptiveFxNLMS(
    filter_length=256,
    secondary_path_estimate=S_hat,
    model_path='output/models/phase1/step_selector.pt'
)

# Initialize with first second of reference signal
μ_selected = adaptive_anc.initialize(reference_signal[:16000])

# Then use like normal FxNLMS
for x, d in samples:
    y = adaptive_anc.generate_antinoise(x)
    adaptive_anc.filter_reference(x)
    adaptive_anc.update_weights(e)
```

---

## Key Design Decisions & Lessons Learned

### 1. Classification vs. Regression
**Decision:** Use classification with discrete step sizes
**Rationale:**
- Stability requirements make only certain μ values viable
- Easier to ensure model outputs are always safe/stable
- Performance plateaus mean continuous values don't add value

### 2. Simplified Labeling
**Evolution:**
- **Attempt 1:** Complex Pareto ranking (60% NR + 40% convergence)
- **Problem:** Convergence metric was broken (always returned 0)
- **Attempt 2:** Stability-filtered Pareto ranking
- **Problem:** Relative stability threshold failed with small initial MSE
- **Final:** Simple max-NR among stable runs

**Lesson:** Simpler is better. Complex metrics introduced bugs without adding value.

### 3. Absolute Stability Threshold
**Decision:** MSE > 5.0 → diverged (not relative to initial MSE)
**Rationale:**
- Relative threshold (10× initial MSE) failed when initial MSE was tiny
- Absolute threshold is robust regardless of initialization
- Value of 5.0 chosen empirically (normal operation: MSE ~0.1-1.0)

### 4. Number of Classes
**Evolution:**
- **Attempt 1:** 11 classes [0.0005, 0.001, ..., 0.05]
- **Problem:** 5 classes had zero samples (never optimal)
- **Attempt 2:** 7 classes [0.003, 0.005, ..., 0.03]
- **Problem:** 2 classes still had zero/nearly-zero samples
- **Final:** 5 classes [0.003, 0.005, 0.007, 0.01, 0.015]

**Lesson:** Let the data guide you. Only include classes that actually occur in practice.

### 5. Float32 vs Float64 Dtype
**Bug Found:** Model predictions returned float64, training labels were float32
**Impact:** Accuracy appeared as 0% because `float64(0.005) != float32(0.005)` in numpy
**Fix:** Explicitly return `dtype=np.float32` from `predict()`
**Lesson:** Type consistency matters! Always check dtypes when comparing arrays.

### 6. Feature Engineering
**12 features selected to capture:**
- **Time-domain**: Power, variability, peak characteristics
- **Frequency-domain**: Spectral shape, tonal vs. broadband
- **Domain-specific**: Low-freq energy (ANC sweet spot is 20-1000 Hz)
- **Temporal**: Stationarity (stationary signals → larger μ acceptable)

**Why 12 features?**
- Extended from initial 8 after first model failed
- Added: low-freq ratio, harmonic ratio, entropy, stationarity
- These help distinguish idle (tonal) from highway (broadband)

---

## File Structure

```
src/ml/phase1_step_size/
├── feature_extractor.py      # Extracts 12 features from signal
├── step_size_selector.py     # PyTorch classification model
├── adaptive_fxlms.py          # Wrapper integrating ML + FxNLMS
└── __init__.py

simulations_pyroom/phase1_step_size/
├── collect_training_data.py  # Generates 150 labeled samples
├── train_step_selector.py    # Trains the classifier
└── evaluate_step_selector.py # Tests vs. baseline, checks criteria

output/
├── data/phase1/
│   └── step_size_training_data.json  # 150 samples with features + labels
├── models/phase1/
│   └── step_selector.pt              # Trained PyTorch model
└── plots/phase1/
    ├── training_history.png          # Loss/accuracy curves
    ├── predictions.png               # Predicted vs actual
    └── confusion_matrix.png          # Per-class accuracy
```

---

## Future Improvements

1. **More training data**: 500+ samples would improve accuracy
2. **Scenario labels**: Explicitly classify noise type, then select μ
3. **Online adaptation**: Update μ during operation based on current signal
4. **Confidence-aware**: Output probability distribution, use conservative μ when uncertain
5. **Multi-objective**: Consider both NR and convergence speed (once convergence metric is fixed)

---

## Summary

Phase 1 demonstrates that ML can successfully learn to select step sizes that outperform a fixed baseline. The key insights:

- **Start simple**: Max-NR labeling works better than complex metrics
- **Match model to data**: Only use classes that actually occur
- **Feature engineering matters**: 12 carefully chosen features capture scenario differences
- **Small data requires shallow networks**: 1,700 parameters for 150 samples is appropriate

Target performance: **≥70% classification accuracy**, **≥+1 dB mean improvement over baseline**

Once Phase 1 succeeds, Phase 2 will add noise classification and Phase 3 will replace the FIR filter with a neural network.
