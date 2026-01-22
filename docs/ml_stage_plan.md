# ML Stage Plan

## Overview

This document defines the ML enhancement strategy for the ANC system. The goal is to improve upon the FxLMS baseline through three incremental phases, each adding intelligence to the system.

---

## Current Baseline (FxLMS)

### Performance Summary

| Scenario | Noise Reduction | Notes |
|----------|-----------------|-------|
| Acceleration | **19.88 dB** | Excellent - engine-dominant |
| City | **8.87 dB** | Good - mixed noise |
| Highway | **5.98 dB** | Needs improvement - broadband road noise |
| Idle | **5.58 dB** | Needs improvement - low-frequency engine |

### Baseline Configuration
- Algorithm: FxNLMS
- Filter length: 256 taps
- Step size: μ = 0.005 (fixed)
- Configuration: 4 ref mics, 4 speakers

---

## ML Goals

### Primary Goals

| Goal | Target | Rationale |
|------|--------|-----------|
| **Improve weak scenarios** | Highway ≥ 8 dB, Idle ≥ 8 dB | Current baseline underperforms on these |
| **Maintain strong scenarios** | Acceleration ≥ 18 dB, City ≥ 8 dB | Don't regress on working cases |
| **Faster convergence** | ≤ 1 second to 90% of final reduction | Adaptive μ should speed up convergence |
| **Automatic adaptation** | No manual parameter tuning needed | Classify noise → select parameters |

### Secondary Goals

| Goal | Target | Rationale |
|------|--------|-----------|
| Memory footprint | Total ML models < 50 MB | Lightweight for potential deployment |
| Inference speed | < 1 ms per sample | Real-time capable |
| Stability | 100% stable runs (no divergence) | ML should never make things worse |

---

## 3-Phase ML Approach

### Phase Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    ML Enhancement Phases                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: Adaptive Step Size                                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ Reference   │───▶│ Feature     │───▶│ MLP: Select │         │
│  │ Signal      │    │ Extraction  │    │ optimal μ   │         │
│  └─────────────┘    └─────────────┘    └──────┬──────┘         │
│                                               │                 │
│  Phase 2: Noise Classification                ▼                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ Mel         │───▶│ CNN         │───▶│ Parameter   │         │
│  │ Spectrogram │    │ Classifier  │    │ Lookup      │         │
│  └─────────────┘    └─────────────┘    └──────┬──────┘         │
│                                               │                 │
│  Phase 3: Neural ANC                          ▼                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ Reference   │───▶│ Neural Net  │───▶│ Anti-Noise  │         │
│  │ Buffer      │    │ (CNN/LSTM)  │    │ Output      │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Adaptive Step Size Selector

### Purpose
Automatically select the optimal step size (μ) based on signal characteristics. Different noise types converge best with different μ values.

### Architecture

**Model:** MLP (Multi-Layer Perceptron)
- Input: 8 signal features
- Hidden: 64 → 32 neurons
- Output: μ ∈ [0.0005, 0.05]

**Features Extracted:**
1. Variance - Signal power variability
2. RMS amplitude - Average signal level
3. Zero-crossing rate - High-frequency content
4. Spectral centroid - Center of mass of spectrum
5. Spectral bandwidth - Frequency spread
6. Spectral rolloff - 85% energy frequency
7. Dominant frequency - Strongest component
8. Crest factor - Peak-to-RMS ratio

### Expected Improvement
- Faster convergence: 30-50% reduction in convergence time
- Better final performance on scenarios with suboptimal fixed μ
- No divergence (adaptive μ stays in stable range)

### Implementation Status

| Component | Location | Status |
|-----------|----------|--------|
| Feature extractor | `src/ml/phase1_step_size/feature_extractor.py` | ✓ Complete |
| Step size selector | `src/ml/phase1_step_size/step_size_selector.py` | ✓ Complete |
| Adaptive FxNLMS | `src/ml/phase1_step_size/adaptive_fxlms.py` | ✓ Complete |
| Training script | `simulations_pyroom/phase1_step_size/train_step_selector.py` | ✓ Complete |
| Evaluation script | `simulations_pyroom/phase1_step_size/evaluate_step_selector.py` | ✓ Complete |

---

## Phase 2: Noise Type Classification

### Purpose
Classify the noise type and select optimal FxLMS parameters for each type. Different scenarios benefit from different filter lengths and step sizes.

### Architecture

**Model:** CNN on mel spectrograms
- Input: Mel spectrogram [64 mels × 32 time frames]
- Conv layers: 32 → 64 → 128 channels
- Output: 4 classes {engine, road, highway, idle}

**Parameter Lookup:**

| Noise Class | Step Size (μ) | Filter Length | Rationale |
|-------------|---------------|---------------|-----------|
| Engine | 0.003 | 256 | Narrow-band, needs precision |
| Road | 0.008 | 192 | Broadband, faster adaptation |
| Highway | 0.005 | 256 | Mixed noise, balanced |
| Idle | 0.002 | 128 | Low-level, conservative |

### Expected Improvement
- Scenario-specific optimization without manual tuning
- Better handling of transitions between driving conditions
- Improved performance on difficult scenarios (road, idle)

### Implementation Status

| Component | Location | Status |
|-----------|----------|--------|
| Spectrogram extractor | `src/ml/phase2_classifier/spectrogram.py` | ✓ Complete |
| Noise classifier | `src/ml/phase2_classifier/noise_classifier.py` | ✓ Complete |
| Parameter lookup | `src/ml/phase2_classifier/parameter_lookup.py` | ✓ Complete |
| Classified FxNLMS | `src/ml/phase2_classifier/classified_fxlms.py` | ✓ Complete |

---

## Phase 3: Neural Anti-Noise Generator

### Purpose
Replace the linear FIR filter with a neural network capable of learning non-linear noise-to-anti-noise mappings.

### Architecture Options

**Option A: 1D CNN (Recommended for speed)**
```
Input: [batch, 256] reference buffer
Conv1d: 1 → 32 channels, k=7
Conv1d: 32 → 64 channels, k=5
Conv1d: 64 → 32 channels, k=3
AdaptiveAvgPool → Linear → Output: [batch, 1]
```

**Option B: LSTM (Better for sequences)**
```
Input: [batch, 256, 1] reference buffer
LSTM: hidden=64, layers=2
Linear: 64 → 1
Output: [batch, 1]
```

**Option C: MLP (Fastest inference)**
```
Input: [batch, 256] reference buffer
Linear: 256 → 128, ReLU
Linear: 128 → 128, ReLU
Linear: 128 → 1
Output: [batch, 1]
```

### Training Approach
1. Generate training pairs: reference buffer → ideal anti-noise
2. Include secondary path S(z) in forward pass
3. Loss: MSE between (noise + anti-noise) and zero
4. Backprop through S(z) convolution

### Expected Improvement
- Non-linear modeling for complex acoustics
- Better handling of reverberant environments
- Potential improvement on broadband noise (highway scenario)

### Implementation Status

| Component | Location | Status |
|-----------|----------|--------|
| Neural models | `src/ml/phase3_neural/neural_anc.py` | ✓ Complete |
| Trainer | `src/ml/phase3_neural/neural_anc_trainer.py` | ✓ Complete |
| Wrapper | `src/ml/phase3_neural/neural_anc_wrapper.py` | ✓ Complete |

---

## Success Metrics

### Noise Reduction Comparison

| Scenario | FxLMS Baseline | ML Target | Success Criterion |
|----------|----------------|-----------|-------------------|
| Acceleration | 19.88 dB | ≥ 18 dB | Maintain performance |
| City | 8.87 dB | ≥ 9 dB | Slight improvement |
| Highway | 5.98 dB | ≥ 8 dB | **+2 dB improvement** |
| Idle | 5.58 dB | ≥ 8 dB | **+2.5 dB improvement** |

### Other Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Convergence time | ≤ 1 second | Time to 90% of final reduction |
| Stability | 100% | No divergence across all tests |
| Memory | < 50 MB | Sum of all model sizes |
| Inference | < 1 ms/sample | Time per forward pass |

### Statistical Validation
- Run each configuration 10 times (account for randomness)
- Report mean ± std deviation
- ML wins if: mean improvement > 1 dB AND p-value < 0.05

---

## Implementation Priority

### Phase 1 (Current Focus)
1. ✓ Collect training data (μ → performance mapping)
2. ✓ Train step size selector
3. ✓ Evaluate against fixed μ baseline
4. → Integrate with playground for interactive testing

### Phase 2 (Next)
1. Collect labeled noise samples for each class
2. Train CNN classifier
3. Evaluate classification accuracy
4. Integrate classified FxNLMS into simulation

### Phase 3 (Final)
1. Generate training pairs (reference → ideal anti-noise)
2. Train CNN/LSTM/MLP models
3. Compare against FxNLMS baseline
4. Select best architecture for final system

---

## Directory Structure

```
src/ml/
├── common/
│   ├── metrics.py           # noise_reduction_db, convergence_time
│   └── comparison.py        # statistical comparison utilities
├── phase1_step_size/
│   ├── feature_extractor.py # 8-feature extraction
│   ├── step_size_selector.py # MLP model
│   └── adaptive_fxlms.py    # FxNLMS with adaptive μ
├── phase2_classifier/
│   ├── spectrogram.py       # Mel spectrogram extraction
│   ├── noise_classifier.py  # CNN model
│   ├── parameter_lookup.py  # Class → parameters
│   └── classified_fxlms.py  # FxNLMS with classification
└── phase3_neural/
    ├── neural_anc.py        # CNN/LSTM/MLP architectures
    ├── neural_anc_trainer.py # Training with S(z) backprop
    └── neural_anc_wrapper.py # Integration wrapper

simulations_pyroom/
├── phase1_step_size/
│   ├── collect_training_data.py
│   ├── train_step_selector.py
│   └── evaluate_step_selector.py
├── phase2_classifier/
│   ├── collect_noise_data.py
│   ├── train_classifier.py
│   └── evaluate_classifier.py
└── phase3_neural/
    ├── generate_training_pairs.py
    ├── train_neural_anc.py
    └── evaluate_neural_anc.py

output/
├── models/phase{1,2,3}/     # Trained model weights
├── data/phase{1,2,3}/       # Training data
└── plots/phase{1,2,3}/      # Comparison plots
```

---

## Conclusion

The 3-phase ML approach enhances FxLMS incrementally:

1. **Phase 1** - Faster convergence through adaptive step size
2. **Phase 2** - Scenario-specific optimization through classification
3. **Phase 3** - Non-linear modeling through neural networks

Primary focus: Improve Highway and Idle scenarios from ~6 dB to ≥8 dB while maintaining strong performance on Acceleration and City scenarios.
