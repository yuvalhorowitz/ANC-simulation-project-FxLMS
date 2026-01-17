# Machine Learning Enhancements for FxLMS ANC

A comprehensive guide to enhancing the FxLMS Active Noise Cancellation system with machine learning.

---

## Overview

This project implements a three-phase incremental approach to enhance the FxLMS algorithm with ML:

```
Phase 1: Adaptive Step Size    → Learn optimal μ from signal features
Phase 2: Noise Classification  → Classify noise type, select parameters
Phase 3: Neural ANC            → Replace FIR filter with neural network
```

Each phase builds on the previous one, and all ML components are **separate from the core FxLMS** - the original `src/core/fxlms.py` is never modified.

---

## Phase 1: Adaptive Step Size Selector

### The Problem

The standard FxNLMS uses a fixed step size (μ = 0.005). However:
- Different noise types converge best with different μ values
- Too small μ = slow convergence
- Too large μ = instability

### The Solution

Train an MLP to predict optimal μ from signal features:

```
Reference Signal x(n)
        │
        ▼
┌─────────────────┐
│ Feature         │ → [variance, rms, spectral_centroid, ...]
│ Extractor       │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Step Size       │ → μ ∈ [0.0005, 0.05]
│ Selector (MLP)  │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ FxNLMS with     │ → Anti-noise y(n)
│ adaptive μ      │
└─────────────────┘
```

### Features Used (8 total)

1. **Variance** - Signal power variability
2. **RMS Amplitude** - Average signal level
3. **Zero-Crossing Rate** - High-frequency content indicator
4. **Spectral Centroid** - "Center of mass" of spectrum
5. **Spectral Bandwidth** - Spread of frequencies
6. **Spectral Rolloff** - Frequency below which 85% of energy exists
7. **Dominant Frequency** - Strongest frequency component
8. **Crest Factor** - Peak-to-RMS ratio

### Usage

```python
from src.ml.phase1_step_size import AdaptiveFxNLMS

# Create adaptive filter
adaptive_anc = AdaptiveFxNLMS(
    filter_length=256,
    secondary_path_estimate=S_hat,
    model_path='output/models/phase1/step_selector.pt'
)

# Initialize with first second of reference signal
mu = adaptive_anc.initialize(reference_signal[:16000])
print(f"Selected step size: {mu}")

# Then use like normal FxNLMS
y = adaptive_anc.generate_antinoise(x)
adaptive_anc.filter_reference(x)
adaptive_anc.update_weights(e)
```

---

## Phase 2: Noise Type Classification

### The Problem

Different driving conditions produce different noise types:
- **Engine noise**: Low frequency, periodic
- **Road noise**: Broadband, random
- **Highway noise**: Mix of engine, road, and wind
- **Idle**: Low-level engine hum

Each type responds best to different FxLMS parameters.

### The Solution

Train a CNN to classify noise type from mel spectrograms, then look up optimal parameters:

```
Reference Signal x(n) [1 second window]
        │
        ▼
┌─────────────────┐
│ Mel Spectrogram │ → [n_mels=64, time=32]
│ Extractor       │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ CNN Classifier  │ → class ∈ {engine, road, highway, idle}
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Parameter       │ → (μ, filter_length)
│ Lookup Table    │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ FxNLMS          │ → Anti-noise y(n)
└─────────────────┘
```

### Parameter Lookup Table

| Noise Class | Step Size (μ) | Filter Length |
|-------------|---------------|---------------|
| Engine      | 0.003         | 256           |
| Road        | 0.008         | 192           |
| Highway     | 0.005         | 256           |
| Idle        | 0.002         | 128           |

### Usage

```python
from src.ml.phase2_classifier import ClassifiedFxNLMS

# Create classified filter
classified_anc = ClassifiedFxNLMS(
    secondary_path_estimate=S_hat,
    classifier_path='output/models/phase2/noise_classifier.pt'
)

# Initialize with mel spectrogram of first second
noise_class, params = classified_anc.initialize(mel_spectrogram)
print(f"Detected: {noise_class}, using μ={params['step_size']}")

# Then use like normal FxNLMS
y = classified_anc.generate_antinoise(x)
# ...
```

---

## Phase 3: Neural Anti-Noise Generator

### The Problem

FxLMS is limited to linear filtering. Complex acoustic environments may benefit from nonlinear processing.

### The Solution

Replace the FIR filter entirely with a neural network trained to generate anti-noise:

```
Reference buffer: [x(n), x(n-1), ..., x(n-255)]
        │
        ▼
┌─────────────────┐
│ Neural Network  │ → y(n) (anti-noise sample)
│ (1D CNN / LSTM) │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Secondary Path  │ → y'(n) (anti-noise at error mic)
│ S(z) [known]    │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Error: e = d+y' │ ← d(n) (noise at error mic)
└─────────────────┘
        │
        ▼
Loss = E[e²]  → Backprop through S(z) to update NN
```

### Key Insight

The neural network must account for the secondary path S(z). During training, we:
1. Forward pass through NN → y(n)
2. Convolve y with S(z) → y'(n)
3. Compute error e = d + y'
4. Backprop through the S(z) convolution

### Architecture Options

**1D CNN (Fast inference)**
```python
class NeuralANC_CNN(nn.Module):
    def __init__(self, buffer_len=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 1)
        )
```

**LSTM (Better for sequences)**
```python
class NeuralANC_LSTM(nn.Module):
    def __init__(self, buffer_len=256, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
```

---

## Comparison Methodology

### Baseline

The baseline is standard FxNLMS with:
- Fixed step size: μ = 0.005
- Fixed filter length: L = 256
- No noise-type awareness

### Quality Metrics

| Metric | Weight | Success Threshold |
|--------|--------|-------------------|
| Noise Reduction (dB) | 40% | ML ≥ Baseline |
| Improvement (dB) | 25% | > 0 dB average |
| Convergence Time | 15% | ML ≤ Baseline |
| Stability Score | 15% | ≥ 99% |
| Generalization | 5% | Within 2dB of training |

### Statistical Significance

For claiming "ML is better", we require:
- **Paired t-test**: p-value < 0.05
- **Effect size**: Cohen's d > 0.3
- **Win rate**: ML wins on ≥ 70% of scenarios

---

## Directory Structure

```
src/ml/
├── __init__.py
├── common/
│   ├── metrics.py         # noise_reduction_db, convergence_time
│   └── comparison.py      # is_significant_improvement
├── phase1_step_size/
│   ├── feature_extractor.py
│   ├── step_size_selector.py
│   └── adaptive_fxlms.py
├── phase2_classifier/
│   ├── spectrogram.py
│   ├── noise_classifier.py
│   ├── parameter_lookup.py
│   └── classified_fxlms.py
└── phase3_neural/
    ├── neural_anc.py
    ├── neural_anc_trainer.py
    └── neural_anc_wrapper.py

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
├── models/phase{1,2,3}/    # Trained model weights
├── data/phase{1,2,3}/      # Training data
└── plots/phase{1,2,3}/     # Comparison plots
```

---

## Dependencies

Add to requirements.txt:
```
torch>=2.0.0
torchaudio>=2.0.0
librosa>=0.10.0
scipy>=1.10.0
```

---

## Running the Pipeline

### Phase 1

```bash
# 1. Collect training data
python simulations_pyroom/phase1_step_size/collect_training_data.py

# 2. Train the step size selector
python simulations_pyroom/phase1_step_size/train_step_selector.py

# 3. Evaluate against baseline
python simulations_pyroom/phase1_step_size/evaluate_step_selector.py
```

### Phase 2

```bash
# 1. Collect labeled noise data
python simulations_pyroom/phase2_classifier/collect_noise_data.py

# 2. Train the noise classifier
python simulations_pyroom/phase2_classifier/train_classifier.py

# 3. Evaluate
python simulations_pyroom/phase2_classifier/evaluate_classifier.py
```

### Phase 3

```bash
# 1. Generate training pairs
python simulations_pyroom/phase3_neural/generate_training_pairs.py

# 2. Train the neural ANC
python simulations_pyroom/phase3_neural/train_neural_anc.py

# 3. Evaluate against FxLMS
python simulations_pyroom/phase3_neural/evaluate_neural_anc.py
```

---

## References

- Widrow, B., & Stearns, S. D. (1985). Adaptive Signal Processing
- Kuo, S. M., & Morgan, D. R. (1996). Active Noise Control Systems
- Zhang, J., & Bhattacharya, B. (2020). Deep ANC: A Deep Learning Approach to Active Noise Control
