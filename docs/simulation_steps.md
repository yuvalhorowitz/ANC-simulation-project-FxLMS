# ANC Simulation Steps - Code Explanation

## Overview

The simulation is structured as 7 progressive steps, each building on the previous one to demonstrate Active Noise Cancellation concepts from basic room acoustics to a complete car interior ANC system.

---

## Step 1: Room Acoustics Basics
**File:** `simulations_pyroom/step1_room_acoustics.py`

### Concept
Introduces the **Room Impulse Response (RIR)** - the fundamental building block showing how sound travels from source to microphone, including direct sound and wall reflections.

### How It Works
1. Creates a 5m × 4m × 3m room using pyroomacoustics
2. Places a sound source and microphone 3 meters apart
3. Computes RIR using Image Source Method (max_order=3)
4. Tests different absorption coefficients (5%, 20%, 50%, 90%)
5. Convolves test signals with RIRs to simulate room acoustics

### Key Formula
```
Delay = Distance / Speed of Sound
     = 3m / 343 m/s ≈ 8.7ms ≈ 140 samples at 16kHz
```

### Outputs
- `pyroom_step1_rir.png` - RIR showing direct sound + reflections
- `pyroom_step1_absorption.png` - Effect of absorption on reverb
- Audio files: original, dry (anechoic), wet (reverberant)

---

## Step 2: Microphone Placement
**File:** `simulations_pyroom/step2_microphones.py`

### Concept
Introduces **multiple microphones** and the critical concept of **delay measurement** between reference mic (upstream) and error mic (downstream).

### How It Works
1. Places 5 microphones at increasing distances (0.5m to 4.5m)
2. Measures first arrival time at each mic using threshold detection
3. Sets up ANC configuration: reference mic closer to noise, error mic at listener
4. Calculates **delay difference** - the TIME BUDGET for ANC processing

### Key Insight
```
Reference mic receives noise FIRST
Error mic receives noise LATER
Delay difference = Processing time available for ANC
Example: 8.88ms = 142 samples at 16kHz
```

### Outputs
- `pyroom_step2_distances.png` - RIRs at different distances
- `pyroom_step2_anc_mics.png` - Reference vs error mic timing
- Audio files at each microphone position

---

## Step 3: Superposition & Destructive Interference
**File:** `simulations_pyroom/step3_superposition.py`

### Concept
Demonstrates the **core ANC principle**: when noise and anti-noise meet with equal amplitude but opposite phase, they cancel.

### How It Works
1. Sets up room with noise source AND control speaker
2. Computes transfer functions from each source to error mic
3. Calculates exact amplitude ratio and phase shift needed
4. Generates anti-noise: inverted, scaled, phase-shifted
5. Measures cancellation: combined = noise + anti-noise

### Key Formula
```python
# Transfer function at frequency f
H(f) = Σ[rir[n] * exp(-j*2π*f*n/fs)]

# Anti-noise calculation
amplitude_ratio = gain_noise / gain_speaker
phase_shift = phase_noise - phase_speaker
anti_noise = -amplitude_ratio * sin(2πft + phase_shift)
```

### Key Finding
- Perfect timing → 30+ dB cancellation
- Each sample of timing error degrades performance
- Reflections make perfect cancellation harder

### Outputs
- `pyroom_step3_superposition.png` - Noise + anti-noise = silence
- `pyroom_step3_phase_error.png` - Cancellation vs timing error
- Audio: noise, anti-noise, cancelled comparison

---

## Step 4: Ideal ANC (Known Paths)
**File:** `simulations_pyroom/step4_anc_ideal.py`

### Concept
Shows **theoretical maximum** ANC performance when acoustic paths are perfectly known - an impossible but useful baseline.

### How It Works
1. Computes exact RIRs for primary path (noise→error) and secondary path (speaker→error)
2. Generates multi-frequency noise (100Hz, or 50+80+120Hz, or harmonics)
3. Calculates ideal anti-noise in frequency domain
4. Measures residual error after perfect cancellation

### Configurations Tested
| Config | Room | Noise | Result |
|--------|------|-------|--------|
| A | Small office | 100 Hz HVAC | ~60 dB reduction |
| B | Living room | 50+80+120 Hz traffic | ~50 dB reduction |
| C | Industrial | 30-240 Hz harmonics | ~40 dB reduction |

### Outputs
- Time-domain before/after plots
- Frequency spectrum comparison
- Audio comparison files

---

## Step 5: ANC with Latency Problem
**File:** `simulations_pyroom/step5_anc_latency.py`

### Concept
Reveals the **critical flaw** in ideal approach: processing delays cause phase misalignment, turning cancellation into amplification.

### How It Works
1. Analyzes "time budget" = (delay to error mic) - (delay to reference mic) - (secondary path delay)
2. Tests multiple processing latencies: 0, 0.5, 1.0, 2.0, 5.0 ms
3. Shows that naive inversion fails beyond time budget
4. Demonstrates need for adaptive algorithms

### The Problem
```
If processing_delay > time_budget:
    Anti-noise arrives OUT OF PHASE
    Result: AMPLIFICATION instead of cancellation
```

### Key Finding
- 0ms delay → cancellation works
- Beyond time budget → noise INCREASES
- This motivates FxLMS algorithm in Step 6

### Outputs
- Bar chart: reduction vs latency (green=good, red=amplification)
- Path timing analysis

---

## Step 6: FxLMS Algorithm
**File:** `simulations_pyroom/step6_fxlms.py`

### Concept
Implements **FxNLMS (Filtered-X Normalized LMS)** - an adaptive algorithm that LEARNS the optimal anti-noise without knowing paths beforehand.

### How It Works
```python
# For each sample n:
x[n] = reference_path.filter(noise[n])     # Reference signal
d[n] = primary_path.filter(noise[n])       # Desired (noise at error)
y[n] = fxlms.generate_antinoise(x[n])      # Adaptive filter output
y_error[n] = secondary_path.filter(y[n])   # Anti-noise at error mic
e[n] = d[n] + y_error[n]                   # Error signal
fxlms.update_weights(e[n])                 # Adapt filter
```

### FxNLMS Parameters
| Parameter | Value | Purpose |
|-----------|-------|---------|
| Filter length | 256 taps | Capture RIR characteristics |
| Step size (μ) | 0.001-0.02 | Convergence speed vs stability |
| Regularization | 1e-4 | Prevent division by zero |
| Secondary path est. | 5% error | Realistic mismatch |

### Configurations Tested
| Config | Absorption | Max Order | Difficulty |
|--------|------------|-----------|------------|
| A | 20% | 15 | Hard (reverberant) |
| B | 40% | 10 | Medium |
| C | 60% | 5 | Easy (damped) |

### Outputs (4 plots per config)
1. Time domain: before/after waveforms
2. MSE convergence curve (log scale)
3. Frequency spectrum with 20-300 Hz target band
4. Learned filter coefficients

---

## Step 7: Car Interior ANC
**File:** `simulations_pyroom/step7_car_interior.py`

### Concept
Complete **real-world car ANC system** with multiple noise sources (engine, road, wind) and different driving scenarios.

### How It Works
1. Creates realistic car cabin with per-surface materials
2. Generates scenario-specific noise using `NoiseMixer`:
   - Engine: broadband 50-500 Hz, peak ~200 Hz
   - Road: 20-150 Hz rumble
   - Wind: 50-300 Hz with peaks
3. Runs FxNLMS sample-by-sample
4. Tracks noise reduction over time in 0.5s windows

### Car Configurations
| Car | Dimensions | Characteristics |
|-----|------------|-----------------|
| Compact | 3.5×1.8×1.5m | Tight coupling, fast convergence |
| Sedan | 4.8×1.85×1.5m | Balanced acoustics |
| SUV | 4.7×1.9×1.8m | Larger volume, slower convergence |

### Driving Scenarios
- **Highway**: 2800 RPM, 120 km/h, balanced engine/road/wind
- **City**: 2000 RPM, 50 km/h, engine-dominant
- **Acceleration**: 4500 RPM, engine-heavy
- **Idle**: 800 RPM, mostly engine

### FxNLMS Parameters
| Parameter | Value |
|-----------|-------|
| Filter length | 128-256 taps |
| Step size | 0.003-0.005 |
| Regularization | 1e-4 |

### Outputs (6 plots per scenario)
1. Reference signal waveform
2. Before/after ANC comparison
3. MSE convergence
4. Frequency spectrum (20-300 Hz band)
5. Filter coefficients
6. Noise reduction over time (shows learning)

### Typical Results
- Highway (Sedan): ~11 dB reduction
- City (Sedan): ~9 dB reduction
- Acceleration (SUV): ~22 dB reduction

---

## Progressive Learning Path

```
Step 1: "Sound bounces in rooms" (RIR basics)
   ↓
Step 2: "Sound arrives at different times at different places" (delays)
   ↓
Step 3: "Opposite signals cancel" (superposition principle)
   ↓
Step 4: "Perfect cancellation is possible with perfect knowledge" (ideal)
   ↓
Step 5: "But delays break it" (latency problem)
   ↓
Step 6: "Adaptive algorithms solve this" (FxLMS)
   ↓
Step 7: "Applied to real car noise" (practical system)
```

---

## Key Files Used

| Component | File |
|-----------|------|
| FxLMS/FxNLMS algorithm | `src/core/fxlms.py` |
| Acoustic path generator | `src/acoustic/path_generator.py` |
| Noise mixer (car noise) | `src/noise/noise_mixer.py` |
| Step configurations | `simulations_pyroom/configurations.py` |
| Placement optimization | `src/placement/microphone_config.py` |

---

## Step 8: Microphone Placement Optimization
**File:** `simulations_pyroom/step8_placement_optimization.py`

### Concept
Finds **optimal reference and error microphone locations** when using existing car stereo speakers for anti-noise generation. This answers the practical question: given a car's existing speakers, where should we place the microphones?

### How It Works
1. Defines all possible speaker positions (car stereo locations)
2. Defines candidate reference mic positions (noise detection)
3. Defines candidate error mic positions (listener locations)
4. Tests all combinations across driving scenarios
5. Analyzes results to find optimal placements

### Speaker Positions Tested
| Position | Location | Rationale |
|----------|----------|-----------|
| Door Left | [2.0, 0.1, 0.4] | Common in all cars |
| Door Right | [2.0, 1.75, 0.4] | Passenger side |
| Dashboard Left | [0.8, 0.25, 0.9] | A-pillar tweeter |
| Dashboard Center | [0.8, 0.92, 0.85] | Center stack |
| Headrest Driver | [3.2, 0.55, 1.0] | Premium systems |
| Rear Left | [4.0, 0.40, 0.9] | Rear deck |

### Reference Microphone Candidates
| Position | Location | Best For |
|----------|----------|----------|
| Firewall | [0.3, 0.92, 0.5] | Engine noise |
| Dashboard | [0.9, 0.92, 0.8] | General pickup |
| A-Pillar | [0.7, 0.15, 1.0] | Wind/road noise |
| Under Seat | [2.5, 0.55, 0.15] | Road noise |

### Error Microphone Candidates
| Position | Location | Target |
|----------|----------|--------|
| Driver Headrest | [3.2, 0.55, 1.0] | Primary listener |
| Driver Ear Left | [3.2, 0.40, 1.0] | Precise targeting |
| Driver Ear Right | [3.2, 0.70, 1.0] | Binaural |
| Passenger Headrest | [3.2, 1.30, 1.0] | Passenger comfort |

### Key Finding
```
Optimal configuration:
- Speaker: Headrest (closest to ear)
- Reference mic: Firewall (earliest noise detection)
- Error mic: Driver headrest (at listener)

This achieves 15-20 dB reduction vs 5-10 dB for door speakers
```

### Outputs
- `output/data/pyroom_step8_sweep_results.csv` - Full results table
- `output/data/pyroom_step8_analysis.json` - Rankings and recommendations
- `output/plots/pyroom_step8_heatmap_*.png` - Performance heatmaps
- `output/plots/pyroom_step8_speaker_comparison.png` - Speaker rankings
- `output/plots/pyroom_step8_top_configurations.png` - Top 10 configurations

---

## Progressive Learning Path (Updated)

```
Step 1: "Sound bounces in rooms" (RIR basics)
   ↓
Step 2: "Sound arrives at different times at different places" (delays)
   ↓
Step 3: "Opposite signals cancel" (superposition principle)
   ↓
Step 4: "Perfect cancellation is possible with perfect knowledge" (ideal)
   ↓
Step 5: "But delays break it" (latency problem)
   ↓
Step 6: "Adaptive algorithms solve this" (FxLMS)
   ↓
Step 7: "Applied to real car noise" (practical system)
   ↓
Step 8: "Where to place components for best results" (optimization)
```

---

## Key Files Used

| Component | File |
|-----------|------|
| FxLMS/FxNLMS algorithm | `src/core/fxlms.py` |
| Acoustic path generator | `src/acoustic/path_generator.py` |
| Noise mixer (car noise) | `src/noise/noise_mixer.py` |
| Step configurations | `simulations_pyroom/configurations.py` |
| Placement module | `src/placement/` |
| Placement config | `src/placement/microphone_config.py` |
| Placement simulation | `simulations_pyroom/step8_placement_optimization.py` |
| Playground presets | `playground/presets.py` |
