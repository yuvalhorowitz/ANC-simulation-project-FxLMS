# Active Noise Cancellation (ANC) Simulation using FxLMS

A step-by-step simulation of Active Noise Cancellation for car interior, using the Filtered-x Least Mean Square (FxLMS) algorithm. Target frequency range: 20-300 Hz.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the simulations in order
python simulations/step1_wave_basics.py
python simulations/step2_virtual_mics.py
python simulations/step3_speaker_interaction.py
python simulations/step4_simple_anc_ideal.py
python simulations/step5_anc_with_latency.py
python simulations/step6_fxlms_simple.py
python simulations/step7_car_interior.py
```

## Project Overview

This project implements ANC simulation following an **incremental learning approach**:

### Phase 1: Understanding Wave Physics
- **Step 1**: 1D acoustic wave propagation and reflections
- **Step 2**: Virtual microphones and time delay measurement
- **Step 3**: Speaker interaction and destructive interference

### Phase 2: Simple ANC System
- **Step 4**: Ideal ANC (no latency) - demonstrates the goal
- **Step 5**: ANC with latency - shows why simple inversion fails
- **Step 6**: FxLMS algorithm - the adaptive solution

### Phase 3: Car Interior Application
- **Step 7**: Full car interior simulation with realistic noise

## The FxLMS Algorithm

The core equation from the paper:

```
w(n+1) = w(n) + μ * e(n) * f(n)
```

Where:
- `w(n)`: Adaptive filter weights
- `μ`: Step size (learning rate)
- `e(n)`: Error signal (residual noise)
- `f(n)`: Filtered reference signal (reference filtered through secondary path estimate)

## Simulation Steps

### Step 1: Wave Basics (`step1_wave_basics.py`)
Learn how acoustic waves propagate in a 1D space.

**What you'll see:**
- Pulse propagation at speed of sound (343 m/s)
- Wave reflections at boundaries
- Standing wave patterns

### Step 2: Virtual Microphones (`step2_virtual_mics.py`)
Understand how microphones measure pressure waves.

**Key concept:** The time delay between microphones = distance / speed of sound

### Step 3: Speaker Interaction (`step3_speaker_interaction.py`)
Demonstrate the **superposition principle** - the foundation of ANC.

**Key concept:** If speaker_wave = -noise_wave, they cancel out!

### Step 4: Ideal ANC (`step4_simple_anc_ideal.py`)
The simplest possible ANC system with perfect knowledge and zero latency.

**Result:** Near-perfect cancellation (demonstrates the goal)

### Step 5: ANC with Latency (`step5_anc_with_latency.py`)
What happens when we add realistic processing delay.

**Problem:** Naive inversion fails because anti-noise arrives LATE!

### Step 6: FxLMS Solution (`step6_fxlms_simple.py`)
The adaptive algorithm that learns to compensate for latency.

**Result:** 15-25 dB noise reduction despite secondary path delay

### Step 7: Car Interior (`step7_car_interior.py`)
Full simulation with realistic car noise (engine + road + wind).

**Scenarios:**
- Highway (120 km/h, 2800 RPM)
- City (50 km/h, 2000 RPM)
- Acceleration (80 km/h, 4500 RPM)

## Project Structure

```
ANC-simulation-project-FxLMS/
├── config.py                    # Global parameters
├── requirements.txt             # Dependencies
├── src/
│   ├── core/
│   │   └── fxlms.py            # FxLMS algorithm
│   ├── noise/
│   │   ├── engine_noise.py     # Engine harmonic generator
│   │   ├── road_noise.py       # Broadband road noise
│   │   ├── wind_noise.py       # Low-frequency wind noise
│   │   └── noise_mixer.py      # Combines noise sources
│   └── ...
├── simulations/
│   ├── step1_wave_basics.py
│   ├── step2_virtual_mics.py
│   ├── step3_speaker_interaction.py
│   ├── step4_simple_anc_ideal.py
│   ├── step5_anc_with_latency.py
│   ├── step6_fxlms_simple.py
│   └── step7_car_interior.py
└── output/
    └── plots/                   # Generated figures
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Sample rate | 16000 Hz | Audio sampling frequency |
| Filter length | 256 taps | Adaptive filter order |
| Step size (μ) | 0.001-0.01 | Learning rate |
| Frequency range | 20-300 Hz | Target noise band |

## Expected Results

- **Step 6 (Simple FxLMS)**: 15-25 dB noise reduction
- **Step 7 (Car Interior)**: 10-20 dB noise reduction

## References

Based on: "FxLMS-based Active Noise Control: A Quick Review" by Ardekani & Abdulla (APSIPA 2011)

## Dependencies

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
