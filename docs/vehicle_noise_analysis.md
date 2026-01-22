# Vehicle Noise Analysis

## Driving Scenarios Overview

Combined table showing noise mixing percentages and fundamental frequency calculations:

| Scenario | RPM | Speed | Engine | Road | Wind | Fundamental (f0) | Harmonics (Hz) |
|----------|-----|-------|--------|------|------|------------------|----------------|
| **Acceleration** | 4500 | 80 km/h | 70% | 20% | 10% | **150 Hz** | 150, 300 |
| **City** | 2000 | 50 km/h | 50% | 35% | 15% | **66.7 Hz** | 67, 133, 200, 267 |
| **Highway** | 2800 | 120 km/h | 30% | 40% | 30% | **93.3 Hz** | 93, 187, 280 |
| **Idle** | 800 | 0 km/h | 90% | 5% | 5% | **26.7 Hz** | 27, 53, 80, 107, 133, 160 |

**Formula:** `f0 = (RPM / 60) × (cylinders / 2)` for 4-cylinder, 4-stroke engine

---

## Why Idle Scenario Has Low Noise Reduction

### The Problem

| Scenario | Noise Reduction |
|----------|-----------------|
| Acceleration | **19.88 dB** |
| City | **8.87 dB** |
| Highway | **5.98 dB** |
| Idle | **5.58 dB** |

Despite Idle being 90% engine noise (which should be predictable/harmonic), it achieves poor cancellation.

### Root Cause: Very Low Fundamental Frequency (26.7 Hz)

1. **Wavelength Issue**
   - Wavelength at 26.7 Hz ≈ 12.8 meters (huge compared to car cabin)
   - At 16 kHz sample rate, one period = ~600 samples
   - Our 256-tap filter may be **too short** to capture the full waveform

2. **Filter Length Problem**
   ```
   Acceleration: 150 Hz → period = 107 samples → 256 taps = 2.4 periods ✓
   Idle:         26.7 Hz → period = 600 samples → 256 taps = 0.4 periods ✗
   ```

3. **Low Signal Energy**
   - Idle engine is much quieter than acceleration
   - Lower signal-to-noise ratio
   - Adaptive filter has less "signal" to correlate with

4. **Acoustic Challenges at Low Frequencies**
   - Low frequencies create room modes and standing waves
   - Longer wavelengths diffract around seats and obstacles
   - Phase alignment is more critical (small timing error = poor cancellation)

### Potential Solutions

1. **Increase filter length** for Idle (512 or 1024 taps)
2. **Lower step size** for more stable adaptation at low frequencies
3. **ML Phase 2** could detect Idle and switch to optimized parameters

---

## Step Size (μ) and Its Effects on ANC

### The Update Equation

```
w(n+1) = w(n) + μ × e(n) × x'(n)
```

Where:
- `w` = filter weights
- `μ` = step size
- `e` = error signal (what we want to minimize)
- `x'` = filtered reference signal

### Large vs Small Step Size

| Aspect | Large μ (0.01+) | Small μ (0.001) |
|--------|-----------------|-----------------|
| **Convergence** | Fast (0.1-0.5 sec) | Slow (2-5 sec) |
| **Final MSE** | Higher (noisy) | Lower (precise) |
| **Stability** | Risk of divergence | Always stable |
| **Tracking** | Good for changing noise | Poor tracking |

### The Fundamental Trade-off

```
Large μ:  Fast convergence, but oscillates around optimum (high final MSE)
Small μ:  Slow convergence, but settles precisely at optimum (low final MSE)
```

---

## Signal Strength and Step Size Relationship

### The Core Problem

The step size's effect depends on signal power. If the reference signal `x'(n)` is large, the weight update becomes large, potentially causing instability.

### Standard LMS vs Normalized LMS

**Standard LMS:**
```
w(n+1) = w(n) + μ × e(n) × x'(n)
```
- μ must be tuned for specific signal power
- Same μ fails if signal amplitude changes

**Normalized LMS (FxNLMS - what we use):**
```
w(n+1) = w(n) + μ × e(n) × x'(n) / (x'ᵀx' + δ)
```
- Divides by signal power (`x'ᵀx'`)
- μ becomes signal-independent (in theory)
- δ = regularization (prevents division by zero)

### How Normalization Affects Effective Step Size

| Signal Amplitude | Power (x'ᵀx') | Effective μ | Result |
|------------------|---------------|-------------|--------|
| High (1.0) | ~1.0 | μ / 1.0 = μ | Normal adaptation |
| Medium (0.5) | ~0.25 | μ / 0.25 = 4μ | Faster adaptation |
| Low (0.2) | ~0.04 | μ / 0.04 = 25μ | Much faster (risky!) |

### Realistic Amplitude Scaling

Our scenarios now use realistic amplitudes:

| Scenario | Amplitude | Power | Effective μ (base μ=0.005) | Risk |
|----------|-----------|-------|----------------------------|------|
| **Acceleration** | 1.0 | 1.0 | 0.005 (normal) | None |
| **Highway** | 0.8 | 0.64 | ~0.008 | Low |
| **City** | 0.5 | 0.25 | ~0.02 | Medium |
| **Idle** | 0.2 | 0.04 | ~0.125 | **High** |

### The SNR Problem

Beyond step size, low amplitude means low Signal-to-Noise Ratio:

```
Acceleration (amp=1.0):  Strong signal, low numerical noise → High SNR → Good learning
Idle (amp=0.2):          Weak signal, same numerical noise → Low SNR → Poor learning
```

### Summary: Why Low Amplitude Hurts Performance

| Effect | High Amplitude | Low Amplitude |
|--------|----------------|---------------|
| Effective μ | As configured | Much larger (can destabilize) |
| SNR | High (good learning) | Low (noisy learning) |
| Convergence | Controlled | Fast but potentially unstable |
| Final performance | Good | May be poor |

### Possible Solutions

1. **Increase regularization (δ)** for low-amplitude scenarios
2. **Use smaller base μ** when amplitude is low
3. **ML Phase 1**: Learn to predict optimal μ based on signal power

---

## Real-World Vehicle Noise Frequency Ranges

### Noise Sources in a Car

| Noise Source | Frequency Range | Characteristics | Dominates When |
|--------------|-----------------|-----------------|----------------|
| **Engine (combustion)** | 20-500 Hz | Harmonic, RPM-dependent | Idle, acceleration |
| **Engine (mechanical)** | 500-2000 Hz | Valve train, timing chain | High RPM |
| **Road/Tire (structure-borne)** | 20-400 Hz | Low rumble through chassis | Rough roads |
| **Road/Tire (airborne)** | 400-1000 Hz | Tire-road contact patch | Highway cruising |
| **Tire tread pattern** | 800-2000 Hz | Whine/hum from tread | All speeds |
| **Wind (turbulence)** | 500-5000 Hz | Around mirrors, A-pillars | High speed (>100 km/h) |
| **Wind (buffeting)** | 10-50 Hz | Low-frequency pulsing | Window cracked open |
| **Exhaust** | 50-300 Hz | Drone, resonance | Certain RPM ranges |
| **HVAC/Fan** | 200-2000 Hz | Broadband airflow | AC/heat on |
| **Transmission whine** | 500-3000 Hz | Gear mesh frequencies | Varies with speed |

### Summary by Frequency Band

| Band | Range | Main Sources | ANC Feasibility |
|------|-------|--------------|-----------------|
| **Very Low** | 20-100 Hz | Engine idle, road rumble, exhaust | Difficult (long wavelengths) |
| **Low** | 100-500 Hz | Engine harmonics, road noise | **Best for ANC** |
| **Mid** | 500-2000 Hz | Tire noise, wind, mechanical | Moderate (needs more speakers) |
| **High** | 2000-5000 Hz | Wind, tire tread, HVAC | Passive insulation better |

### Key Insight

Most automotive ANC systems target **20-500 Hz** because:
1. These frequencies are hardest to block with passive insulation (would need heavy materials)
2. Long wavelengths create global sound fields (easier to cancel with few speakers)
3. Engine and road noise dominate driver discomfort in this range

Above 500 Hz, passive sound deadening (foam, mass) is more cost-effective than active cancellation.

**Our simulation's 20-300 Hz range is realistic for the primary ANC target zone.**

---

## Implications for ML Enhancement

### Phase 2 (Noise Classification) Opportunity

The classifier can detect the scenario and apply optimized parameters:

| Detected Class | Recommended Filter Length | Recommended Step Size |
|----------------|---------------------------|----------------------|
| Engine (high RPM) | 256 | 0.005 |
| Engine (idle) | 512-1024 | 0.002 |
| Road (broadband) | 192 | 0.008 |
| Highway (mixed) | 256 | 0.005 |

This is why ML Phase 2 should significantly improve Idle scenario performance.
