# Position Optimization — Speaker & Reference Mic Placement

## Overview

We discovered that **component placement** has a far greater impact on ANC performance than algorithm tuning. Moving the reference mic and speaker to optimal positions improved noise reduction from ~2 dB to 8-12 dB — a 4-6x improvement that no algorithm change (bandpass, subband, leaky) could achieve.

## Experimental Setup

- Room: Sedan preset [4.5, 1.85, 1.2] m
- Noise source: Fixed at [0.5, 0.92, 0.4] (front firewall)
- Error mic: Fixed at [2.5, 0.55, 1.05] (driver headrest)
- FxNLMS: 512 taps, step=0.003, leakage off
- Tested on: LA Loud Low-Freq, Real Car 4, LA City Start, Real Car 1

## Key Finding: Reference Mic Position

**Optimal: ~0.1m from noise source** (not co-located, not far)

| Distance to Noise | Position | Avg Reduction |
|-------------------|----------|---------------|
| 0.00 m | Co-located with noise | ~8.5 dB |
| **0.11 m** | Very near noise | **~10 dB** |
| 0.22 m | Near noise | ~8.8 dB |
| 0.58 m | Front cabin | ~4.0 dB |
| 0.72 m | Default preset | ~5.8 dB |
| 1.08 m | Mid cabin | ~5.5 dB |
| 1.62 m | Near error mic | ~3.4 dB |

**Why ~0.1m is better than 0m (co-located):**
When the reference mic is exactly at the noise source, it picks up the same reverberant field as the source itself. A small offset (~10cm) gives it a slightly different acoustic perspective that better correlates with what the error mic will receive through the primary path.

**Why further is worse:**
As distance increases, the reference signal accumulates room reflections and loses coherence with the direct path to the error mic. The adaptive filter needs a clean, predictive reference — a reverberant one forces the filter to model both the noise AND the room, wasting capacity.

## Key Finding: Speaker Position

**Optimal: 0.2-1.0m from the driver's ear (error mic)**

| Distance to Error | Position | Avg Reduction |
|-------------------|----------|---------------|
| 0.10 m | Adjacent to ear | ~10.3 dB |
| 0.21 m | Headrest speaker | ~9.5 dB |
| 0.50 m | Behind driver | ~8.5 dB |
| 0.70 m | Door panel | ~8.7 dB |
| **1.00 m** | Dash close | **~8.6 dB** |
| 1.30 m | Mid dash | ~5.8 dB |
| 1.51 m | Front-mid cabin | ~4.5 dB |
| 1.73 m | Default preset | ~3.0 dB |
| 2.01 m | Far front dash | ~4.3 dB |

**Why closer is better:**
A shorter secondary path means:
1. Less delay between anti-noise generation and arrival at the ear
2. Fewer room reflections distorting the anti-noise waveform
3. Better phase alignment — the anti-noise arrives with the correct phase for destructive interference

**The 0.7m bump (room resonance):**
The curve is not perfectly monotonic. A bump appears at ~0.7m because this distance equals the half-wavelength of ~245 Hz (a key frequency in the ANC-effective 20-300 Hz range). At half-wavelength distance, the direct path and first reflection constructively interfere, strengthening the secondary path response at that frequency and improving cancellation.

## Before vs After Optimization

| Sample | Default Positions | Optimal Positions | Improvement |
|--------|------------------|-------------------|-------------|
| LA Loud Low-Freq | 2.34 dB | **8.03 dB** | +5.69 dB |
| Real Car 4 | 4.04 dB | **12.36 dB** | +8.32 dB |
| LA City Start | 1.46 dB | **8.19 dB** | +6.73 dB |

## Optimal Positions (Sedan)

| Component | Position | Coordinates |
|-----------|----------|-------------|
| Noise source | Front firewall (fixed) | [0.5, 0.92, 0.4] |
| Reference mic | Very near noise source | [0.6, 0.92, 0.45] |
| Speaker | Door panel / headrest area | [2.0, 0.05, 0.9] or [2.3, 0.55, 1.1] |
| Error mic | Driver headrest (fixed) | [2.5, 0.55, 1.05] |

## Physical Interpretation

The two rules are complementary aspects of the same principle — **signal coherence through acoustic paths**:

1. **Reference mic close to noise** → the reference signal is a clean, early version of the noise before it gets smeared by room acoustics. The adaptive filter can extract a strong correlation between reference and error.

2. **Speaker close to ear** → the anti-noise arrives at the error mic through a short, direct path with minimal room coloring. The filter's output reaches the ear as intended, not distorted by reflections.

When both conditions are met, the system approaches the theoretical limit of single-channel feedforward ANC. When either is violated, room acoustics degrade the signal paths and the filter cannot compensate.

## Practical Implications for Real Cars

- **Headrest speakers** (common in luxury cars) are ideal — ~0.2m from the ear
- **Reference mic on firewall/engine bay partition** captures engine noise before it enters the cabin
- **Door speakers at 0.7m** benefit from the half-wavelength resonance at ~245 Hz
- Dashboard speakers at 1.5m+ are too far for effective single-channel ANC

## Plots

- `output/plots/position_optimization.png` — Top-down car layout with color-coded positions
- `output/plots/speaker_distance_vs_reduction.png` — Noise reduction vs speaker-to-ear distance curve
