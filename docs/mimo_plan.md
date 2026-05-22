# MIMO ANC: Design Document and Implementation Plan

## Status: Design only — not yet implemented

This document describes the design for true MIMO (Multiple-Input, Multiple-Output) Active Noise Control as a planned future direction for this project. It captures:
- What our current code actually does (pseudo-MIMO)
- What true MIMO would mean
- Why MIMO matters for car cabins (the spatial robustness argument)
- A staged implementation plan with explicit isolation from the existing code

---

## 1. Status Quo: What We Have Today

All 6 simulation classes in `playground/simulation/runner.py` use a single scalar `FxNLMS` instance from `src/core/fxlms.py`. The `FxNLMS` class is strictly scalar:

```python
def generate_antinoise(self, x: float) -> float:
def update_weights(self, e: float) -> None:
```

There is no matrix support. Multi-channel features in the playground are achieved via two patterns:

- **Multi-reference-mic mode**: N reference signals are *averaged* into a single scalar before being fed to the adaptive filter.
- **Multi-speaker mode**: A single anti-noise signal y(n) is *broadcast* to all M speakers — every speaker emits the same waveform.

The result is **pseudo-MIMO**: topologically multi-channel, but algorithmically equivalent to single-channel ANC with averaging at the input and broadcasting at the output.

### Why this matters in physical terms

A SISO (single-input, single-output) ANC system creates a tiny spherical "zone of quiet" centered on the error microphone. If the listener moves their head a few centimeters, they exit the quiet zone — and at certain distances the noise actually gets *louder* (the waterbed effect we observed in our spectrum analysis). This is the same physics described in standard ANC literature.

Our pseudo-MIMO approach inherits this limitation. With four speakers all emitting the same signal, we still optimize for a single error point, just with more anti-noise energy. The quiet zone remains small.

---

## 2. Channel Topology Taxonomy

Mapping our 6 simulation classes to standard ANC terminology:

| Sim Class | Inputs (ref mics) | Outputs (speakers) | Error mics | Algorithm | Standard Term |
|-----------|------------------:|-------------------:|-----------:|-----------|---------------|
| `PlaygroundSimulation` | 1 | 1 | 1 | scalar FxNLMS | SISO |
| `MultiRefMicSimulation` | N (averaged) | 1 | 1 | scalar FxNLMS | "MISO" topology, SISO algorithm |
| `MultiSpeakerSimulation` | 1 | M (broadcast) | 1 | scalar FxNLMS | "SIMO" topology, SISO algorithm |
| `MultiRefMicMultiSpeakerSimulation` | N (averaged) | M (broadcast) | 1 | scalar FxNLMS | "MIMO" topology, SISO algorithm |
| `DynamicRideSimulation` | 1 | 1 | 1 | scalar FxNLMS | SISO |
| `DynamicRideMultiRefMicSimulation` | N (averaged) | 1 | 1 | scalar FxNLMS | "MISO" topology, SISO algorithm |

**None of these is true MIMO.** They all use one scalar adaptive filter under the hood.

---

## 3. What True MIMO Means

In ANC literature, "MIMO" is the gold standard for vehicle cabins. The defining features:

- **N reference signals** kept *independent* (no averaging at the input)
- **M independent anti-noise signals** (one per speaker, not broadcast)
- **K error microphones** defining a *region* of cancellation (not a single point)
- The filter becomes a **matrix** W of shape M × N — each weight is per (speaker, ref) pair
- **Cross-coupling**: every speaker emits a signal that propagates through *every* secondary path s_{m,k} (speaker m to error mic k)

The cost function expands from a single scalar mean-squared error to a sum across all error mics:

```
J = Σ_k E[e_k(n)²]
```

The scalar weight update from our poster:
```
w(n+1) = w(n) − μ · e(n) · xf(n)
```

becomes a matrix update:
```
W(n+1) = W(n) − μ · E(n) · Xf^T(n)
```

(with appropriate dimensions for the per-(speaker, ref-mic, error-mic) cross-coupling).

---

## 4. Why MIMO Solves Problems Pseudo-MIMO Cannot

### The spatial argument

Pseudo-MIMO with 4 speakers all emitting the same signal creates 4 copies of the same anti-noise pattern. It cannot create *independent* cancellation patterns at different points in space.

True MIMO with K=4 error mics distributed around a head-sized area can co-optimize the M speaker signals to minimize noise at *all K points simultaneously*. The result is a wide quiet zone covering the entire physical area where a human head would be.

### Quantitative claims to validate experimentally

Based on the design hypothesis:

| Configuration | At ear center | 5 cm offset | 10 cm offset |
|---------------|---------------|-------------|--------------|
| SISO | ~15 dB | ~0 dB | negative (amplifies) |
| MIMO (K=4 error mics) | ~10–15 dB | ~10 dB | >10 dB |

The headline result MIMO is supposed to deliver: a **flat, robust attenuation curve** rather than SISO's sharp spike that collapses with head movement.

### Why this matters for the project narrative

Our position-optimization analysis already showed that placement matters more than algorithm tuning — but it presumed a single, fixed listener position. In a real car the driver's head moves continuously. A system that gives 15 dB at the headrest and 0 dB 5 cm away is not deployable. MIMO is the algorithmic answer to head-movement robustness.

---

## 5. Recommended Implementation Scope

A staged approach lets us validate the algorithm before tackling the full matrix.

### Stage 1 — 1×M×1 MIMO (1 ref, M speakers, 1 error mic)
- Simplest tractable variant
- Per-speaker independent filters with per-speaker secondary paths
- Sanity test: with M=1, must produce numerically identical results to scalar FxNLMS
- Validates the algorithm core without multi-error-mic complexity
- ~4 hours

### Stage 2 — 1×M×K MIMO (the real win)
- Add K=4 error mics distributed around a head-sized area
- Cost function = sum of squared errors across all K
- Generates the two key poster graphs (see §9)
- This is where the spatial argument becomes provable
- ~6 additional hours

### Stage 3 — Full N×M×K MIMO (matrix filter)
- All N reference mics kept independent
- Full matrix update with cross-coupling
- Most general, most computationally expensive
- Demonstrates the engineering challenge described in standard literature
- ~6 additional hours

---

## 6. Algorithm Specification — Stage 1 (1×M×1)

```
For each speaker m ∈ {1..M}:
  Maintain independent filter weights w_m of length L
  Maintain independent secondary path estimate s_hat_m
  Maintain independent filtered-reference buffer xf_m

Per sample n:
  Anti-noise from speaker m:  y_m(n) = w_m^T · x_buffer
  Total at error mic:          y_total(n) = Σ_m (y_m * s_m)
                               (each speaker's signal passes through its OWN real secondary path)
  Error:                       e(n) = d(n) + y_total(n)
  Update each filter:          w_m(n+1) = (1 − μ·γ) · w_m(n)
                                          − (μ · e(n) / norm) · xf_m_buffer
```

The single scalar error feeds back into all M filter updates. Each filter has its own filtered reference because each speaker has its own secondary path.

---

## 7. Algorithm Specification — Stage 2 (1×M×K)

```
For each (speaker m, error mic k) pair:
  Maintain s_hat_{m,k} — secondary path estimate from speaker m to error mic k

For each speaker m:
  Maintain weights w_m and per-error-mic filtered-reference buffers xf_{m,k}

Per sample n:
  Anti-noise:           y_m(n) = w_m^T · x_buffer
  Error at mic k:       e_k(n) = d_k(n) + Σ_m (y_m * s_{m,k})
  Update speaker m:     w_m(n+1) = (1 − μγ) · w_m(n)
                                   − (μ / norm) · Σ_k (e_k(n) · xf_{m,k}_buffer)
```

The update for each speaker m sums contributions from all K error mics — this is the cross-coupling that creates the spatial co-optimization.

---

## 8. Stability Considerations

- M independent filters × L taps each = M× the weight count of scalar FxNLMS. Convergence will be slower; expect to need smaller step sizes.
- Cross-coupling can cause oscillation if step size is too aggressive. Start at μ ≤ 0.001 and only increase after validating stability.
- Regularization (the δ in NLMS) should scale with M to prevent ill-conditioning.
- Per-speaker secondary path estimate errors compound — if each s_hat_m has 5% error, the combined system has more degrees of freedom for the error to manifest as instability.

---

## 9. Evaluation Plan

### Sanity tests
- M=1, K=1 case must match scalar FxNLMS to numerical precision
- M=4, K=1 with all secondary paths *identical* must reduce to pseudo-MIMO behavior

### Numerical comparisons
- All 13 real audio recordings, 4 speakers, single error mic: compare 1×M×1 MIMO to existing pseudo-MIMO baseline AND to SISO baseline
- All 13 recordings with 4 speakers and 4 error mics: compare 1×M×K MIMO to baseline

### Spatial heatmaps showing cancellation AND amplification (added requirement)

For each of {SISO, pseudo-MIMO, true MIMO Stage 1, true MIMO Stage 2}, generate a 2D heatmap of the head zone:
- Train the filter once (with the canonical error mic position(s))
- After training, evaluate noise reduction at each point on a 30×30 cm grid
- Use a **diverging colormap** so positive dB (cancellation) is one color and negative dB (amplification) is the opposite color, with white at 0 dB
- This visually exposes the *waterbed effect*: SISO shows a sharp blue dot at center surrounded by red rings (amplification at off-center points). MIMO is expected to show a wider blue zone with smaller/no red regions.

Output: `output/plots/cancellation_heatmaps.png` (4-panel figure: SISO, pseudo-MIMO, MIMO-K=1, MIMO-K=4)

### Two key poster graphs

**Graph 1 — Spatial Attenuation Heatmap ("Zone of Quiet"):**
- Top-down 2D contour plot of a 30×30 cm grid centered on the driver headrest
- Color intensity = noise attenuation (dB)
- SISO result: deep blue concentrated dot, attenuation drops off in a tight circle
- MIMO result: wide dispersed blue cloud covering the head-sized area
- Visually proves MIMO is required to cover the physical space of a human head

**Graph 2 — Attenuation vs Head Movement (Robustness):**
- X-axis: distance of evaluation point from the central error mic (cm)
- Y-axis: attenuation (dB)
- SISO curve: sharp spike — 15 dB at 0 cm, ~0 dB at 5 cm, possibly negative at 10 cm
- MIMO curve: much flatter — maintains >10 dB out to 10–15 cm of offset
- Proves the system works for a real driver who moves their head naturally

---

## 10. Separation Guarantees (Critical)

**The current pseudo-MIMO setup must remain the default and fully reproducible.** Any future MIMO implementation will live in **new files only**.

### NEW files to be added:
- `src/core/mimo_fxnlms.py` — `MIMOFxNLMS` class
- `playground/simulation/runner_mimo.py` — new simulation class (or new class within `runner.py` clearly separated)
- `simulations_pyroom/mimo/evaluate_mimo.py` — comparison runner: pseudo-MIMO vs true MIMO
- `scripts/plots/plot_zone_of_quiet.py` — Graph 1 generator
- `scripts/plots/plot_head_movement.py` — Graph 2 generator

### Files that must NOT be modified:
- `src/core/fxlms.py` (scalar FxNLMS stays exactly as is)
- All 6 existing simulation classes in `playground/simulation/runner.py`
- `playground/presets.py` (no preset value changes)
- The current default sidebar behavior

### Sidebar integration:
A new toggle "True MIMO (independent per-speaker filters)" will be added to the sidebar, **off by default**. When enabled (and 4-speaker mode is active), the runner will instantiate `MIMOSimulation` instead of the existing `MultiSpeakerSimulation`. The default user experience remains unchanged.

---

## 11. Effort Estimate

| Stage | Scope | Effort |
|-------|-------|--------|
| Stage 1 | 1×M×1 MIMO + sanity tests | ~4 hours |
| Stage 2 | 1×M×K MIMO + the two poster graphs | ~6 hours |
| Stage 3 | Full N×M×K matrix filter | ~6 hours |
| **Total** | **Stages 1+2 (poster-ready)** | **~10 hours** |
| **Total** | **All three stages** | **~16 hours** |

For the finals project, Stages 1+2 are sufficient: they produce both the algorithm and the two graphs that tell the spatial-robustness story.

---

## 12. Why This Is Different From Our Failed ML Attempts

Our previous attempts (Phase 1, Phase 2 v1, Phase 2 v2 — see `docs/ml_journey.md`) all tried to use ML to *tune the hyperparameters* of a scalar FxLMS. They failed because FxLMS already adapts at the sample level and our test recordings were too stationary to benefit from meta-learning.

True MIMO is a fundamentally different kind of improvement: it changes the **algorithm structure**, not the hyperparameters. The improvement comes from physics — independent control of multiple speakers, multiple optimization points — not from clever parameter selection. This is why it has real headroom that further parameter tuning does not.

---

## File Map (when implemented)

| File | Status | Purpose |
|------|--------|---------|
| `docs/mimo_plan.md` | THIS FILE | Design document |
| `src/core/mimo_fxnlms.py` | TBD | `MIMOFxNLMS` class |
| `playground/simulation/runner_mimo.py` | TBD | MIMO simulation class |
| `simulations_pyroom/mimo/evaluate_mimo.py` | TBD | Pseudo vs true MIMO comparison |
| `scripts/plots/plot_zone_of_quiet.py` | TBD | Spatial heatmap |
| `scripts/plots/plot_head_movement.py` | TBD | Robustness curve |
| `output/plots/zone_of_quiet.png` | TBD | Graph 1 output |
| `output/plots/head_movement_robustness.png` | TBD | Graph 2 output |
