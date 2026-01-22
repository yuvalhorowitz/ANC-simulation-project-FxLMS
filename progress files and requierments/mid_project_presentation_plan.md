# Mid-Project Presentation Plan

## Overview
- **Format**: 10 slides, 10 minutes, engineering bullets only, PPTX
- **Two Main Stages**: FxLMS Implementation (Steps 1-8) + ML Enhancement (Phase 1-3)

---

## SLIDE STRUCTURE (10 Slides)

### Slide 1: Title Slide
**Content:**
- Project Name: ML-Enhanced Active Noise Cancellation for Car Interiors
- Project Number: [Fill in]
- Student Names: [Fill in]
- Supervisor Name: [Fill in]
- Supervisor Approval: [Signature]

---

### Slide 2: Problem & Goal
**Bullets:**
- Problem: Car cabin noise (engine, road, wind) in 20-300 Hz frequency range
- Traditional passive methods (damping, insulation) add weight and cost
- Goal: Active Noise Cancellation using adaptive FxLMS algorithm + ML enhancement
- Target: 10-20 dB noise reduction at driver's ear position

---

### Slide 3: Two-Stage Approach
**Bullets:**
- **Stage 1: FxLMS Implementation** (Steps 1-8)
  - Room acoustics simulation with pyroomacoustics
  - Adaptive FxNLMS algorithm for real-time cancellation
  - Car interior scenarios: idle, city, highway
- **Stage 2: ML Enhancement** (Phase 1-3)
  - Phase 1: Adaptive step size prediction (done)
  - Phase 2: Noise type classification (planned)
  - Phase 3: Neural ANC (planned)

---

### Slide 4: System Block Diagram - FxLMS
**Include diagram:**
```
Noise Source ──→ [Primary Path P(z)] ──→ d(n) ─┐
     │                                          │
     └──→ [Ref Mic] ──→ x(n) ──→ [W(z)] ──→ y(n) ──→ [Secondary Path S(z)] ──→ y'(n)
                               │                                                  │
                               └──── [Ŝ(z)] ──→ x'(n) ──→ [LMS Update] ←── e(n) ←┘
```
**Bullets:**
- Platform: Python + pyroomacoustics for room acoustic simulation
- Algorithm: FxNLMS (Filtered-x Normalized LMS)
- Key components: Reference mic, Error mic, Adaptive filter W(z), Speaker

---

### Slide 5: Stage 1 Implementation Steps
**Table format:**

| Step | Name | Key Concept | Status |
|------|------|-------------|--------|
| 1 | Room Acoustics | RIR, reflections, absorption coefficients | Done |
| 2 | Microphones | Delays, time budget concept | Done |
| 3 | Superposition | Destructive interference principle | Done |
| 4 | Ideal ANC | Perfect cancellation benchmark (40-60 dB) | Done |
| 5 | Latency Problem | Why naive approach fails with real paths | Done |
| 6 | FxLMS | Adaptive algorithm implementation | Done |
| 7 | Car Interior | Real car cabin scenarios | Done |
| 8 | Placement | Speaker/mic position optimization | Done |

---

### Slide 6: FxLMS Results - Step 7 Car Interior
**Bullets:**
- Tested 3 car configurations: Compact (3.5m), Sedan (4.8m), SUV (4.7m)
- Noise scenarios: idle, city driving, highway

**Results Table:**
| Car Type | Scenario | Noise Reduction |
|----------|----------|-----------------|
| Compact | Highway | 5.3 dB |
| Sedan | City | 9.1 dB |
| SUV | Acceleration | 22.2 dB |

**Plot to include:** `simulations_pyroom/output/plots/pyroom_step7_config_C.png`

---

### Slide 7: Placement Optimization - Step 8
**Bullets:**
- Tested multiple speaker positions: headrest, door, dashboard, rear
- Tested multiple mic positions for reference and error signals
- Found optimal configurations for each noise type

**Plots to include:**
- `simulations_pyroom/output/plots/pyroom_step8_top_configurations.png`
- `simulations_pyroom/output/plots/pyroom_step8_heatmap_combined.png`

---

### Slide 8: ML Enhancement - Phase 1 Architecture
**Include diagram:**
```
Reference Signal x(n) [1 second window]
        │
        ▼
┌─────────────────┐
│ Feature         │ → 12 signal features
│ Extractor       │   (variance, RMS, spectral_centroid, etc.)
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Binary MLP      │ → Step size class (low μ / high μ)
│ Classifier      │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Adaptive FxNLMS │ → μ selected based on signal characteristics
└─────────────────┘
```

**Bullets:**
- Input: 12 signal features extracted from 1-second window
- Model: Binary MLP classifier (95.8% training accuracy)
- Output: Optimal step size μ for current noise conditions

---

### Slide 9: ML Results - Phase 1
**Results Table (from evaluation_results_binary.json):**
| Scenario | Baseline NR | ML-Enhanced NR | Improvement |
|----------|-------------|----------------|-------------|
| IDLE | 4.9 dB | 6.2 dB | +1.35 dB |
| CITY | 13.6 dB | 13.7 dB | +0.08 dB |
| HIGHWAY | 4.4 dB | 4.4 dB | +0.02 dB |
| **Average** | **7.6 dB** | **8.1 dB** | **+0.48 dB** |

**Statistical Significance:**
- p-value: 0.0021 (statistically significant)
- Cohen's d: 0.628 (medium effect size)
- Win rate: 73.3% (ML beats baseline in 73% of cases)

---

### Slide 10: Timeline & Next Steps
**Timeline Table:**
| Milestone | Status |
|-----------|--------|
| Step 1-4: Acoustics Basics | Done |
| Step 5-6: FxLMS Implementation | Done |
| Step 7-8: Car Interior & Optimization | Done |
| Phase 1: Adaptive Step Size | Done |
| Phase 2: Noise Classification | Planned |
| Phase 3: Neural ANC | Planned |

**Next Steps:**
- Phase 2: Train CNN classifier for noise type detection
- Phase 3: Replace FIR filter with neural network
- Final evaluation: Compare all approaches

---

## HOW TO EXTRACT PLOTS & DATA

### Step 7 Plots
```
simulations_pyroom/output/plots/pyroom_step7_config_A.png  # Compact car
simulations_pyroom/output/plots/pyroom_step7_config_B.png  # Sedan
simulations_pyroom/output/plots/pyroom_step7_config_C.png  # SUV (best - use this)
```
Each plot shows: Before/after waveform, MSE convergence, spectrum, filter coefficients

### Step 8 Plots
```
simulations_pyroom/output/plots/pyroom_step8_top_configurations.png  # Rankings
simulations_pyroom/output/plots/pyroom_step8_heatmap_combined.png    # Spatial heatmap
simulations_pyroom/output/plots/pyroom_step8_speaker_comparison.png  # Speaker comparison
```

### Step 1 Plots (for background/appendix)
```
simulations_pyroom/output/plots/pyroom_step1_rir.png          # Room impulse response
simulations_pyroom/output/plots/pyroom_step1_reflections.png  # Reflection visualization
```

### Phase 1 ML Data
```
output/data/phase1/evaluation_results_binary.json
```
Key values:
- mean_improvement: 0.48 dB
- p_value: 0.0021
- cohens_d: 0.628
- win_rate: 73.3%

Per-scenario results:
- IDLE: baseline=4.89 dB, ML=6.24 dB (+1.35 dB)
- CITY: baseline=13.57 dB, ML=13.65 dB (+0.08 dB)
- HIGHWAY: baseline=4.36 dB, ML=4.38 dB (+0.02 dB)

---

## KEY TALKING POINTS

### For FxLMS Stage (Slides 4-7):
1. **Why adaptive filtering?** - Fixed filters can't handle changing noise
2. **Why FxNLMS?** - Normalized version handles varying signal power
3. **Challenge:** Secondary path introduces delay - need to estimate it
4. **Results:** 5-22 dB reduction depending on car type and scenario

### For ML Stage (Slides 8-9):
1. **Why ML?** - Different noise types need different step sizes
2. **Phase 1 approach:** Learn optimal μ from signal features
3. **Result:** Statistically significant improvement (+0.48 dB, p=0.002)
4. **Best improvement on idle noise** (+1.35 dB) where ML selected larger step size

---

## RECOMMENDED VISUALS PER SLIDE

| Slide | Visual Type | File/Source |
|-------|-------------|-------------|
| 4 | Block diagram | Draw in PowerPoint (see ASCII above) |
| 5 | Table | Copy from above |
| 6 | 4-panel plot | `pyroom_step7_config_C.png` |
| 7 | Bar chart + heatmap | `pyroom_step8_top_configurations.png` + `heatmap_combined.png` |
| 8 | Architecture diagram | Draw in PowerPoint (see ASCII above) |
| 9 | Bar chart + table | Create from JSON data |
| 10 | Gantt/timeline | Draw in PowerPoint |

---

## VERIFICATION

Before creating the PPTX:
1. Verify all plots exist at the paths listed
2. Check that Phase 1 JSON has the latest results
3. Fill in project details (project number, names, dates)
4. Create block diagrams in PowerPoint based on ASCII art above
