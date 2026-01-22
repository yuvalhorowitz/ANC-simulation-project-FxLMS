# Mid-Presentation Content
## Project No. 3214: Active System for Noise Reduction in a Vehicle

**Presentation Structure - 10 minutes, up to 10 slides**

---

## Slide 1: Title Slide (Mandatory)

**Project Number:** 3214

**Project Name:** Active System for Noise Reduction in a Vehicle

- **Students:**
  - Ariel Turnovsky (ID: 206483513)
  - Yuval Horowitz (ID: 206587719)
- **Supervisor:** Dr. Lior Arbel
- **Location:** Tel Aviv University
- **Supervisor Approval:** [Signature]

---

## Slide 2: Project Topic - Background and Motivation (1/2)

### Background
- Active Noise Cancellation (ANC) reduces unwanted ambient noise in enclosed spaces
- Principle: Generate anti-phase sound wave for destructive interference
- Application: Car cabin noise reduction (engine, road, wind)

### Motivation
- Intersection of Digital Signal Processing (DSP), Acoustics, and Machine Learning
- Improved passenger comfort and reduced driver fatigue
- Large cancellation zone requires adaptive system (vs. headphones/cockpits)

### Project Goal
- Develop comprehensive ANC simulation for vehicle interior
- Implement FxLMS baseline + ML enhancements
- Achieve significant noise reduction across all driving scenarios

---

## Slide 3: Project Topic - System Architecture (2/2)

### Implementation Approach
Four-stage development process:
1. **Room/Car Simulation** - Acoustic environment with pyroomacoustics
2. **FxLMS Baseline** - Classical adaptive filter implementation
3. **ML Enhancement** - 3-phase approach for improved performance
4. **Integration & Testing** - Compare ML vs baseline, optimize

### System Block Diagram
```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│ Noise Source│────▶│ ANC Controller   │────▶│ Anti-Noise  │
│             │     │ (FxLMS or ML)    │     │ (Inverted)  │
└─────────────┘     └──────────────────┘     └──────┬──────┘
                            ▲                       │
                            │    Feedback Loop      │
                            └───────────────────────┘
                                      │
                                      ▼
                              Residual Noise Output
```

---

## Slide 4: System Requirements (1/2)

### Noise Reduction Targets

| Scenario | Baseline Achieved | ML Target |
|----------|-------------------|-----------|
| Acceleration (engine-dominant) | **19.88 dB** ✓ | ≥ 18 dB |
| City (mixed noise) | **8.87 dB** | ≥ 9 dB |
| Highway (road noise) | 5.98 dB | **≥ 8 dB** |
| Idle (low-frequency) | 5.58 dB | **≥ 8 dB** |

### ML Goals
- **Improve weak scenarios**: Highway and Idle need +2 dB improvement
- **Maintain strong scenarios**: Don't regress on Acceleration and City
- **Faster convergence**: ≤ 1 second to 90% of final reduction
- **Automatic adaptation**: No manual parameter tuning needed

---

## Slide 5: System Requirements - Implementation Details (2/2)

### Development Platform
- **Language:** Python 3.10
- **Libraries:** NumPy, SciPy, PyTorch, pyroomacoustics
- **Environment:** Streamlit (interactive interface)

### Algorithmic Foundation
| Approach | Description | Status |
|----------|-------------|--------|
| **FxLMS (Baseline)** | Classical adaptive filter, linear | ✓ Implemented |
| **FxNLMS** | Normalized version, more stable | ✓ Implemented |
| **ML Enhancement** | 3-phase: adaptive μ, classification, neural ANC | In Progress |

### Why ML Enhancement?
- Different noise types need different parameters
- Adaptive step size for faster convergence
- Neural networks for non-linear acoustic modeling
- Automatic optimization without manual tuning

---

## Slide 6: Block Diagram - Current Implementation (1/2)

### FxNLMS System (Baseline - Completed)

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Noise Source│────▶│ Primary Path │────▶│ Error       │
│ (Engine/Road│     │ P(z)         │     │ Microphone  │
│ /Wind)      │     └──────────────┘     └──────┬──────┘
└──────┬──────┘                                 │
       │                                        │ e(n)
       ▼                                        │
┌──────────────┐                               │
│ Reference    │                               │
│ Microphone   │                               │
└──────┬───────┘                               │
       │ x(n)                                  │
       ▼                                       │
┌──────────────┐                               │
│ FxNLMS       │◀──────────────────────────────┘
│ Adaptive     │
│ Filter       │
└──────┬───────┘
       │ y(n) Anti-noise
       ▼
┌──────────────┐     ┌─────────────┐
│ Speakers     │────▶│ Secondary   │───▶ Cancellation
│              │     │ Path S(z)   │     at ear
└──────────────┘     └─────────────┘
```

### Implementation Details
| Component | Algorithm/Tool | Language |
|-----------|----------------|----------|
| Noise Generators | Engine/Road/Wind models | Python/NumPy |
| Acoustic Paths | Image Source Method | pyroomacoustics |
| Adaptive Filter | FxNLMS (256 taps) | Python (custom) |

---

## Slide 7: Block Diagram - ML Enhancement (2/2)

### 3-Phase ML Approach (In Progress)

```
Phase 1: Adaptive Step Size
┌─────────────┐    ┌──────────────────┐    ┌─────────────┐
│ Reference   │───▶│ Feature Extractor│───▶│ MLP: Select │
│ Signal x(n) │    │ (8 features)     │    │ optimal μ   │
└─────────────┘    └──────────────────┘    └──────┬──────┘
                                                  │
Phase 2: Noise Classification                     ▼
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Mel         │───▶│ CNN Classifier   │───▶│ FxNLMS with     │
│ Spectrogram │    │ (noise type)     │    │ adaptive params │
└─────────────┘    └──────────────────┘    └─────────────────┘

Phase 3: Neural ANC (alternative to FxLMS)
┌─────────────┐    ┌──────────────────┐    ┌─────────────┐
│ Reference   │───▶│ Neural Network   │───▶│ Anti-Noise  │
│ Buffer      │    │ (CNN/LSTM/MLP)   │    │ y(n)        │
└─────────────┘    └──────────────────┘    └─────────────┘
```

### Phase Goals
| Phase | Purpose | Expected Improvement |
|-------|---------|---------------------|
| 1 | Adaptive step size | 30-50% faster convergence |
| 2 | Noise classification | Scenario-specific optimization |
| 3 | Neural anti-noise | Non-linear modeling |

---

## Slide 8: Project Outputs - Results Achieved (1/3)

### Deliverable 1: Acoustic Simulation Environment ✓

**Completed:**
- Room/car simulation using pyroomacoustics
- Car cabin model: 4.5m × 1.85m × 1.2m (Sedan)
- Realistic material absorption (headliner, carpet, windows, dashboard)
- Multiple driving scenarios: Highway, City, Acceleration, Idle

### Deliverable 2: Noise Dataset ✓

**Noise Generators Implemented:**
| Type | Frequency Range | Characteristics |
|------|-----------------|-----------------|
| Engine | 20-500 Hz | RPM-based harmonics |
| Road | 20-1000 Hz | Speed-dependent broadband |
| Wind | 20-500 Hz | Low-frequency turbulence |

**Driving Scenario Mixing:**
- Highway: 30% engine, 40% road, 30% wind
- City: 50% engine, 35% road, 15% wind
- Acceleration: 70% engine, 20% road, 10% wind
- Idle: 90% engine, 5% road, 5% wind

---

## Slide 9: Project Outputs - Noise Reduction Results (2/3)

### FxLMS Baseline Results

**Comparison of 3 Configurations:**

| Configuration | Highway | City | Acceleration | Idle |
|---------------|---------|------|--------------|------|
| 1 ref mic, 1 speaker | 0.16 dB | 0.69 dB | 6.99 dB | 2.38 dB |
| 1 ref mic, 4 speakers | 2.95 dB | 1.83 dB | 9.19 dB | 6.55 dB |
| **4 ref mics, 4 speakers** | **5.98 dB** | **8.87 dB** | **19.88 dB** | **5.58 dB** |

### Key Findings
- Distributed system (4 mics + 4 speakers) significantly improves performance
- Engine-dominant scenarios (Acceleration) achieve best results
- Highway and Idle scenarios need ML improvement (+2 dB target)

### ML Target vs Baseline
| Scenario | Baseline | ML Target | Gap |
|----------|----------|-----------|-----|
| Highway | 5.98 dB | ≥ 8 dB | +2 dB needed |
| Idle | 5.58 dB | ≥ 8 dB | +2.5 dB needed |

---

## Slide 10: Project Outputs - Simulation Platform (3/3)

### Deliverable 3: Interactive Playground ✓

**Run:** `cd playground && streamlit run app.py`

### Capabilities
- Real-time parameter adjustment (step size, filter length)
- Multiple configurations (1/4 speakers, 1/4 reference mics)
- 4 driving scenarios with automatic noise source positioning
- Interactive room diagram with component positioning
- Visualization: waveforms, spectrum, convergence, filter coefficients
- Audio playback: before/after comparison

### Generated Outputs (per test)
- `before_after.png` - Time domain comparison
- `spectrum.png` - Frequency spectrum analysis
- `convergence.png` - MSE over time
- `room_layout.html` - Interactive car cabin diagram
- `summary_report.txt` - Quantitative results

---

## Slide 11: Updated Timeline

### Completed Milestones ✓

| Milestone | Status |
|-----------|--------|
| Literature review & algorithm study | ✓ Complete |
| pyroomacoustics simulation setup | ✓ Complete |
| FxLMS/FxNLMS baseline implementation | ✓ Complete |
| Car interior simulation (4 scenarios) | ✓ Complete |
| Configuration comparison testing | ✓ Complete |
| Interactive Playground | ✓ Complete |
| ML Phase 1 code (adaptive step size) | ✓ Complete |

### Upcoming Milestones

| Milestone | Status |
|-----------|--------|
| ML Phase 1 training & evaluation | In Progress |
| **Mid-term presentation** | **Current** |
| ML Phase 2 (noise classification) | Planned |
| ML Phase 3 (neural ANC) | Planned |
| Final comparison & optimization | Planned |
| Final poster, report, presentation | Planned |

---

## Appendix: Project Deliverables Summary

### Intermediate Deliverables
| Deliverable | Status |
|-------------|--------|
| 1. Acoustic simulation environment | ✓ Complete |
| 2. Noise generators (engine/road/wind) | ✓ Complete |
| 3. FxLMS baseline implementation | ✓ Complete |
| 4. ML infrastructure (Phase 1-3 code) | ✓ Complete |

### Final Deliverables
| Deliverable | Status |
|-------------|--------|
| 1. Trained ML models (Phase 1-3) | In Progress |
| 2. Complete simulation with ML enhancement | Planned |
| 3. Comparison report: ML vs FxLMS | Planned |

### Current Implementation Summary
- **FxNLMS baseline**: Fully functional, achieving up to 19.88 dB
- **Simulation platform**: Complete with pyroomacoustics
- **Interactive GUI**: Streamlit playground operational
- **ML code**: All 3 phases implemented, training in progress

---

## Presenter Notes

### Recommended Timing (10 minutes)
- Slides 1-3 (Introduction): 2 minutes
- Slides 4-5 (Requirements): 1.5 minutes
- Slides 6-7 (Block diagrams): 2 minutes
- Slides 8-10 (Outputs): 3 minutes
- Slide 11 (Timeline): 1.5 minutes

### Points to Emphasize
1. FxLMS baseline works well: **19.88 dB** on Acceleration
2. Identified weak scenarios: Highway (5.98 dB) and Idle (5.58 dB)
3. ML goal: Improve weak scenarios by +2 dB while maintaining strong ones
4. 3-phase ML approach provides incremental improvements
5. All code infrastructure complete, training in progress

### Expected Questions
- "Why not just use a better FxLMS configuration?"
  → Already optimized (4 ref mics, 4 speakers). Further improvement needs ML.
- "How will ML improve highway noise?"
  → Adaptive μ for faster convergence, classification for scenario-specific tuning
- "What if ML doesn't improve performance?"
  → Each phase is incremental. Phase 1 alone should help convergence speed.

### File Locations for Demo
- Results: `output/comparison_test/`
- Best result: `output/comparison_test/4ref_4spk/Acceleration/`
- Summary: `output/comparison_test/summary_report.txt`
- ML Plan: `docs/ml_stage_plan.md`
