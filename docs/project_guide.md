# Project Guide

A comprehensive guide for running and reviewing the ANC Simulation Project.

---

## Quick Start

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Run any command below

# 3. When done
deactivate
```

Your terminal should show `(.venv)` when activated.

---

## Running Simulations

All simulations use pyroomacoustics for realistic room acoustics.

### Learning Steps (simulations_pyroom/)

Run these in order to understand ANC fundamentals:

```bash
python simulations_pyroom/step1_room_acoustics.py   # Room acoustics basics
python simulations_pyroom/step2_microphones.py      # Microphone measurement
python simulations_pyroom/step3_superposition.py    # Superposition principle
python simulations_pyroom/step4_anc_ideal.py        # Ideal ANC (no latency)
python simulations_pyroom/step5_anc_latency.py      # Shows latency problem
python simulations_pyroom/step6_fxlms.py            # FxLMS adaptive solution
python simulations_pyroom/step7_car_interior.py     # Full car simulation
python simulations_pyroom/step8_placement_optimization.py  # Speaker/mic placement
```

### Comparison Test

```bash
python simulations_pyroom/comparison_test.py   # Compare configs across scenarios
```

---

## Interactive Playground

Launch the Streamlit GUI for interactive exploration:

```bash
streamlit run playground/app.py
```

Features:
- Adjust FxLMS parameters (step size, filter length)
- Select noise scenarios (highway, city, acceleration, idle, dynamic)
- Real-time plots and audio output
- Compare different configurations

---

## ML Phase 1 - Evaluation

### Run Full Evaluation (All Scenarios + Dynamic)

```bash
python simulations_pyroom/phase1_step_size/evaluate_step_selector_binary.py
```

This evaluates:
- **Static scenarios**: idle, city, highway, acceleration (10 variations each)
- **Dynamic scenario**: random sequence with crossfade transitions
- Compares baseline FxLMS vs adaptive ML step size

**Output:**
- Results: `output/data/phase1/evaluation_results_binary.json`
- Plots: `output/plots/phase1/`

### View Results

```bash
cat output/data/phase1/evaluation_results_binary.json | python -m json.tool
```

---

## Current Performance Summary

| Scenario | Baseline NR | Notes |
|----------|-------------|-------|
| Acceleration | ~17 dB | Best - 70% tonal engine noise |
| City | ~14 dB | Good - 50% tonal, 50% broadband |
| Idle | ~5 dB | Poor - low amplitude, low frequency |
| Highway | ~4.5 dB | Worst - 70% broadband (road + wind) |

---

## Project Structure

```
ANC-simulation-project-FxLMS/
├── .venv/                      # Virtual environment
├── config.py                   # Global parameters
├── requirements.txt            # Dependencies
├── src/
│   ├── core/fxlms.py          # FxLMS/FxNLMS algorithm
│   ├── noise/                  # Noise generators
│   │   ├── engine_noise.py
│   │   ├── road_noise.py
│   │   ├── wind_noise.py
│   │   └── noise_mixer.py
│   └── ml/                     # ML enhancements
│       ├── phase1_step_size/   # Adaptive step size
│       ├── phase2_classifier/  # Noise classification
│       └── phase3_neural/      # Neural ANC
├── simulations_pyroom/         # All simulations (step1-8, phase1)
│   ├── step1-8_*.py           # Learning steps
│   ├── comparison_test.py     # Multi-config comparison
│   └── phase1_step_size/      # ML evaluation scripts
├── playground/                 # Interactive Streamlit app
├── output/
│   ├── data/                   # JSON results
│   ├── models/                 # Trained models (.pt)
│   └── plots/                  # Generated figures
└── docs/                       # Documentation
```

---

## Key Files

| File | Purpose |
|------|---------|
| `src/core/fxlms.py` | Core FxLMS/FxNLMS implementation |
| `src/noise/noise_mixer.py` | Combines engine/road/wind noise by scenario |
| `src/ml/phase1_step_size/step_size_selector_binary.py` | ML model for step size |
| `playground/app.py` | Interactive GUI |
| `docs/ml_stage_plan.md` | ML enhancement roadmap |

---

## Troubleshooting

### ModuleNotFoundError
Make sure virtual environment is activated:
```bash
source .venv/bin/activate
```

### Missing dependencies
```bash
pip install -r requirements.txt
```

### Streamlit not found
```bash
pip install streamlit
```

---

## Development Status

- [x] Phase 1: Adaptive Step Size (binary classifier)
- [ ] Phase 2: Noise Type Classification
- [ ] Phase 3: Neural ANC

See `docs/ml_stage_plan.md` for detailed ML roadmap.

---

*Last updated: 2026-05-03*
