"""
Scenario Performance Table — 5 models × 3 driving scenarios

Compares all five ANC configurations on the three classic driving conditions
using the LA recordings as proxies:
  IDLE         — la_idle.wav         (engine idling, near-stationary)
  CRUISING     — la_medium_cruise.wav (steady highway-style cruise)
  ACCELERATION — la_varying.wav      (varying speed with transients)

Outputs:
  - Markdown-style table to stdout
  - JSON file with full per-mic and per-scenario data
"""

import sys
import json
import warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from playground.simulation.runner import run_simulation


SCENARIOS = [
    ('IDLE',         'real_noises/la_idle.wav',          20.0),
    ('CRUISING',     'real_noises/la_medium_cruise.wav', 20.0),
    ('ACCELERATION', 'real_noises/la_varying.wav',       20.0),
]

# Driver headrest = the user's standard target
ERROR_MIC_POS = [2.5, 0.55, 1.05]
HEAD_ZONE_RADIUS_M = 0.05  # ±5 cm head-zone for Stage 2/3

FOUR_SPEAKERS = {
    'door_L':  [2.0, 0.10, 0.4],
    'door_R':  [2.0, 1.75, 0.4],
    'dash_L':  [0.8, 0.25, 0.9],
    'dash_R':  [0.8, 1.60, 0.9],
}
FOUR_REF_MICS = {
    'firewall':  [0.3, 0.92, 0.5],
    'floor':     [2.0, 0.55, 0.15],
    'a_pillar':  [0.5, 0.15, 1.0],
    'dashboard': [0.9, 0.92, 0.8],
}

BASE_PARAMS = {
    'dimensions': [4.5, 1.85, 1.2],
    'absorption': 0.35, 'max_order': 3,
    'positions': {
        'noise_source': [0.5, 0.92, 0.4],
        'reference_mic': [1.1, 0.92, 0.8],
        'speaker': [0.8, 0.25, 0.9],
        'error_mic': ERROR_MIC_POS,
    },
    'leakage': 0.0,
    'sample_rate': 16000,
    'scenario': 'highway',
    'filter_length': 512,  # bigger filter for fairness; Stage 3 will use 256
}

# Five model configurations
def model_configs(audio, duration):
    cx, cy, cz = ERROR_MIC_POS
    d = HEAD_ZONE_RADIUS_M
    head_zone_mics = [
        [cx, cy + d, cz + d], [cx, cy - d, cz + d],
        [cx, cy + d, cz - d], [cx, cy - d, cz - d],
    ]

    base = dict(BASE_PARAMS)
    base['audio_file'] = audio
    base['duration'] = duration

    return [
        # Label, params overrides
        ('SISO', dict(base, speaker_mode='Single Speaker',
                      ref_mic_mode='Single Reference Mic',
                      mimo_mode='Off',
                      step_size=0.003)),
        ('Pseudo-SIMO', dict(base, speaker_mode='4-Speaker System',
                              ref_mic_mode='Single Reference Mic',
                              mimo_mode='Off',
                              speakers=dict(FOUR_SPEAKERS),
                              step_size=0.003)),
        ('Stage 1 SIMO', dict(base, speaker_mode='4-Speaker System',
                               ref_mic_mode='Single Reference Mic',
                               mimo_mode='Stage 1 SIMO (1×M×1)',
                               speakers=dict(FOUR_SPEAKERS),
                               step_size=0.003)),
        ('Stage 2 SIMO+multi-error', dict(base,
                                           speaker_mode='4-Speaker System',
                                           ref_mic_mode='Single Reference Mic',
                                           mimo_mode='Stage 2 SIMO+multi-error (1×M×K)',
                                           speakers=dict(FOUR_SPEAKERS),
                                           error_mics_positions=head_zone_mics,
                                           step_size=0.003)),
        ('Stage 3 Full MIMO', dict(base,
                                    speaker_mode='4-Speaker System',
                                    ref_mic_mode='4-Reference Mic System',
                                    mimo_mode='Stage 3 Full MIMO (N×M×K)',
                                    speakers=dict(FOUR_SPEAKERS),
                                    ref_mics=dict(FOUR_REF_MICS),
                                    error_mics_positions=head_zone_mics,
                                    filter_length=256,  # smaller for stability
                                    step_size=0.001)),
    ]


def run_one(label, params):
    """Run a single config, return (nr_db, conv_time, stable)."""
    np.random.seed(42)
    res = run_simulation(params)
    if not res.get('success'):
        return None, None, False
    if np.any(np.isnan(res['mse'])) or np.any(np.isinf(res['mse'])):
        return None, None, False
    nr_db = float(res['noise_reduction_db'])
    conv = float(res['convergence_time'])
    return nr_db, conv, True


def main():
    print("=" * 95)
    print(" Scenario Performance Comparison (5 models × 3 driving scenarios)")
    print("=" * 95)

    print(f"\n{'Algorithm':<28} | "
          f"{'IDLE':^17} | {'CRUISING':^17} | {'ACCELERATION':^17}")
    print(f"{'':<28} | {'NR (dB)':>8} {'Conv (s)':>8} | "
          f"{'NR (dB)':>8} {'Conv (s)':>8} | "
          f"{'NR (dB)':>8} {'Conv (s)':>8}")
    print("-" * 95)

    all_results = {}

    # We rebuild model configs for each scenario, so the audio_file/duration are right
    sample_configs = model_configs(SCENARIOS[0][1], SCENARIOS[0][2])
    model_labels = [c[0] for c in sample_configs]

    for label in model_labels:
        row_results = {}
        cells = []
        for scen_name, audio, duration in SCENARIOS:
            configs = model_configs(audio, duration)
            params = dict(c for c in configs if c[0] == label)[label]
            nr, conv, stable = run_one(label, params)
            if stable:
                cells.append(f"{nr:>+7.2f} {conv:>8.2f}")
                row_results[scen_name] = {'nr_db': nr, 'conv_s': conv, 'stable': True}
            else:
                cells.append(f"{'DIVERGED':>17}")
                row_results[scen_name] = {'nr_db': None, 'conv_s': None, 'stable': False}
        print(f"{label:<28} | {cells[0]:>17} | {cells[1]:>17} | {cells[2]:>17}",
              flush=True)
        all_results[label] = row_results

    print()
    print("Configuration:")
    print(f"  Filter length: {BASE_PARAMS['filter_length']} taps "
          f"(Stage 3 uses 256 due to 4× more weights)")
    print(f"  Step size: 0.003 (Stage 3 uses 0.001 for stability)")
    print(f"  Head-zone radius (Stage 2/3): ±{HEAD_ZONE_RADIUS_M*100:.0f} cm "
          f"(K=4 error mics in 2×2 grid)")
    print(f"  Error mic position: {ERROR_MIC_POS}")
    print(f"  Recordings: idle, medium cruise, varying (LA driving downtown)")

    out_path = Path(__file__).parent.parent.parent / 'output' / 'data' / 'mimo' / 'scenario_comparison.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            'scenarios': {
                name: {'audio': audio, 'duration_s': duration}
                for name, audio, duration in SCENARIOS
            },
            'results': all_results,
            'config': {
                'filter_length_default': BASE_PARAMS['filter_length'],
                'filter_length_stage3': 256,
                'step_size_default': 0.003,
                'step_size_stage3': 0.001,
                'head_zone_radius_m': HEAD_ZONE_RADIUS_M,
                'error_mic_pos': ERROR_MIC_POS,
            },
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
