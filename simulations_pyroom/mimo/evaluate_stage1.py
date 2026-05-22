"""
Stage 1 MIMO Evaluation — Three-Way Comparison

Compares for each real audio recording:
  - SISO baseline:   PlaygroundSimulation (1 ref, 1 speaker, 1 error mic)
  - Pseudo-MIMO:     MultiSpeakerSimulation (1 ref, M speakers broadcast, 1 error mic)
  - True MIMO:       MIMOSimulation (1 ref, M speakers independent, 1 error mic)

All three use the same room, same noise source, same primary path, same
reference mic, and the SAME error mic position. The only differences are:
  - Number of speakers (1 for SISO, M for both MIMO variants)
  - Algorithm (scalar FxNLMS for SISO/pseudo, MIMOFxNLMS for true MIMO)

Outputs a markdown table to stdout and a JSON file at
output/data/mimo/stage1_comparison.json.
"""

import sys
import json
import warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from playground.simulation.runner import PlaygroundSimulation, MultiSpeakerSimulation
from playground.simulation.mimo_runner import MIMOSimulation


REAL_AUDIO_FILES = [
    ('Real Car 1', 'real_noises/realcar1.wav', 30.0),
    ('Real Car 2', 'real_noises/realcar2.wav', 13.0),
    ('Real Car 3', 'real_noises/realcar3.wav', 18.0),
    ('Real Car 4', 'real_noises/realcar4.wav', 14.0),
    ('Real Car 5', 'real_noises/realcar5.wav', 25.0),
    ('LA City Start', 'real_noises/la_city_start.wav', 20.0),
    ('LA Stop & Go', 'real_noises/la_city_stop_go.wav', 20.0),
    ('LA Quiet Cruise', 'real_noises/la_quiet_cruise.wav', 20.0),
    ('LA Idle', 'real_noises/la_idle.wav', 20.0),
    ('LA Varying', 'real_noises/la_varying.wav', 20.0),
    ('LA Medium Cruise', 'real_noises/la_medium_cruise.wav', 20.0),
    ('LA Loud Low', 'real_noises/la_loud_low.wav', 20.0),
    ('LA Late Drive', 'real_noises/la_late_drive.wav', 20.0),
]

# Standard 4-speaker configuration matching playground default
FOUR_SPEAKERS = {
    'door_L': [2.0, 0.1, 0.4],
    'door_R': [2.0, 1.75, 0.4],
    'dash_L': [0.8, 0.25, 0.9],
    'dash_R': [0.8, 1.60, 0.9],
}

BASE_PARAMS = {
    'dimensions': [4.5, 1.85, 1.2],
    'absorption': 0.35,
    'max_order': 3,
    'positions': {
        'noise_source': [0.5, 0.92, 0.4],
        'reference_mic': [1.1, 0.92, 0.8],
        'speaker': [0.8, 0.25, 0.9],   # used by SISO baseline
        'error_mic': [2.5, 0.55, 1.05],
    },
    'filter_length': 512,
    'step_size': 0.003,
    'leakage': 0.0,
    'sample_rate': 16000,
    'scenario': 'highway',
}

OUTPUT_PATH = Path(__file__).parent.parent.parent / 'output' / 'data' / 'mimo' / 'stage1_comparison.json'


def run_one(sim_class, params, with_speakers=False):
    """Run a single simulation and return (dB, conv_time, stable)."""
    p = dict(params)
    if with_speakers:
        p['speakers'] = FOUR_SPEAKERS
    sim = sim_class(p)
    res = sim.run()
    if np.any(np.isnan(res['mse'])) or np.any(np.isinf(res['mse'])):
        return None, None, False
    return res['noise_reduction_db'], res['convergence_time'], True


def main():
    print("=" * 92)
    print(" Stage 1 MIMO Evaluation — SISO vs Pseudo-MIMO vs True MIMO")
    print("=" * 92)
    print(f"\n{'Recording':<20} | {'SISO':>9} | {'Pseudo-MIMO':>12} | {'True MIMO':>10} | "
          f"{'Δ vs SISO':>10} | {'Δ vs Pseudo':>11}")
    print("-" * 92)

    results = []

    for name, audio, duration in REAL_AUDIO_FILES:
        params = dict(BASE_PARAMS)
        params['audio_file'] = audio
        params['duration'] = duration

        # SISO baseline
        np.random.seed(42)
        siso_db, siso_conv, siso_ok = run_one(PlaygroundSimulation, params)

        # Pseudo-MIMO baseline
        np.random.seed(42)
        pseudo_db, pseudo_conv, pseudo_ok = run_one(MultiSpeakerSimulation, params, with_speakers=True)

        # True MIMO
        np.random.seed(42)
        mimo_db, mimo_conv, mimo_ok = run_one(MIMOSimulation, params, with_speakers=True)

        if siso_ok and pseudo_ok and mimo_ok:
            d_siso = mimo_db - siso_db
            d_pseudo = mimo_db - pseudo_db
            print(f"{name:<20} | {siso_db:>7.2f}dB | {pseudo_db:>10.2f}dB | "
                  f"{mimo_db:>8.2f}dB | {d_siso:>+8.2f}dB | {d_pseudo:>+9.2f}dB",
                  flush=True)
        else:
            print(f"{name:<20} | DIV/ERR — siso={siso_ok} pseudo={pseudo_ok} mimo={mimo_ok}",
                  flush=True)

        results.append({
            'recording': name,
            'audio_file': audio,
            'duration': duration,
            'siso': {'nr_db': siso_db, 'conv_time': siso_conv, 'stable': siso_ok},
            'pseudo_mimo': {'nr_db': pseudo_db, 'conv_time': pseudo_conv, 'stable': pseudo_ok},
            'true_mimo': {'nr_db': mimo_db, 'conv_time': mimo_conv, 'stable': mimo_ok},
            'delta_vs_siso': (mimo_db - siso_db) if (siso_ok and mimo_ok) else None,
            'delta_vs_pseudo': (mimo_db - pseudo_db) if (pseudo_ok and mimo_ok) else None,
        })

    # Aggregate stats
    siso_values = [r['siso']['nr_db'] for r in results if r['siso']['stable']]
    pseudo_values = [r['pseudo_mimo']['nr_db'] for r in results if r['pseudo_mimo']['stable']]
    mimo_values = [r['true_mimo']['nr_db'] for r in results if r['true_mimo']['stable']]
    deltas_vs_siso = [r['delta_vs_siso'] for r in results if r['delta_vs_siso'] is not None]
    deltas_vs_pseudo = [r['delta_vs_pseudo'] for r in results if r['delta_vs_pseudo'] is not None]

    print("-" * 92)
    print(f"{'MEAN':<20} | {np.mean(siso_values):>7.2f}dB | "
          f"{np.mean(pseudo_values):>10.2f}dB | {np.mean(mimo_values):>8.2f}dB | "
          f"{np.mean(deltas_vs_siso):>+8.2f}dB | {np.mean(deltas_vs_pseudo):>+9.2f}dB")

    wins_vs_siso = sum(1 for d in deltas_vs_siso if d > 0)
    wins_vs_pseudo = sum(1 for d in deltas_vs_pseudo if d > 0)

    print(f"\nMIMO wins vs SISO:        {wins_vs_siso}/{len(deltas_vs_siso)}")
    print(f"MIMO wins vs Pseudo-MIMO: {wins_vs_pseudo}/{len(deltas_vs_pseudo)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump({
            'config': {
                'speakers': FOUR_SPEAKERS,
                'base_params': {k: v for k, v in BASE_PARAMS.items() if k != 'positions'},
                'positions': BASE_PARAMS['positions'],
            },
            'results': results,
            'summary': {
                'mean_siso_db': float(np.mean(siso_values)),
                'mean_pseudo_db': float(np.mean(pseudo_values)),
                'mean_mimo_db': float(np.mean(mimo_values)),
                'mean_delta_vs_siso': float(np.mean(deltas_vs_siso)),
                'mean_delta_vs_pseudo': float(np.mean(deltas_vs_pseudo)),
                'wins_vs_siso': wins_vs_siso,
                'wins_vs_pseudo': wins_vs_pseudo,
                'total_stable': len(deltas_vs_siso),
            },
        }, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
