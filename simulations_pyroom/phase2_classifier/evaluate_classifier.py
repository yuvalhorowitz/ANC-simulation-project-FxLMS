"""
Phase 2: Evaluate Classifier for ANC

Tests classifier-based parameter selection against fixed baseline.

Input: output/models/phase2/noise_classifier.pt
Output: output/plots/phase2/anc_comparison_*.png
"""

import numpy as np
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pyroomacoustics as pra
from src.core.fxlms import FxNLMS
from src.acoustic.path_generator import FIRPath
from src.noise.noise_mixer import NoiseMixer
from src.ml.phase2_classifier.classified_fxlms import ClassifiedFxNLMS
from src.ml.common.metrics import noise_reduction_db, convergence_time, stability_score
from src.ml.common.comparison import is_significant_improvement


# Configuration (same as Phase 1)
ROOM_DIMS = [4.5, 1.85, 1.2]
ROOM_MATERIALS = {
    'ceiling': 0.38, 'floor': 0.52,
    'east': 0.14, 'west': 0.14,
    'north': 0.20, 'south': 0.30,
}
NOISE_SOURCE_POS = [0.3, 0.92, 0.4]
REF_MIC_POS = [0.3, 0.92, 0.5]
SPEAKER_POS = [1.9, 0.55, 1.0]
ERROR_MIC_POS = [1.8, 0.55, 1.0]

BASELINE_STEP_SIZE = 0.005
BASELINE_FILTER_LENGTH = 256
FS = 16000
DURATION = 3.0
TEST_SCENARIOS = ['idle', 'city', 'highway']
N_TEST = 5
TEST_SEED = 2000


def create_room():
    """Create room and compute RIRs."""
    materials = {k: pra.Material(v) for k, v in ROOM_MATERIALS.items()}
    room = pra.ShoeBox(ROOM_DIMS, fs=FS, materials=materials, max_order=3, air_absorption=True)
    room.add_source(NOISE_SOURCE_POS)
    room.add_source(SPEAKER_POS)
    room.add_microphone_array(pra.MicrophoneArray(np.array([REF_MIC_POS, ERROR_MIC_POS]).T, fs=FS))
    room.compute_rir()
    return {
        'primary': room.rir[1][0][:512],
        'reference': room.rir[0][0][:512],
        'secondary': room.rir[1][1][:512],
    }


def run_baseline(noise_signal, paths):
    """Run baseline FxNLMS."""
    n_samples = len(noise_signal)
    primary = FIRPath(paths['primary'])
    reference = FIRPath(paths['reference'])
    secondary = FIRPath(paths['secondary'])

    s_hat = paths['secondary'] * (1 + 0.05 * np.random.randn(len(paths['secondary'])))
    fxnlms = FxNLMS(BASELINE_FILTER_LENGTH, BASELINE_STEP_SIZE, s_hat, 1e-6)

    desired, error = np.zeros(n_samples), np.zeros(n_samples)
    for i in range(n_samples):
        x = reference.filter_sample(noise_signal[i])
        d = primary.filter_sample(noise_signal[i])
        desired[i] = d
        y = fxnlms.generate_antinoise(x)
        error[i] = d + secondary.filter_sample(y)
        fxnlms.filter_reference(x)
        fxnlms.update_weights(error[i])

    return {
        'noise_reduction_db': noise_reduction_db(desired, error),
        'convergence_time': convergence_time(fxnlms.mse_history),
        'stability_score': stability_score(fxnlms.mse_history),
    }


def run_classified(noise_signal, paths, model_path):
    """Run classified FxNLMS."""
    n_samples = len(noise_signal)
    primary = FIRPath(paths['primary'])
    reference = FIRPath(paths['reference'])
    secondary = FIRPath(paths['secondary'])

    s_hat = paths['secondary'] * (1 + 0.05 * np.random.randn(len(paths['secondary'])))

    classified = ClassifiedFxNLMS(s_hat, model_path, fs=FS)

    # Get reference signal for initialization
    ref_init = np.zeros(FS)
    temp_ref = FIRPath(paths['reference'])
    for i in range(min(FS, n_samples)):
        ref_init[i] = temp_ref.filter_sample(noise_signal[i])

    noise_class, params = classified.initialize(ref_init)

    desired, error = np.zeros(n_samples), np.zeros(n_samples)
    for i in range(n_samples):
        x = reference.filter_sample(noise_signal[i])
        d = primary.filter_sample(noise_signal[i])
        desired[i] = d
        y = classified.generate_antinoise(x)
        error[i] = d + secondary.filter_sample(y)
        classified.filter_reference(x)
        classified.update_weights(error[i])

    return {
        'noise_reduction_db': noise_reduction_db(desired, error),
        'convergence_time': convergence_time(classified.mse_history),
        'stability_score': stability_score(classified.mse_history),
        'noise_class': noise_class,
        'step_size': params.step_size,
        'filter_length': params.filter_length,
    }


def main():
    print("=" * 70)
    print("Phase 2: Evaluating Noise Classifier for ANC")
    print("=" * 70)

    model_path = Path('output/models/phase2/noise_classifier.pt')
    plot_dir = Path('output/plots/phase2')
    data_dir = Path('output/data/phase2')
    plot_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    print("\nCreating room...")
    paths = create_room()
    noise_gen = NoiseMixer(FS)

    baseline_results = {s: [] for s in TEST_SCENARIOS}
    classified_results = {s: [] for s in TEST_SCENARIOS}

    print("\nRunning evaluation...")
    for scenario in TEST_SCENARIOS:
        print(f"\n  {scenario}:")
        for i in range(N_TEST):
            print(f"\r    [{i+1}/{N_TEST}]", end="")
            np.random.seed(TEST_SEED + i)
            noise = noise_gen.generate_scenario(DURATION, scenario)
            baseline_results[scenario].append(run_baseline(noise, paths))
            classified_results[scenario].append(run_classified(noise, paths, model_path))

    print("\n\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    all_baseline, all_classified = [], []
    for scenario in TEST_SCENARIOS:
        b_nr = [r['noise_reduction_db'] for r in baseline_results[scenario]]
        c_nr = [r['noise_reduction_db'] for r in classified_results[scenario]]
        all_baseline.extend(b_nr)
        all_classified.extend(c_nr)
        print(f"\n{scenario.upper()}:")
        print(f"  Baseline: {np.mean(b_nr):.2f} dB")
        print(f"  Classified: {np.mean(c_nr):.2f} dB")
        print(f"  Improvement: {np.mean(c_nr) - np.mean(b_nr):+.2f} dB")

    comparison = is_significant_improvement(all_baseline, all_classified)
    print(f"\nOverall: {comparison['mean_improvement']:+.2f} dB, p={comparison['p_value']:.4f}")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'comparison': comparison,
        'per_scenario': {s: {'baseline': np.mean([r['noise_reduction_db'] for r in baseline_results[s]]),
                            'classified': np.mean([r['noise_reduction_db'] for r in classified_results[s]])}
                        for s in TEST_SCENARIOS}
    }
    with open(data_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nDone!")


if __name__ == '__main__':
    main()
