"""
Phase 3: Evaluate Neural ANC

Compares neural ANC against FxLMS baseline.

Input: output/models/phase3/neural_anc.pt
Output: output/plots/phase3/comparison_*.png
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
from src.ml.phase3_neural.neural_anc_wrapper import NeuralANCWrapper
from src.ml.common.metrics import noise_reduction_db
from src.ml.common.comparison import is_significant_improvement


# Configuration
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

FS = 16000
DURATION = 3.0
BUFFER_LEN = 256
TEST_SCENARIOS = ['idle', 'city', 'highway']
N_TEST = 5
TEST_SEED = 3000


def create_room():
    """Create room and get paths."""
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


def run_fxlms(noise, paths):
    """Run FxLMS baseline."""
    n = len(noise)
    primary = FIRPath(paths['primary'])
    reference = FIRPath(paths['reference'])
    secondary = FIRPath(paths['secondary'])
    s_hat = paths['secondary'] * (1 + 0.05 * np.random.randn(len(paths['secondary'])))

    fxlms = FxNLMS(BUFFER_LEN, 0.005, s_hat, 1e-6)

    desired, error = np.zeros(n), np.zeros(n)
    for i in range(n):
        x = reference.filter_sample(noise[i])
        d = primary.filter_sample(noise[i])
        desired[i] = d
        y = fxlms.generate_antinoise(x)
        error[i] = d + secondary.filter_sample(y)
        fxlms.filter_reference(x)
        fxlms.update_weights(error[i])

    return noise_reduction_db(desired, error)


def run_neural(noise, paths, model_path):
    """Run neural ANC."""
    n = len(noise)
    primary = FIRPath(paths['primary'])
    reference = FIRPath(paths['reference'])
    secondary = FIRPath(paths['secondary'])

    wrapper = NeuralANCWrapper(model_path=model_path, buffer_len=BUFFER_LEN)

    desired, error = np.zeros(n), np.zeros(n)
    for i in range(n):
        x = reference.filter_sample(noise[i])
        d = primary.filter_sample(noise[i])
        desired[i] = d
        y = wrapper.generate_antinoise(x)
        error[i] = d + secondary.filter_sample(y)

    return noise_reduction_db(desired, error)


def main():
    print("=" * 70)
    print("Phase 3: Evaluating Neural ANC")
    print("=" * 70)

    model_path = Path('output/models/phase3/neural_anc.pt')
    plot_dir = Path('output/plots/phase3')
    data_dir = Path('output/data/phase3')
    plot_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    paths = create_room()
    noise_gen = NoiseMixer(FS)

    fxlms_results = {s: [] for s in TEST_SCENARIOS}
    neural_results = {s: [] for s in TEST_SCENARIOS}

    print("\nRunning evaluation...")
    for scenario in TEST_SCENARIOS:
        print(f"\n  {scenario}:")
        for i in range(N_TEST):
            print(f"\r    [{i+1}/{N_TEST}]", end="")
            np.random.seed(TEST_SEED + i)
            noise = noise_gen.generate_scenario(DURATION, scenario)
            fxlms_results[scenario].append(run_fxlms(noise, paths))
            neural_results[scenario].append(run_neural(noise, paths, model_path))

    print("\n\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    all_fxlms, all_neural = [], []
    for scenario in TEST_SCENARIOS:
        f_nr = fxlms_results[scenario]
        n_nr = neural_results[scenario]
        all_fxlms.extend(f_nr)
        all_neural.extend(n_nr)
        print(f"\n{scenario.upper()}:")
        print(f"  FxLMS:  {np.mean(f_nr):.2f} dB")
        print(f"  Neural: {np.mean(n_nr):.2f} dB")
        print(f"  Delta:  {np.mean(n_nr) - np.mean(f_nr):+.2f} dB")

    comparison = is_significant_improvement(all_fxlms, all_neural)
    print(f"\nOverall: {comparison['mean_improvement']:+.2f} dB, p={comparison['p_value']:.4f}")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'comparison': comparison,
    }
    with open(data_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nDone!")


if __name__ == '__main__':
    main()
