"""
Evaluate Binary Step Size Selector (IDLE vs Non-IDLE)

Tests the binary model against baseline fixed μ=0.005.
Strategy:
- IDLE → μ=0.015 (aggressive)
- Non-IDLE → μ=0.005 (conservative baseline)

This should give +1.47 dB on IDLE with no loss on other scenarios.
"""

import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pyroomacoustics as pra
from src.core.fxlms import FxNLMS
from src.acoustic.path_generator import FIRPath
from src.noise.noise_mixer import NoiseMixer
from src.ml.phase1_step_size.feature_extractor import extract_features
from src.ml.phase1_step_size.step_size_selector_binary import (
    BinaryStepSizeSelector, MU_IDLE, MU_DEFAULT
)
from src.ml.common.metrics import noise_reduction_db, convergence_time, stability_score


# Configuration
ROOM_DIMS = [4.5, 1.85, 1.2]
ROOM_MATERIALS = {
    'ceiling': 0.38, 'floor': 0.52,
    'east': 0.14, 'west': 0.14,
    'north': 0.20, 'south': 0.30,
}

# Multi-channel configuration (matching playground presets.py)
FOUR_SPEAKERS = {
    'door_L': [2.0, 0.1, 0.4],       # Front left door
    'door_R': [2.0, 1.75, 0.4],      # Front right door
    'dash_L': [0.8, 0.25, 0.9],      # Dashboard left
    'dash_R': [0.8, 1.60, 0.9],      # Dashboard right
}

FOUR_REF_MICS = {
    'firewall': [0.3, 0.92, 0.5],    # Engine noise detection
    'floor': [2.0, 0.55, 0.15],      # Road/tire noise
    'a_pillar': [0.5, 0.15, 1.0],    # Wind noise
    'dashboard': [0.9, 0.92, 0.8],   # General
}

ERROR_MIC_POS = [1.8, 0.55, 1.0]  # Driver's ear

SCENARIO_NOISE_POSITIONS = {
    'idle': [0.15, 0.92, 0.5],       # Engine (Firewall)
    'city': [0.5, 0.92, 0.5],        # Combined (Dashboard)
    'highway': [2.0, 0.92, 0.12],    # Road (Floor)
    'acceleration': [0.15, 0.92, 0.5], # Engine (Firewall)
}

BASELINE_STEP_SIZE = 0.005
FS = 16000
DURATION = 3.0
FILTER_LENGTH = 256

TEST_SCENARIOS = ['idle', 'city', 'highway', 'acceleration']
N_TEST_VARIATIONS = 10
TEST_SEED_OFFSET = 1000


def create_room_simulation_multi_channel(scenario: str, fs: int = FS) -> dict:
    """Create room with 4 speakers + 4 ref mics for given scenario."""
    materials = {
        name: pra.Material(coef)
        for name, coef in ROOM_MATERIALS.items()
    }

    room = pra.ShoeBox(
        ROOM_DIMS, fs=fs, materials=materials,
        max_order=3, air_absorption=True
    )

    # Scenario-specific noise source
    room.add_source(SCENARIO_NOISE_POSITIONS[scenario])

    # Add all 4 speakers
    speaker_names = list(FOUR_SPEAKERS.keys())
    for name in speaker_names:
        room.add_source(FOUR_SPEAKERS[name])

    # Build mic array: 4 ref mics + 1 error mic
    ref_mic_names = list(FOUR_REF_MICS.keys())
    mic_positions = [FOUR_REF_MICS[name] for name in ref_mic_names]
    mic_positions.append(ERROR_MIC_POS)
    mic_array = np.array(mic_positions).T
    room.add_microphone_array(pra.MicrophoneArray(mic_array, fs=fs))

    room.compute_rir()
    max_len = 512

    error_mic_idx = len(ref_mic_names)  # Index 4

    # Primary path
    primary = room.rir[error_mic_idx][0][:max_len]

    # Reference paths (will be averaged)
    reference_paths = {
        name: room.rir[i][0][:max_len]
        for i, name in enumerate(ref_mic_names)
    }

    # Secondary paths (sum of all 4 speakers)
    secondary_combined = np.zeros(max_len)
    for i, name in enumerate(speaker_names):
        rir = room.rir[error_mic_idx][i + 1][:max_len]
        secondary_combined[:len(rir)] += rir

    return {
        'primary': primary,
        'reference_paths': reference_paths,
        'secondary': secondary_combined,
        'speaker_names': speaker_names,
        'ref_mic_names': ref_mic_names,
    }


def run_simulation(noise_signal, paths, step_size):
    """Run FxNLMS simulation with given step size (multi-channel)."""
    n_samples = len(noise_signal)

    primary_path = FIRPath(paths['primary'])

    # Create FIR filter for each reference mic
    reference_path_filters = {
        name: FIRPath(paths['reference_paths'][name])
        for name in paths['ref_mic_names']
    }

    secondary_path = FIRPath(paths['secondary'])

    s_hat = paths['secondary'] * (1 + 0.05 * np.random.randn(len(paths['secondary'])))

    # CRITICAL: regularization changed to 1e-4 (matching playground)
    fxnlms = FxNLMS(
        filter_length=FILTER_LENGTH,
        step_size=step_size,
        secondary_path_estimate=s_hat,
        regularization=1e-4  # Changed from 1e-6
    )

    desired = np.zeros(n_samples)
    error = np.zeros(n_samples)

    for i in range(n_samples):
        sample = noise_signal[i]

        # AVERAGE 4 reference signals (signal fusion)
        ref_signals = {}
        for name in paths['ref_mic_names']:
            ref_signals[name] = reference_path_filters[name].filter_sample(sample)
        x = np.mean(list(ref_signals.values()))

        d = primary_path.filter_sample(sample)
        desired[i] = d

        y = fxnlms.generate_antinoise(x)
        y_at_error = secondary_path.filter_sample(y)
        e = d + y_at_error
        error[i] = e

        fxnlms.filter_reference(x)
        fxnlms.update_weights(e)

    return {
        'noise_reduction_db': noise_reduction_db(desired, error),
        'convergence_time': convergence_time(fxnlms.mse_history),
        'stability_score': stability_score(fxnlms.mse_history),
        'step_size': step_size,
    }


def run_adaptive_simulation(noise_signal, paths, model):
    """Run simulation with binary adaptive step size selection (multi-channel)."""
    n_samples = len(noise_signal)

    # Extract features from AVERAGED reference signal (first second)
    ref_path_filters = {
        name: FIRPath(paths['reference_paths'][name])
        for name in paths['ref_mic_names']
    }

    # Collect first second from all 4 ref mics
    ref_signals_1s = {name: [] for name in paths['ref_mic_names']}
    for i in range(min(FS, n_samples)):
        sample = noise_signal[i]
        for name in paths['ref_mic_names']:
            sig = ref_path_filters[name].filter_sample(sample)
            ref_signals_1s[name].append(sig)

    # Average the 4 reference signals
    ref_signal_averaged = np.mean([
        np.array(ref_signals_1s[name])
        for name in paths['ref_mic_names']
    ], axis=0)

    features = extract_features(ref_signal_averaged, FS)

    # Binary prediction: IDLE → 0.015, else → 0.005
    selected_mu = model.predict(features)
    predicted_class = model.predict_class(features)

    # Run simulation with selected step size
    result = run_simulation(noise_signal, paths, selected_mu)
    result['predicted_class'] = predicted_class  # 0=non-idle, 1=idle
    result['step_size'] = selected_mu

    return result


def main():
    print("=" * 70)
    print("Evaluating Binary Step Size Selector (IDLE vs Non-IDLE)")
    print("=" * 70)
    print(f"\nStrategy:")
    print(f"  IDLE detected     → μ = {MU_IDLE}")
    print(f"  Non-IDLE detected → μ = {MU_DEFAULT} (same as baseline)")

    model_path = Path('output/models/phase1/step_selector_binary.pt')
    output_dir = Path('output/data/phase1')
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Run train_step_selector_binary.py first.")
        return

    # Load model
    print(f"\nLoading model from {model_path}...")
    model = BinaryStepSizeSelector.load(model_path)
    print("Model loaded.")

    # Initialize noise generator
    noise_gen = NoiseMixer(FS)

    # Storage
    baseline_results = {s: [] for s in TEST_SCENARIOS}
    adaptive_results = {s: [] for s in TEST_SCENARIOS}

    total_tests = len(TEST_SCENARIOS) * N_TEST_VARIATIONS
    test_count = 0

    print("\nRunning evaluations...")

    for scenario in TEST_SCENARIOS:
        print(f"\n  Scenario: {scenario} (noise at {SCENARIO_NOISE_POSITIONS[scenario]})")

        # Create scenario-specific room
        paths = create_room_simulation_multi_channel(scenario)

        for var in range(N_TEST_VARIATIONS):
            test_count += 1
            print(f"\r    [{test_count}/{total_tests}] Variation {var+1}/{N_TEST_VARIATIONS}", end="")

            np.random.seed(TEST_SEED_OFFSET + var)
            noise_signal = noise_gen.generate_scenario(DURATION, scenario)

            # Baseline
            baseline_result = run_simulation(noise_signal, paths, BASELINE_STEP_SIZE)
            baseline_results[scenario].append(baseline_result)

            # Adaptive
            adaptive_result = run_adaptive_simulation(noise_signal, paths, model)
            adaptive_results[scenario].append(adaptive_result)

    print("\n")

    # Compute statistics
    print("\n" + "=" * 70)
    print("RESULTS BY SCENARIO")
    print("=" * 70)

    all_baseline_nr = []
    all_adaptive_nr = []

    for scenario in TEST_SCENARIOS:
        baseline_nr = [r['noise_reduction_db'] for r in baseline_results[scenario]]
        adaptive_nr = [r['noise_reduction_db'] for r in adaptive_results[scenario]]
        adaptive_mu = [r['step_size'] for r in adaptive_results[scenario]]
        adaptive_class = [r['predicted_class'] for r in adaptive_results[scenario]]

        all_baseline_nr.extend(baseline_nr)
        all_adaptive_nr.extend(adaptive_nr)

        improvement = np.mean(adaptive_nr) - np.mean(baseline_nr)
        expected_class = 1 if scenario == 'idle' else 0
        class_accuracy = np.mean([c == expected_class for c in adaptive_class])

        print(f"\n{scenario.upper()}:")
        print(f"  Baseline NR:    {np.mean(baseline_nr):.2f} ± {np.std(baseline_nr):.2f} dB")
        print(f"  Adaptive NR:    {np.mean(adaptive_nr):.2f} ± {np.std(adaptive_nr):.2f} dB")
        print(f"  Improvement:    {improvement:+.2f} dB")
        print(f"  Selected μ:     {np.mean(adaptive_mu):.4f}")
        print(f"  Class accuracy: {class_accuracy:.0%} (expected: {'IDLE' if expected_class == 1 else 'Non-IDLE'})")

    # Overall statistics
    all_baseline_nr = np.array(all_baseline_nr)
    all_adaptive_nr = np.array(all_adaptive_nr)
    improvement = all_adaptive_nr - all_baseline_nr

    mean_improvement = np.mean(improvement)
    worst_case = np.min(improvement)
    win_rate = np.mean(improvement > 0)

    print("\n" + "=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)
    print(f"Mean Improvement: {mean_improvement:+.3f} dB")
    print(f"Worst Case:       {worst_case:+.3f} dB")
    print(f"Win Rate:         {win_rate:.1%}")

    # Check Phase 1 criteria
    print("\n" + "=" * 70)
    print("PHASE 1 SUCCESS CRITERIA")
    print("=" * 70)

    # Calculate expected improvement
    # Only IDLE should improve (+1.47 dB), others stay same (0 dB change)
    # Mean = 1.47 / 4 = 0.37 dB if IDLE detection is perfect
    expected_mean = 1.47 / len(TEST_SCENARIOS)

    criteria = {
        'mean_improvement_db': (mean_improvement, 0.30, mean_improvement >= 0.30),
        'worst_case_drop_db': (worst_case, -0.1, worst_case >= -0.1),
        'win_rate': (win_rate, 0.25, win_rate >= 0.25),  # At least IDLE scenarios should win
    }

    all_passed = True
    for name, (value, target, passed) in criteria.items():
        status = "PASS" if passed else "FAIL"
        print(f"{name:25s}: {value:+.3f} (target: {target:+.3f}) [{status}]")
        if not passed:
            all_passed = False

    # Save results
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'binary',
        'baseline_step_size': BASELINE_STEP_SIZE,
        'mu_idle': MU_IDLE,
        'mu_default': MU_DEFAULT,
        'mean_improvement_db': float(mean_improvement),
        'worst_case_db': float(worst_case),
        'win_rate': float(win_rate),
        'per_scenario': {
            scenario: {
                'baseline_mean_nr': float(np.mean([r['noise_reduction_db'] for r in baseline_results[scenario]])),
                'adaptive_mean_nr': float(np.mean([r['noise_reduction_db'] for r in adaptive_results[scenario]])),
                'adaptive_mean_mu': float(np.mean([r['step_size'] for r in adaptive_results[scenario]])),
            }
            for scenario in TEST_SCENARIOS
        },
        'passed': all_passed,
    }

    results_path = output_dir / 'evaluation_results_binary.json'
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nSaved results to {results_path}")

    print("\n" + "=" * 70)
    if all_passed:
        print("BINARY MODEL EVALUATION: PASSED")
        print(f"Expected ~{expected_mean:.2f} dB mean improvement, achieved {mean_improvement:.2f} dB")
    else:
        print("BINARY MODEL EVALUATION: NEEDS IMPROVEMENT")
        print("Check IDLE detection accuracy and false positive rate.")
    print("=" * 70)


if __name__ == '__main__':
    main()
