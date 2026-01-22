"""
Evaluate Binary Step Size Selector on Phase 1 Criteria

Tests the binary model (Low μ vs High μ) against baseline fixed μ=0.005
to see if it meets Phase 1 success criteria.
"""

import numpy as np
import pyroomacoustics as pra
import json
from pathlib import Path
import sys
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.fxlms import FxNLMS
from src.noise.noise_mixer import NoiseMixer
from src.ml.phase1_step_size.feature_extractor import extract_features
from src.ml.phase1_step_size.step_size_selector_binary import BinaryStepSizeSelector


# Simulation parameters
FS = 16000
DURATION = 3.0
N_SAMPLES = int(FS * DURATION)
FILTER_LENGTH = 256
BASELINE_STEP_SIZE = 0.005

# Test scenarios
SCENARIOS = ['idle', 'city', 'highway']
N_VARIATIONS = 10  # variations per scenario


def noise_reduction_db(desired, error, steady_state_start=None):
    """Calculate noise reduction in dB."""
    if steady_state_start is None:
        steady_state_start = len(desired) // 2

    d_power = np.mean(desired[steady_state_start:]**2)
    e_power = np.mean(error[steady_state_start:]**2)

    return 10 * np.log10(d_power / (e_power + 1e-10))


def convergence_time(mse_history, threshold_ratio=0.1):
    """Find when MSE drops below threshold."""
    if len(mse_history) == 0:
        return N_SAMPLES

    initial_mse = np.mean(mse_history[:min(100, len(mse_history))])
    threshold = initial_mse * threshold_ratio

    for i, mse in enumerate(mse_history):
        if mse < threshold:
            return i

    return len(mse_history)


def stability_score(error, divergence_threshold=10.0):
    """Check if the filter diverged."""
    rms_error = np.sqrt(np.mean(error**2))
    max_error = np.max(np.abs(error))

    # Diverged if max error is much larger than RMS
    if max_error > divergence_threshold * rms_error:
        return 0.0

    return 1.0


def create_room_and_paths():
    """Create simple room and get acoustic paths."""
    room_dim = [8, 6, 3]
    room = pra.ShoeBox(room_dim, fs=FS, max_order=3)

    # Source positions
    ref_source_pos = [2.0, 3.0, 1.5]
    sec_source_pos = [6.0, 3.0, 1.5]
    error_mic_pos = [4.0, 3.0, 1.2]

    # Add sources
    room.add_source(ref_source_pos)
    room.add_source(sec_source_pos)

    # Add microphone
    room.add_microphone(error_mic_pos)

    # Simulate to get impulse responses
    room.compute_rir()

    # Extract paths
    primary_path = room.rir[0][0]  # ref source to error mic
    secondary_path = room.rir[0][1]  # secondary source to error mic

    return primary_path, secondary_path


def simulate_anc(scenario: str, variation: int, step_size: float, use_adaptive: bool = False, model=None):
    """
    Run ANC simulation for a scenario.

    Args:
        scenario: 'idle', 'city', or 'highway'
        variation: variation number
        step_size: fixed step size (if not adaptive)
        use_adaptive: if True, use model to select step size
        model: binary model (required if use_adaptive=True)

    Returns:
        dict with results
    """
    # Generate noise (set seed for reproducibility)
    np.random.seed(1000 + variation)
    mixer = NoiseMixer(sample_rate=FS)
    noise_signal = mixer.generate_scenario(DURATION, scenario)

    # Get room paths
    primary_path, secondary_path = create_room_and_paths()

    # Reference signal (noise passed through primary path)
    reference = np.convolve(noise_signal, primary_path)[:N_SAMPLES]

    # Desired signal at error mic (noise without ANC)
    desired = np.convolve(noise_signal, primary_path)[:N_SAMPLES]

    # Extract features and select step size if adaptive
    if use_adaptive and model is not None:
        # Extract features from first 1 second
        feature_window = reference[:FS]
        features = extract_features(feature_window, FS)

        # Predict class
        class_idx = model.predict(features)

        # Map class to step size
        # Class 0 (low): use 0.005
        # Class 1 (high): use 0.012 (midpoint of 0.010-0.015)
        CLASS_TO_STEP_SIZE = {
            0: 0.005,   # Low μ (for CITY/HIGHWAY)
            1: 0.012    # High μ (for IDLE)
        }
        step_size = CLASS_TO_STEP_SIZE[class_idx]

    # Create FxLMS controller
    controller = FxNLMS(
        filter_length=FILTER_LENGTH,
        step_size=step_size,
        secondary_path_estimate=secondary_path[:FILTER_LENGTH]
    )

    # Run simulation
    error_signal = np.zeros(N_SAMPLES)
    mse_history = []

    for n in range(N_SAMPLES):
        x = reference[n]
        d = desired[n]

        # Generate anti-noise
        y = controller.generate_antinoise(x)

        # Anti-noise at error mic (through secondary path)
        # Simplified: assume instantaneous for this test
        y_filtered = y

        # Error signal
        e = d + y_filtered
        error_signal[n] = e

        # Update controller
        x_filtered = controller.filter_reference(x)
        controller.update_weights(e)

        # Track MSE
        mse_history.append(e**2)

    # Compute metrics
    nr_db = noise_reduction_db(desired, error_signal)
    conv_time = convergence_time(mse_history)
    stability = stability_score(error_signal)

    return {
        'scenario': scenario,
        'variation': variation,
        'step_size': step_size,
        'noise_reduction_db': nr_db,
        'convergence_time': conv_time,
        'stability_score': stability,
        'adaptive': use_adaptive
    }


def main():
    print("=" * 70)
    print("Phase 1 Evaluation: Binary Step Size Selector vs Baseline")
    print("=" * 70)

    # Load binary model
    model_path = Path('output/models/phase1/step_selector_binary.pt')
    print(f"\nLoading binary model from: {model_path}")
    model = BinaryStepSizeSelector.load(model_path)
    print("Model loaded successfully")

    # Run experiments
    baseline_results = []
    adaptive_results = []

    print("\n" + "=" * 70)
    print("RUNNING SIMULATIONS")
    print("=" * 70)

    for scenario in SCENARIOS:
        print(f"\nScenario: {scenario}")

        for variation in range(N_VARIATIONS):
            print(f"  Variation {variation + 1}/{N_VARIATIONS}...", end=" ")

            # Baseline (fixed μ=0.005)
            baseline_res = simulate_anc(
                scenario, variation, BASELINE_STEP_SIZE, use_adaptive=False
            )
            baseline_results.append(baseline_res)

            # Adaptive (binary model)
            adaptive_res = simulate_anc(
                scenario, variation, step_size=0.0,  # will be overridden
                use_adaptive=True, model=model
            )
            adaptive_results.append(adaptive_res)

            print(f"Baseline NR: {baseline_res['noise_reduction_db']:.2f} dB, "
                  f"Adaptive NR: {adaptive_res['noise_reduction_db']:.2f} dB "
                  f"(μ={adaptive_res['step_size']:.4f})")

    # Compute statistics
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    baseline_nr = np.array([r['noise_reduction_db'] for r in baseline_results])
    adaptive_nr = np.array([r['noise_reduction_db'] for r in adaptive_results])
    adaptive_mu = np.array([r['step_size'] for r in adaptive_results])

    baseline_stable = np.array([r['stability_score'] for r in baseline_results])
    adaptive_stable = np.array([r['stability_score'] for r in adaptive_results])

    baseline_conv = np.array([r['convergence_time'] for r in baseline_results])
    adaptive_conv = np.array([r['convergence_time'] for r in adaptive_results])

    print(f"\nBaseline (fixed μ={BASELINE_STEP_SIZE}):")
    print(f"  Mean NR: {np.mean(baseline_nr):.2f} ± {np.std(baseline_nr):.2f} dB")
    print(f"  Stability: {np.mean(baseline_stable):.1%}")
    print(f"  Mean convergence: {np.mean(baseline_conv):.0f} samples")

    print(f"\nAdaptive (binary model):")
    print(f"  Mean NR: {np.mean(adaptive_nr):.2f} ± {np.std(adaptive_nr):.2f} dB")
    print(f"  Stability: {np.mean(adaptive_stable):.1%}")
    print(f"  Mean convergence: {np.mean(adaptive_conv):.0f} samples")
    print(f"  Step size range: {np.min(adaptive_mu):.4f} - {np.max(adaptive_mu):.4f}")
    print(f"  Step size mean: {np.mean(adaptive_mu):.4f} ± {np.std(adaptive_mu):.4f}")

    # Per-scenario breakdown
    print("\n" + "=" * 70)
    print("PER-SCENARIO RESULTS")
    print("=" * 70)

    for scenario in SCENARIOS:
        baseline_scenario = [r for r in baseline_results if r['scenario'] == scenario]
        adaptive_scenario = [r for r in adaptive_results if r['scenario'] == scenario]

        baseline_nr_scenario = np.array([r['noise_reduction_db'] for r in baseline_scenario])
        adaptive_nr_scenario = np.array([r['noise_reduction_db'] for r in adaptive_scenario])
        adaptive_mu_scenario = np.array([r['step_size'] for r in adaptive_scenario])

        print(f"\n{scenario.upper()}:")
        print(f"  Baseline NR: {np.mean(baseline_nr_scenario):.2f} dB")
        print(f"  Adaptive NR: {np.mean(adaptive_nr_scenario):.2f} dB")
        print(f"  Adaptive μ: {np.mean(adaptive_mu_scenario):.4f}")

    # Statistical comparison
    improvement = adaptive_nr - baseline_nr

    t_stat, p_value = stats.ttest_rel(adaptive_nr, baseline_nr)
    mean_improvement = np.mean(improvement)
    std_improvement = np.std(improvement)

    # Cohen's d effect size
    cohens_d = mean_improvement / (std_improvement + 1e-10)

    # Win rate
    win_rate = np.mean(adaptive_nr > baseline_nr)

    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON")
    print("=" * 70)
    print(f"Mean improvement: {mean_improvement:.3f} ± {std_improvement:.3f} dB")
    print(f"Paired t-test p-value: {p_value:.4f}")
    print(f"Cohen's d: {cohens_d:.3f}")
    print(f"Win rate: {win_rate:.1%} ({np.sum(adaptive_nr > baseline_nr)}/{len(adaptive_nr)})")

    # Phase 1 Success Criteria
    print("\n" + "=" * 70)
    print("PHASE 1 SUCCESS CRITERIA")
    print("=" * 70)

    criteria = {
        'mean_improvement_db': {
            'value': mean_improvement,
            'target': 1.0,
            'passed': mean_improvement >= 1.0
        },
        'worst_case_drop_db': {
            'value': np.min(improvement),
            'target': -0.5,
            'passed': np.min(improvement) >= -0.5
        },
        'stability_rate': {
            'value': np.mean(adaptive_stable),
            'target': 0.99,
            'passed': np.mean(adaptive_stable) >= 0.99
        },
        'convergence_speedup': {
            'value': np.mean(baseline_conv) / (np.mean(adaptive_conv) + 1e-10),
            'target': 1.1,
            'passed': np.mean(baseline_conv) / (np.mean(adaptive_conv) + 1e-10) >= 1.1
        }
    }

    passed_count = 0
    for name, criterion in criteria.items():
        status = "✓ PASS" if criterion['passed'] else "✗ FAIL"
        print(f"{name:25s}: {criterion['value']:8.3f} (target: {criterion['target']:6.2f}) {status}")
        if criterion['passed']:
            passed_count += 1

    phase1_passed = passed_count >= 3  # Need 3/4 to pass

    print(f"\nPhase 1 Status: {'✓ PASSED' if phase1_passed else '✗ FAILED'} ({passed_count}/4 criteria)")

    # Save results
    output_path = Path('output/data/phase1/evaluation_results_binary.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'timestamp': str(np.datetime64('now')),
        'baseline_step_size': BASELINE_STEP_SIZE,
        'comparison': {
            'significant': p_value < 0.05 and mean_improvement > 0,
            'p_value': p_value,
            'mean_improvement': mean_improvement,
            'std_improvement': std_improvement,
            'cohens_d': cohens_d,
            'win_rate': win_rate,
            'n_samples': len(adaptive_nr)
        },
        'per_scenario': {},
        'phase1_criteria': criteria,
        'phase1_passed': phase1_passed
    }

    for scenario in SCENARIOS:
        baseline_scenario = [r for r in baseline_results if r['scenario'] == scenario]
        adaptive_scenario = [r for r in adaptive_results if r['scenario'] == scenario]

        baseline_nr_scenario = np.mean([r['noise_reduction_db'] for r in baseline_scenario])
        adaptive_nr_scenario = np.mean([r['noise_reduction_db'] for r in adaptive_scenario])
        adaptive_mu_scenario = np.mean([r['step_size'] for r in adaptive_scenario])

        results['per_scenario'][scenario] = {
            'baseline_mean_nr': baseline_nr_scenario,
            'adaptive_mean_nr': adaptive_nr_scenario,
            'adaptive_mean_mu': adaptive_mu_scenario
        }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    if phase1_passed:
        print("\n" + "=" * 70)
        print("SUCCESS! Phase 1 adaptive step size selector is working.")
        print("The binary model successfully adapts μ to different scenarios.")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("Phase 1 incomplete. Need to improve:")
        for name, criterion in criteria.items():
            if not criterion['passed']:
                print(f"  - {name}: {criterion['value']:.3f} < {criterion['target']:.3f}")
        print("=" * 70)


if __name__ == '__main__':
    main()
