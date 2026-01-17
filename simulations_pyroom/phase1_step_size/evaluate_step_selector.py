"""
Phase 1: Evaluate Step Size Selector

Compares adaptive step size selection against fixed baseline on test scenarios.

Input: output/models/phase1/step_selector.pt
Output: output/plots/phase1/comparison_*.png
"""

import numpy as np
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pyroomacoustics as pra
from src.core.fxlms import FxNLMS
from src.acoustic.path_generator import FIRPath
from src.noise.noise_mixer import NoiseMixer
from src.ml.phase1_step_size.adaptive_fxlms import AdaptiveFxNLMS
from src.ml.phase1_step_size.feature_extractor import extract_features
from src.ml.common.metrics import noise_reduction_db, convergence_time, stability_score
from src.ml.common.comparison import is_significant_improvement, format_comparison_report


# =============================================================================
# Configuration (same as collect_training_data.py)
# =============================================================================

ROOM_DIMS = [4.5, 1.85, 1.2]
ROOM_MATERIALS = {
    'ceiling': 0.38,
    'floor': 0.52,
    'east': 0.14,
    'west': 0.14,
    'north': 0.20,
    'south': 0.30,
}

NOISE_SOURCE_POS = [0.3, 0.92, 0.4]
REF_MIC_POS = [0.3, 0.92, 0.5]
SPEAKER_POS = [1.9, 0.55, 1.0]
ERROR_MIC_POS = [1.8, 0.55, 1.0]

BASELINE_STEP_SIZE = 0.005
FS = 16000
DURATION = 3.0
FILTER_LENGTH = 256

# Test scenarios (use different seeds than training)
TEST_SCENARIOS = ['idle', 'city', 'highway']
N_TEST_VARIATIONS = 5
TEST_SEED_OFFSET = 1000  # Different from training


def create_room_simulation(fs: int = FS) -> dict:
    """Create room and compute acoustic paths."""
    materials = {
        name: pra.Material(coef)
        for name, coef in ROOM_MATERIALS.items()
    }

    room = pra.ShoeBox(
        ROOM_DIMS,
        fs=fs,
        materials=materials,
        max_order=3,
        air_absorption=True
    )

    room.add_source(NOISE_SOURCE_POS)
    room.add_source(SPEAKER_POS)

    mic_array = np.array([REF_MIC_POS, ERROR_MIC_POS]).T
    room.add_microphone_array(pra.MicrophoneArray(mic_array, fs=fs))

    room.compute_rir()
    max_len = 512

    return {
        'primary': room.rir[1][0][:max_len],
        'reference': room.rir[0][0][:max_len],
        'secondary': room.rir[1][1][:max_len],
    }


def run_baseline_simulation(noise_signal, paths, step_size=BASELINE_STEP_SIZE):
    """Run baseline FxNLMS with fixed step size."""
    n_samples = len(noise_signal)

    primary_path = FIRPath(paths['primary'])
    reference_path = FIRPath(paths['reference'])
    secondary_path = FIRPath(paths['secondary'])

    s_hat = paths['secondary'] * (1 + 0.05 * np.random.randn(len(paths['secondary'])))

    fxnlms = FxNLMS(
        filter_length=FILTER_LENGTH,
        step_size=step_size,
        secondary_path_estimate=s_hat,
        regularization=1e-6
    )

    desired = np.zeros(n_samples)
    error = np.zeros(n_samples)

    for i in range(n_samples):
        sample = noise_signal[i]
        x = reference_path.filter_sample(sample)
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
        'desired': desired,
        'error': error,
        'mse_history': fxnlms.mse_history,
    }


def run_adaptive_simulation(noise_signal, paths, model_path):
    """Run adaptive FxNLMS with ML step size selection."""
    n_samples = len(noise_signal)

    primary_path = FIRPath(paths['primary'])
    reference_path = FIRPath(paths['reference'])
    secondary_path = FIRPath(paths['secondary'])

    s_hat = paths['secondary'] * (1 + 0.05 * np.random.randn(len(paths['secondary'])))

    # Create adaptive filter
    adaptive_fxnlms = AdaptiveFxNLMS(
        filter_length=FILTER_LENGTH,
        secondary_path_estimate=s_hat,
        model_path=model_path,
        regularization=1e-6,
        fs=FS
    )

    # Initialize with first second of reference-filtered noise
    # We need to get reference signal for feature extraction
    ref_signal = np.zeros(FS)
    temp_ref_path = FIRPath(paths['reference'])
    for i in range(min(FS, n_samples)):
        ref_signal[i] = temp_ref_path.filter_sample(noise_signal[i])

    selected_mu = adaptive_fxnlms.initialize(ref_signal)

    desired = np.zeros(n_samples)
    error = np.zeros(n_samples)

    for i in range(n_samples):
        sample = noise_signal[i]
        x = reference_path.filter_sample(sample)
        d = primary_path.filter_sample(sample)
        desired[i] = d

        y = adaptive_fxnlms.generate_antinoise(x)
        y_at_error = secondary_path.filter_sample(y)
        e = d + y_at_error
        error[i] = e

        adaptive_fxnlms.filter_reference(x)
        adaptive_fxnlms.update_weights(e)

    return {
        'noise_reduction_db': noise_reduction_db(desired, error),
        'convergence_time': convergence_time(adaptive_fxnlms.mse_history),
        'stability_score': stability_score(adaptive_fxnlms.mse_history),
        'step_size': selected_mu,
        'desired': desired,
        'error': error,
        'mse_history': adaptive_fxnlms.mse_history,
    }


def plot_comparison_bars(baseline_results, adaptive_results, output_path):
    """Plot bar chart comparing noise reduction."""
    scenarios = list(baseline_results.keys())
    n_scenarios = len(scenarios)

    baseline_nr = [np.mean([r['noise_reduction_db'] for r in baseline_results[s]]) for s in scenarios]
    adaptive_nr = [np.mean([r['noise_reduction_db'] for r in adaptive_results[s]]) for s in scenarios]

    baseline_std = [np.std([r['noise_reduction_db'] for r in baseline_results[s]]) for s in scenarios]
    adaptive_std = [np.std([r['noise_reduction_db'] for r in adaptive_results[s]]) for s in scenarios]

    x = np.arange(n_scenarios)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, baseline_nr, width, yerr=baseline_std,
                   label=f'Baseline (μ={BASELINE_STEP_SIZE})', color='#3498db', capsize=5)
    bars2 = ax.bar(x + width/2, adaptive_nr, width, yerr=adaptive_std,
                   label='Adaptive (ML)', color='#2ecc71', capsize=5)

    ax.set_xlabel('Scenario')
    ax.set_ylabel('Noise Reduction (dB)')
    ax.set_title('Baseline vs Adaptive Step Size: Noise Reduction')
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in scenarios])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_step_size_distribution(adaptive_results, output_path):
    """Plot distribution of selected step sizes."""
    fig, axes = plt.subplots(1, len(adaptive_results), figsize=(14, 4))

    for idx, (scenario, results) in enumerate(adaptive_results.items()):
        ax = axes[idx]
        step_sizes = [r['step_size'] for r in results]

        ax.hist(step_sizes, bins=15, edgecolor='black', alpha=0.7, color='#2ecc71')
        ax.axvline(x=BASELINE_STEP_SIZE, color='red', linestyle='--',
                   label=f'Baseline ({BASELINE_STEP_SIZE})')
        ax.axvline(x=np.mean(step_sizes), color='blue', linestyle='-',
                   label=f'Mean ({np.mean(step_sizes):.4f})')

        ax.set_xlabel('Step Size (μ)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{scenario.capitalize()}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Selected Step Sizes by Scenario', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_scatter_comparison(baseline_results, adaptive_results, output_path):
    """Plot scatter of baseline vs adaptive noise reduction."""
    baseline_all = []
    adaptive_all = []
    colors = []
    color_map = {'idle': '#3498db', 'city': '#e74c3c', 'highway': '#2ecc71'}

    for scenario in baseline_results:
        for b, a in zip(baseline_results[scenario], adaptive_results[scenario]):
            baseline_all.append(b['noise_reduction_db'])
            adaptive_all.append(a['noise_reduction_db'])
            colors.append(color_map.get(scenario, '#999'))

    baseline_all = np.array(baseline_all)
    adaptive_all = np.array(adaptive_all)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(baseline_all, adaptive_all, c=colors, alpha=0.7, edgecolors='k', linewidth=0.5, s=60)

    # Diagonal line (equal performance)
    min_val = min(baseline_all.min(), adaptive_all.min())
    max_val = max(baseline_all.max(), adaptive_all.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Equal')

    ax.set_xlabel('Baseline NR (dB)')
    ax.set_ylabel('Adaptive NR (dB)')
    ax.set_title('Adaptive vs Baseline Noise Reduction\n(Points above diagonal = ML wins)')

    # Legend for scenarios
    for scenario, color in color_map.items():
        ax.scatter([], [], c=color, label=scenario.capitalize(), s=60)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Main evaluation script."""
    print("=" * 70)
    print("Phase 1: Evaluating Step Size Selector")
    print("=" * 70)

    model_path = Path('output/models/phase1/step_selector.pt')
    plot_dir = Path('output/plots/phase1')
    data_dir = Path('output/data/phase1')

    plot_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check model exists
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Run train_step_selector.py first.")
        return

    # Create room
    print("\nCreating room simulation...")
    paths = create_room_simulation()

    # Initialize noise generator
    noise_gen = NoiseMixer(FS)

    # Storage
    baseline_results = {s: [] for s in TEST_SCENARIOS}
    adaptive_results = {s: [] for s in TEST_SCENARIOS}

    total_tests = len(TEST_SCENARIOS) * N_TEST_VARIATIONS
    test_count = 0

    print("\nRunning evaluation...")

    for scenario in TEST_SCENARIOS:
        print(f"\n  Scenario: {scenario}")

        for var in range(N_TEST_VARIATIONS):
            test_count += 1
            print(f"\r    [{test_count}/{total_tests}] Variation {var+1}/{N_TEST_VARIATIONS}", end="")

            np.random.seed(TEST_SEED_OFFSET + var)
            noise_signal = noise_gen.generate_scenario(DURATION, scenario)

            # Run baseline
            baseline_result = run_baseline_simulation(noise_signal, paths)
            baseline_results[scenario].append(baseline_result)

            # Run adaptive
            adaptive_result = run_adaptive_simulation(noise_signal, paths, model_path)
            adaptive_results[scenario].append(adaptive_result)

    print("\n")

    # Compute statistics
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    all_baseline_nr = []
    all_adaptive_nr = []

    for scenario in TEST_SCENARIOS:
        baseline_nr = [r['noise_reduction_db'] for r in baseline_results[scenario]]
        adaptive_nr = [r['noise_reduction_db'] for r in adaptive_results[scenario]]
        adaptive_mu = [r['step_size'] for r in adaptive_results[scenario]]

        all_baseline_nr.extend(baseline_nr)
        all_adaptive_nr.extend(adaptive_nr)

        print(f"\n{scenario.upper()}:")
        print(f"  Baseline NR: {np.mean(baseline_nr):.2f} ± {np.std(baseline_nr):.2f} dB")
        print(f"  Adaptive NR: {np.mean(adaptive_nr):.2f} ± {np.std(adaptive_nr):.2f} dB")
        print(f"  Improvement: {np.mean(adaptive_nr) - np.mean(baseline_nr):+.2f} dB")
        print(f"  Selected μ:  {np.mean(adaptive_mu):.5f} ± {np.std(adaptive_mu):.5f}")

    # Statistical comparison
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    comparison = is_significant_improvement(all_baseline_nr, all_adaptive_nr)
    print(f"\nMean Improvement: {comparison['mean_improvement']:.3f} dB")
    print(f"Effect Size (Cohen's d): {comparison['cohens_d']:.3f}")
    print(f"Win Rate: {comparison['win_rate']*100:.1f}%")
    print(f"p-value: {comparison['p_value']:.4f}")
    print(f"Statistically Significant: {'Yes' if comparison['significant'] else 'No'}")

    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    plot_comparison_bars(baseline_results, adaptive_results, plot_dir / 'comparison_bars.png')
    plot_step_size_distribution(adaptive_results, plot_dir / 'step_size_distribution.png')
    plot_scatter_comparison(baseline_results, adaptive_results, plot_dir / 'comparison_scatter.png')

    # Save results
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'baseline_step_size': BASELINE_STEP_SIZE,
        'comparison': comparison,
        'per_scenario': {
            scenario: {
                'baseline_mean_nr': float(np.mean([r['noise_reduction_db'] for r in baseline_results[scenario]])),
                'adaptive_mean_nr': float(np.mean([r['noise_reduction_db'] for r in adaptive_results[scenario]])),
                'adaptive_mean_mu': float(np.mean([r['step_size'] for r in adaptive_results[scenario]])),
            }
            for scenario in TEST_SCENARIOS
        }
    }

    results_path = data_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nSaved results to {results_path}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
