"""
Phase 1: Filter Length Optimization Comparison

Compares the original step7_car_interior.py (filter_length=256)
against optimized filter length based on estimated RT60.

Key Hypothesis:
    Current filter_length=256 (16ms @ 16kHz) is too short.
    Car cabin RT60 is ~100-200ms, so filter should be ~512-2048 taps.

This script runs A/B comparison without changing anything else.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pyroomacoustics as pra
from src.acoustic.path_generator import AcousticPathGenerator, FIRPath
from src.core.fxlms import FxNLMS
from src.noise.noise_mixer import NoiseMixer
from configurations import STEP7_CONFIGS, print_config_summary


def estimate_rt60_from_room(dimensions: list, absorption: float) -> float:
    """
    Estimate RT60 using Sabine's formula.

    RT60 = 0.161 * V / (A * S)
    """
    L, W, H = dimensions
    V = L * W * H
    S = 2 * (L * W + L * H + W * H)

    if absorption <= 0:
        return 0.5

    rt60 = 0.161 * V / (absorption * S)
    return max(0.05, min(rt60, 1.0))


def calculate_optimal_filter_length(rt60: float, fs: int = 16000) -> int:
    """
    Calculate optimal filter length based on RT60.

    Rule: Filter should cover ~1.5x RT60 for good convergence.
    """
    required_time = rt60 * 1.5
    required_samples = int(required_time * fs)

    # Round to power of 2
    if required_samples <= 0:
        return 256

    power = int(np.ceil(np.log2(required_samples)))
    filter_length = 2 ** power

    # Bounds
    return max(256, min(filter_length, 2048))


class Phase1ANC:
    """
    ANC system for Phase 1 comparison.

    Same as CarInteriorANC but with configurable filter length.
    """

    def __init__(self, config: dict, filter_length: int, fs: int = 16000):
        self.fs = fs
        self.config = config
        self.filter_length = filter_length

        room_cfg = config['room']
        pos = config['positions']
        fxlms_cfg = config['fxlms']

        # Create room
        self.room = self._create_room(
            room_cfg['dimensions'],
            room_cfg['materials'],
            room_cfg['max_order'],
            fs
        )

        # Add sources and mics
        self.room.add_source(pos['noise_source'])
        self.room.add_source(pos['speaker'])
        mic_array = np.array([pos['reference_mic'], pos['error_mic']]).T
        self.room.add_microphone_array(pra.MicrophoneArray(mic_array, fs=fs))

        # Compute RIRs
        self.room.compute_rir()

        # Extract paths
        path_gen = AcousticPathGenerator(self.room)
        paths = path_gen.get_all_anc_paths(modeling_error=0.05)

        # Truncate paths (same as original)
        max_len = 512
        self.H_primary = paths['primary'][:max_len]
        self.H_secondary = paths['secondary'][:max_len]
        self.H_secondary_est = paths['secondary_estimate'][:max_len]
        self.H_reference = paths['reference'][:max_len]

        # Create FIR filters
        self.primary_path = FIRPath(self.H_primary)
        self.secondary_path = FIRPath(self.H_secondary)
        self.reference_path = FIRPath(self.H_reference)

        # Create FxNLMS with specified filter length
        self.fxlms = FxNLMS(
            filter_length=filter_length,  # KEY DIFFERENCE
            step_size=fxlms_cfg['step_size'],
            secondary_path_estimate=self.H_secondary_est,
            regularization=1e-4
        )

        # Noise generator
        self.noise_gen = NoiseMixer(fs)

    def _create_room(self, dimensions, materials, max_order, fs):
        pra_materials = {
            'ceiling': pra.Material(materials['ceiling']),
            'floor': pra.Material(materials['floor']),
            'east': pra.Material(materials['east']),
            'west': pra.Material(materials['west']),
            'north': pra.Material(materials['north']),
            'south': pra.Material(materials['south']),
        }

        return pra.ShoeBox(
            dimensions,
            fs=fs,
            materials=pra_materials,
            max_order=max_order,
            air_absorption=True
        )

    def run(self, duration: float, scenario: str) -> dict:
        """Run simulation and return results."""
        noise_source = self.noise_gen.generate_scenario(duration, scenario)
        n_samples = len(noise_source)

        # Reset
        self.fxlms.reset()
        self.primary_path.reset()
        self.secondary_path.reset()
        self.reference_path.reset()

        desired = []
        error = []
        mse = []

        for i in range(n_samples):
            sample = noise_source[i]

            x = self.reference_path.filter_sample(sample)
            d = self.primary_path.filter_sample(sample)
            desired.append(d)

            y = self.fxlms.generate_antinoise(x)
            y_at_error = self.secondary_path.filter_sample(y)

            e = d + y_at_error
            error.append(e)
            mse.append(e ** 2)

            self.fxlms.filter_reference(x)
            self.fxlms.update_weights(e)

        # Calculate noise reduction (steady state)
        steady_start = len(desired) // 2
        d_power = np.mean(np.array(desired[steady_start:])**2)
        e_power = np.mean(np.array(error[steady_start:])**2)

        if e_power > 1e-10:
            nr_db = 10 * np.log10(d_power / e_power)
        else:
            nr_db = 60.0

        return {
            'desired': np.array(desired),
            'error': np.array(error),
            'mse': np.array(mse),
            'noise_reduction_db': nr_db,
            'filter_length': self.filter_length,
        }


def run_comparison(config_key: str, config: dict, duration: float = 5.0):
    """
    Run baseline vs optimized comparison for a single configuration.
    """
    room_cfg = config['room']
    scenario = config.get('scenario', 'highway')

    # Estimate RT60 from average material absorption
    materials = room_cfg['materials']
    avg_absorption = np.mean(list(materials.values()))
    rt60_est = estimate_rt60_from_room(room_cfg['dimensions'], avg_absorption)

    # Calculate optimal filter length
    baseline_length = config['fxlms']['filter_length']  # Original: 256
    optimal_length = calculate_optimal_filter_length(rt60_est)

    print(f"\n{'='*60}")
    print(f"Configuration: {config['name']}")
    print(f"{'='*60}")
    print(f"Room: {room_cfg['dimensions']} m")
    print(f"Avg absorption: {avg_absorption:.2f}")
    print(f"Estimated RT60: {rt60_est*1000:.0f} ms")
    print(f"Baseline filter: {baseline_length} taps ({baseline_length/16000*1000:.1f} ms)")
    print(f"Optimal filter: {optimal_length} taps ({optimal_length/16000*1000:.1f} ms)")
    print(f"Scenario: {scenario}")

    # Run baseline
    print(f"\nRunning baseline (filter_length={baseline_length})...")
    anc_baseline = Phase1ANC(config, filter_length=baseline_length)
    results_baseline = anc_baseline.run(duration, scenario)
    print(f"  Noise Reduction: {results_baseline['noise_reduction_db']:.1f} dB")

    # Run optimized
    print(f"\nRunning optimized (filter_length={optimal_length})...")
    anc_optimized = Phase1ANC(config, filter_length=optimal_length)
    results_optimized = anc_optimized.run(duration, scenario)
    print(f"  Noise Reduction: {results_optimized['noise_reduction_db']:.1f} dB")

    # Calculate improvement
    improvement = results_optimized['noise_reduction_db'] - results_baseline['noise_reduction_db']
    print(f"\n>>> IMPROVEMENT: {improvement:+.1f} dB")

    return {
        'config_key': config_key,
        'config_name': config['name'],
        'scenario': scenario,
        'rt60_est': rt60_est,
        'baseline_length': baseline_length,
        'optimal_length': optimal_length,
        'baseline_nr': results_baseline['noise_reduction_db'],
        'optimized_nr': results_optimized['noise_reduction_db'],
        'improvement': improvement,
        'baseline_mse': results_baseline['mse'],
        'optimized_mse': results_optimized['mse'],
    }


def plot_comparison(all_results: list, save_path: str = None):
    """Generate comparison plot."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Phase 1: Filter Length Optimization Results', fontsize=14, fontweight='bold')

    # Bar chart comparison
    ax1 = axes[0, 0]
    configs = [r['config_name'] for r in all_results]
    baseline_nrs = [r['baseline_nr'] for r in all_results]
    optimized_nrs = [r['optimized_nr'] for r in all_results]

    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax1.bar(x - width/2, baseline_nrs, width, label='Baseline (256 taps)', color='steelblue')
    bars2 = ax1.bar(x + width/2, optimized_nrs, width, label='Optimized', color='darkorange')

    ax1.set_ylabel('Noise Reduction (dB)')
    ax1.set_title('Noise Reduction Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=15, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    # Improvement chart
    ax2 = axes[0, 1]
    improvements = [r['improvement'] for r in all_results]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]

    ax2.bar(configs, improvements, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Improvement (dB)')
    ax2.set_title('Filter Length Optimization Impact')
    ax2.set_xticklabels(configs, rotation=15, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    for i, (imp, cfg) in enumerate(zip(improvements, configs)):
        ax2.annotate(f'{imp:+.1f}', xy=(i, imp),
                    xytext=(0, 5 if imp > 0 else -15),
                    textcoords='offset points', ha='center', fontsize=10, fontweight='bold')

    # Filter length comparison
    ax3 = axes[1, 0]
    baseline_lens = [r['baseline_length'] for r in all_results]
    optimal_lens = [r['optimal_length'] for r in all_results]

    ax3.bar(x - width/2, baseline_lens, width, label='Baseline', color='steelblue')
    ax3.bar(x + width/2, optimal_lens, width, label='Optimized', color='darkorange')
    ax3.set_ylabel('Filter Length (taps)')
    ax3.set_title('Filter Length: Baseline vs Optimized')
    ax3.set_xticks(x)
    ax3.set_xticklabels(configs, rotation=15, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # MSE convergence comparison (first config only as example)
    ax4 = axes[1, 1]
    r = all_results[0]

    window = 200
    mse_baseline = np.convolve(r['baseline_mse'], np.ones(window)/window, mode='valid')
    mse_optimized = np.convolve(r['optimized_mse'], np.ones(window)/window, mode='valid')

    t = np.arange(len(mse_baseline)) / 16000
    ax4.semilogy(t, mse_baseline, 'b-', label=f"Baseline ({r['baseline_length']} taps)", alpha=0.7)
    ax4.semilogy(t, mse_optimized, 'r-', label=f"Optimized ({r['optimal_length']} taps)", alpha=0.7)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('MSE')
    ax4.set_title(f"Convergence: {r['config_name']}")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot: {save_path}")

    plt.close()
    return fig


def main():
    print("=" * 70)
    print("Phase 1: Filter Length Optimization")
    print("Comparing baseline (256 taps) vs optimal filter length")
    print("=" * 70)

    # Create output directory
    output_dir = 'output/phase1_filter_length'
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    # Run comparison for each Step 7 configuration
    for config_key, config in STEP7_CONFIGS.items():
        results = run_comparison(config_key, config, duration=5.0)
        all_results.append(results)

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 1 SUMMARY: FILTER LENGTH OPTIMIZATION")
    print("=" * 70)
    print(f"\n{'Configuration':<20} {'Baseline':<12} {'Optimized':<12} {'Improvement':<12} {'Filter':<15}")
    print("-" * 70)

    total_baseline = 0
    total_optimized = 0

    for r in all_results:
        print(f"{r['config_name']:<20} {r['baseline_nr']:>8.1f} dB {r['optimized_nr']:>8.1f} dB "
              f"{r['improvement']:>+8.1f} dB  {r['baseline_length']}->{r['optimal_length']}")
        total_baseline += r['baseline_nr']
        total_optimized += r['optimized_nr']

    n = len(all_results)
    avg_baseline = total_baseline / n
    avg_optimized = total_optimized / n
    avg_improvement = avg_optimized - avg_baseline

    print("-" * 70)
    print(f"{'AVERAGE':<20} {avg_baseline:>8.1f} dB {avg_optimized:>8.1f} dB {avg_improvement:>+8.1f} dB")
    print("=" * 70)

    # Generate plots
    plot_comparison(all_results, f'{output_dir}/phase1_comparison.png')

    # Save results to file
    results_file = f'{output_dir}/phase1_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"Phase 1: Filter Length Optimization Results\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 60 + "\n\n")

        for r in all_results:
            f.write(f"Configuration: {r['config_name']}\n")
            f.write(f"  Scenario: {r['scenario']}\n")
            f.write(f"  Estimated RT60: {r['rt60_est']*1000:.0f} ms\n")
            f.write(f"  Baseline filter: {r['baseline_length']} taps\n")
            f.write(f"  Optimal filter: {r['optimal_length']} taps\n")
            f.write(f"  Baseline NR: {r['baseline_nr']:.1f} dB\n")
            f.write(f"  Optimized NR: {r['optimized_nr']:.1f} dB\n")
            f.write(f"  Improvement: {r['improvement']:+.1f} dB\n\n")

        f.write(f"\nAverage Improvement: {avg_improvement:+.1f} dB\n")

    print(f"\nResults saved to: {results_file}")
    print(f"Plot saved to: {output_dir}/phase1_comparison.png")

    # Verdict
    print("\n" + "=" * 70)
    if avg_improvement > 0:
        print(f"VERDICT: Filter length optimization provides +{avg_improvement:.1f} dB improvement")
        print("         Recommend proceeding to Phase 2")
    else:
        print(f"VERDICT: Filter length optimization shows {avg_improvement:.1f} dB change")
        print("         May need further investigation")
    print("=" * 70)

    return all_results


if __name__ == '__main__':
    main()
