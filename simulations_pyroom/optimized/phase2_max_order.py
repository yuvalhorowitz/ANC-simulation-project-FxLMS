"""
Phase 2: max_order Optimization

Tests the impact of pyroomacoustics max_order parameter on ANC performance.

Current Issue:
    max_order = 3-4 may be too low for realistic room reflections.

Hypothesis:
    Higher max_order captures more late reflections, providing better
    acoustic modeling and potentially better ANC performance.

Expected Impact: +1-2 dB
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pyroomacoustics as pra
from src.acoustic.path_generator import AcousticPathGenerator, FIRPath
from src.core.fxlms import FxNLMS
from src.noise.noise_mixer import NoiseMixer
from configurations import STEP7_CONFIGS


def calculate_optimal_max_order(dimensions: list, rt60: float, c: float = 343.0) -> int:
    """
    Calculate optimal max_order based on room dimensions and RT60.

    Each reflection order represents sound bouncing off walls.
    We need enough orders to cover the reverb time.
    """
    max_dim = max(dimensions)
    if max_dim <= 0 or rt60 <= 0:
        return 3

    # Time for one reflection across max dimension
    travel_time = max_dim / c

    # Number of reflections to cover RT60
    n_reflections = rt60 / travel_time

    # Each order adds roughly 2 reflections
    optimal = int(np.ceil(n_reflections / 2))

    return max(3, min(optimal, 12))


def estimate_rt60(dimensions: list, absorption: float) -> float:
    """Estimate RT60 using Sabine's formula."""
    L, W, H = dimensions
    V = L * W * H
    S = 2 * (L * W + L * H + W * H)
    if absorption <= 0:
        return 0.5
    rt60 = 0.161 * V / (absorption * S)
    return max(0.05, min(rt60, 1.0))


class Phase2ANC:
    """ANC with configurable max_order."""

    def __init__(self, config: dict, max_order: int, fs: int = 16000):
        self.fs = fs
        self.config = config
        self.max_order = max_order

        room_cfg = config['room']
        pos = config['positions']
        fxlms_cfg = config['fxlms']

        # Create room with specified max_order
        pra_materials = {
            'ceiling': pra.Material(room_cfg['materials']['ceiling']),
            'floor': pra.Material(room_cfg['materials']['floor']),
            'east': pra.Material(room_cfg['materials']['east']),
            'west': pra.Material(room_cfg['materials']['west']),
            'north': pra.Material(room_cfg['materials']['north']),
            'south': pra.Material(room_cfg['materials']['south']),
        }

        self.room = pra.ShoeBox(
            room_cfg['dimensions'],
            fs=fs,
            materials=pra_materials,
            max_order=max_order,  # KEY: Use specified max_order
            air_absorption=True
        )

        self.room.add_source(pos['noise_source'])
        self.room.add_source(pos['speaker'])
        mic_array = np.array([pos['reference_mic'], pos['error_mic']]).T
        self.room.add_microphone_array(pra.MicrophoneArray(mic_array, fs=fs))

        self.room.compute_rir()

        # Extract paths (keep same truncation as original)
        path_gen = AcousticPathGenerator(self.room)
        paths = path_gen.get_all_anc_paths(modeling_error=0.05)

        max_len = 512
        self.H_primary = paths['primary'][:max_len]
        self.H_secondary = paths['secondary'][:max_len]
        self.H_secondary_est = paths['secondary_estimate'][:max_len]
        self.H_reference = paths['reference'][:max_len]

        self.primary_path = FIRPath(self.H_primary)
        self.secondary_path = FIRPath(self.H_secondary)
        self.reference_path = FIRPath(self.H_reference)

        self.fxlms = FxNLMS(
            filter_length=fxlms_cfg['filter_length'],
            step_size=fxlms_cfg['step_size'],
            secondary_path_estimate=self.H_secondary_est,
            regularization=1e-4
        )

        self.noise_gen = NoiseMixer(fs)

    def run(self, duration: float, scenario: str) -> dict:
        """Run simulation."""
        noise_source = self.noise_gen.generate_scenario(duration, scenario)
        n_samples = len(noise_source)

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

        steady_start = len(desired) // 2
        d_power = np.mean(np.array(desired[steady_start:])**2)
        e_power = np.mean(np.array(error[steady_start:])**2)

        nr_db = 10 * np.log10(d_power / e_power) if e_power > 1e-10 else 60.0

        return {
            'desired': np.array(desired),
            'error': np.array(error),
            'mse': np.array(mse),
            'noise_reduction_db': nr_db,
        }


def run_phase2_comparison(config_key: str, config: dict, duration: float = 5.0):
    """Run Phase 2 max_order comparison."""
    room_cfg = config['room']
    scenario = config.get('scenario', 'highway')

    # Baseline max_order
    baseline_order = room_cfg['max_order']

    # Calculate optimal max_order
    materials = room_cfg['materials']
    avg_absorption = np.mean(list(materials.values()))
    rt60_est = estimate_rt60(room_cfg['dimensions'], avg_absorption)
    optimal_order = calculate_optimal_max_order(room_cfg['dimensions'], rt60_est)

    print(f"\n{'='*60}")
    print(f"Configuration: {config['name']}")
    print(f"{'='*60}")
    print(f"Room: {room_cfg['dimensions']} m")
    print(f"Estimated RT60: {rt60_est*1000:.0f} ms")
    print(f"Scenario: {scenario}")
    print(f"\nBaseline max_order: {baseline_order}")
    print(f"Optimal max_order:  {optimal_order}")

    # Run baseline
    print(f"\nRunning baseline (max_order={baseline_order})...")
    anc_baseline = Phase2ANC(config, baseline_order)
    results_baseline = anc_baseline.run(duration, scenario)
    print(f"  NR: {results_baseline['noise_reduction_db']:.1f} dB")

    # Run optimized
    print(f"\nRunning optimized (max_order={optimal_order})...")
    anc_optimized = Phase2ANC(config, optimal_order)
    results_optimized = anc_optimized.run(duration, scenario)
    print(f"  NR: {results_optimized['noise_reduction_db']:.1f} dB")

    improvement = results_optimized['noise_reduction_db'] - results_baseline['noise_reduction_db']
    print(f"\n>>> IMPROVEMENT: {improvement:+.1f} dB")

    return {
        'config_name': config['name'],
        'scenario': scenario,
        'rt60_est': rt60_est,
        'baseline_order': baseline_order,
        'optimal_order': optimal_order,
        'baseline_nr': results_baseline['noise_reduction_db'],
        'optimized_nr': results_optimized['noise_reduction_db'],
        'improvement': improvement,
        'baseline_mse': results_baseline['mse'],
        'optimized_mse': results_optimized['mse'],
    }


def plot_results(all_results: list, save_path: str = None):
    """Generate comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Phase 2: max_order Optimization', fontsize=14, fontweight='bold')

    configs = [r['config_name'] for r in all_results]
    baseline_nrs = [r['baseline_nr'] for r in all_results]
    optimized_nrs = [r['optimized_nr'] for r in all_results]

    x = np.arange(len(configs))
    width = 0.35

    # NR comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x - width/2, baseline_nrs, width, label='Baseline', color='steelblue')
    bars2 = ax1.bar(x + width/2, optimized_nrs, width, label='Optimized', color='forestgreen')
    ax1.set_ylabel('Noise Reduction (dB)')
    ax1.set_title('Noise Reduction Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    for bar in bars1 + bars2:
        h = bar.get_height()
        ax1.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    # Improvement
    ax2 = axes[0, 1]
    improvements = [r['improvement'] for r in all_results]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax2.bar(configs, improvements, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Improvement (dB)')
    ax2.set_title('max_order Optimization Impact')
    ax2.grid(True, alpha=0.3, axis='y')

    for i, imp in enumerate(improvements):
        ax2.annotate(f'{imp:+.1f}', xy=(i, imp),
                    xytext=(0, 5 if imp > 0 else -15),
                    textcoords='offset points', ha='center', fontsize=10, fontweight='bold')

    # max_order comparison
    ax3 = axes[1, 0]
    baseline_orders = [r['baseline_order'] for r in all_results]
    optimal_orders = [r['optimal_order'] for r in all_results]

    ax3.bar(x - width/2, baseline_orders, width, label='Baseline', color='steelblue')
    ax3.bar(x + width/2, optimal_orders, width, label='Optimized', color='forestgreen')
    ax3.set_ylabel('max_order')
    ax3.set_title('Reflection Order: Baseline vs Optimized')
    ax3.set_xticks(x)
    ax3.set_xticklabels(configs)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Convergence
    ax4 = axes[1, 1]
    r = all_results[0]
    window = 200
    mse_b = np.convolve(r['baseline_mse'], np.ones(window)/window, mode='valid')
    mse_o = np.convolve(r['optimized_mse'], np.ones(window)/window, mode='valid')
    t = np.arange(len(mse_b)) / 16000
    ax4.semilogy(t, mse_b, 'b-', label=f"Baseline (order={r['baseline_order']})", alpha=0.7)
    ax4.semilogy(t, mse_o, 'g-', label=f"Optimized (order={r['optimal_order']})", alpha=0.7)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('MSE')
    ax4.set_title(f"Convergence: {r['config_name']}")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {save_path}")
    plt.close()


def main():
    print("=" * 70)
    print("Phase 2: max_order Optimization")
    print("=" * 70)

    output_dir = 'output/phase2_max_order'
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    for config_key, config in STEP7_CONFIGS.items():
        results = run_phase2_comparison(config_key, config, duration=5.0)
        all_results.append(results)

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 2 RESULTS: max_order OPTIMIZATION")
    print("=" * 70)
    print(f"\n{'Config':<15} {'Baseline':<10} {'Optimized':<10} {'Improvement':<12} {'Order':<12}")
    print("-" * 60)

    total_baseline = 0
    total_optimized = 0

    for r in all_results:
        print(f"{r['config_name']:<15} {r['baseline_nr']:>7.1f} dB {r['optimized_nr']:>7.1f} dB "
              f"{r['improvement']:>+9.1f} dB  {r['baseline_order']}->{r['optimal_order']}")
        total_baseline += r['baseline_nr']
        total_optimized += r['optimized_nr']

    n = len(all_results)
    avg_baseline = total_baseline / n
    avg_optimized = total_optimized / n
    avg_improvement = avg_optimized - avg_baseline

    print("-" * 60)
    print(f"{'AVERAGE':<15} {avg_baseline:>7.1f} dB {avg_optimized:>7.1f} dB {avg_improvement:>+9.1f} dB")
    print("=" * 70)

    plot_results(all_results, f'{output_dir}/phase2_comparison.png')

    # Save results
    with open(f'{output_dir}/phase2_results.txt', 'w') as f:
        f.write(f"Phase 2: max_order Optimization Results\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 60 + "\n\n")
        for r in all_results:
            f.write(f"{r['config_name']}: {r['baseline_nr']:.1f} -> {r['optimized_nr']:.1f} dB "
                   f"({r['improvement']:+.1f} dB), order {r['baseline_order']}->{r['optimal_order']}\n")
        f.write(f"\nAverage improvement: {avg_improvement:+.1f} dB\n")

    print(f"\nResults saved to: {output_dir}/")

    # Verdict
    print("\n" + "=" * 70)
    if avg_improvement >= 1.0:
        print(f"SUCCESS: Phase 2 provides +{avg_improvement:.1f} dB improvement")
        print("         Recommend implementing max_order optimization")
    elif avg_improvement > 0:
        print(f"MODEST: Phase 2 provides +{avg_improvement:.1f} dB improvement")
    else:
        print(f"NO IMPROVEMENT: Phase 2 shows {avg_improvement:.1f} dB change")
        print("         Skip this optimization")
    print("=" * 70)

    return all_results


if __name__ == '__main__':
    main()
