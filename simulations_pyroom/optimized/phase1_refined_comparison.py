"""
Phase 1 Refined: Filter Length with Adjusted Step Size

The initial Phase 1 test showed mixed results because longer filters
need smaller step sizes to converge properly.

This refined test adjusts step size inversely proportional to filter length:
    optimal_step = baseline_step * (baseline_length / optimal_length)

This ensures the adaptation rate is appropriately scaled.
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
from configurations import STEP7_CONFIGS


def estimate_rt60_from_room(dimensions: list, absorption: float) -> float:
    """Estimate RT60 using Sabine's formula."""
    L, W, H = dimensions
    V = L * W * H
    S = 2 * (L * W + L * H + W * H)
    if absorption <= 0:
        return 0.5
    rt60 = 0.161 * V / (absorption * S)
    return max(0.05, min(rt60, 1.0))


def calculate_optimal_filter_length(rt60: float, fs: int = 16000) -> int:
    """Calculate optimal filter length based on RT60."""
    required_time = rt60 * 1.5
    required_samples = int(required_time * fs)
    if required_samples <= 0:
        return 256
    power = int(np.ceil(np.log2(required_samples)))
    filter_length = 2 ** power
    return max(256, min(filter_length, 2048))


class Phase1RefinedANC:
    """ANC system with configurable filter length and step size."""

    def __init__(self, config: dict, filter_length: int, step_size: float, fs: int = 16000):
        self.fs = fs
        self.config = config
        self.filter_length = filter_length
        self.step_size = step_size

        room_cfg = config['room']
        pos = config['positions']

        # Create room
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
            max_order=room_cfg['max_order'],
            air_absorption=True
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

        max_len = 512
        self.H_primary = paths['primary'][:max_len]
        self.H_secondary = paths['secondary'][:max_len]
        self.H_secondary_est = paths['secondary_estimate'][:max_len]
        self.H_reference = paths['reference'][:max_len]

        self.primary_path = FIRPath(self.H_primary)
        self.secondary_path = FIRPath(self.H_secondary)
        self.reference_path = FIRPath(self.H_reference)

        # Create FxNLMS with specified parameters
        self.fxlms = FxNLMS(
            filter_length=filter_length,
            step_size=step_size,
            secondary_path_estimate=self.H_secondary_est,
            regularization=1e-4
        )

        self.noise_gen = NoiseMixer(fs)

    def run(self, duration: float, scenario: str) -> dict:
        """Run simulation and return results."""
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
            'step_size': self.step_size,
        }


def run_refined_comparison(config_key: str, config: dict, duration: float = 5.0):
    """Run baseline vs optimized with adjusted step size."""
    room_cfg = config['room']
    scenario = config.get('scenario', 'highway')

    materials = room_cfg['materials']
    avg_absorption = np.mean(list(materials.values()))
    rt60_est = estimate_rt60_from_room(room_cfg['dimensions'], avg_absorption)

    # Baseline parameters
    baseline_length = config['fxlms']['filter_length']
    baseline_step = config['fxlms']['step_size']

    # Optimal filter length
    optimal_length = calculate_optimal_filter_length(rt60_est)

    # KEY INSIGHT: Adjust step size proportionally
    # Longer filters need smaller step sizes
    optimal_step = baseline_step * (baseline_length / optimal_length)

    print(f"\n{'='*60}")
    print(f"Configuration: {config['name']}")
    print(f"{'='*60}")
    print(f"Room: {room_cfg['dimensions']} m")
    print(f"Estimated RT60: {rt60_est*1000:.0f} ms")
    print(f"Scenario: {scenario}")
    print(f"\nBaseline:  filter={baseline_length}, step={baseline_step}")
    print(f"Optimized: filter={optimal_length}, step={optimal_step:.6f}")

    # Run baseline
    print(f"\nRunning baseline...")
    anc_baseline = Phase1RefinedANC(config, baseline_length, baseline_step)
    results_baseline = anc_baseline.run(duration, scenario)
    print(f"  NR: {results_baseline['noise_reduction_db']:.1f} dB")

    # Run optimized (filter only - for comparison with previous results)
    print(f"\nRunning optimized filter (same step)...")
    anc_filter_only = Phase1RefinedANC(config, optimal_length, baseline_step)
    results_filter_only = anc_filter_only.run(duration, scenario)
    print(f"  NR: {results_filter_only['noise_reduction_db']:.1f} dB")

    # Run optimized (filter + adjusted step)
    print(f"\nRunning optimized filter + adjusted step...")
    anc_both = Phase1RefinedANC(config, optimal_length, optimal_step)
    results_both = anc_both.run(duration, scenario)
    print(f"  NR: {results_both['noise_reduction_db']:.1f} dB")

    improvement_filter = results_filter_only['noise_reduction_db'] - results_baseline['noise_reduction_db']
    improvement_both = results_both['noise_reduction_db'] - results_baseline['noise_reduction_db']

    print(f"\n>>> Filter only improvement: {improvement_filter:+.1f} dB")
    print(f">>> Filter + step improvement: {improvement_both:+.1f} dB")

    return {
        'config_key': config_key,
        'config_name': config['name'],
        'scenario': scenario,
        'rt60_est': rt60_est,
        'baseline_length': baseline_length,
        'baseline_step': baseline_step,
        'optimal_length': optimal_length,
        'optimal_step': optimal_step,
        'baseline_nr': results_baseline['noise_reduction_db'],
        'filter_only_nr': results_filter_only['noise_reduction_db'],
        'optimized_nr': results_both['noise_reduction_db'],
        'improvement_filter': improvement_filter,
        'improvement_both': improvement_both,
        'baseline_mse': results_baseline['mse'],
        'optimized_mse': results_both['mse'],
    }


def plot_refined_comparison(all_results: list, save_path: str = None):
    """Generate comparison plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase 1 Refined: Filter Length + Step Size Optimization', fontsize=14, fontweight='bold')

    configs = [r['config_name'] for r in all_results]
    baseline_nrs = [r['baseline_nr'] for r in all_results]
    filter_only_nrs = [r['filter_only_nr'] for r in all_results]
    optimized_nrs = [r['optimized_nr'] for r in all_results]

    x = np.arange(len(configs))
    width = 0.25

    # Bar chart comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x - width, baseline_nrs, width, label='Baseline', color='steelblue')
    bars2 = ax1.bar(x, filter_only_nrs, width, label='Filter Only', color='lightcoral')
    bars3 = ax1.bar(x + width, optimized_nrs, width, label='Filter + Step', color='forestgreen')

    ax1.set_ylabel('Noise Reduction (dB)')
    ax1.set_title('Noise Reduction Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    # Improvement chart
    ax2 = axes[0, 1]
    imp_filter = [r['improvement_filter'] for r in all_results]
    imp_both = [r['improvement_both'] for r in all_results]

    x2 = np.arange(len(configs))
    ax2.bar(x2 - 0.2, imp_filter, 0.35, label='Filter Only', color='lightcoral')
    ax2.bar(x2 + 0.2, imp_both, 0.35, label='Filter + Step', color='forestgreen')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Improvement (dB)')
    ax2.set_title('Improvement over Baseline')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(configs)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Step size comparison
    ax3 = axes[1, 0]
    baseline_steps = [r['baseline_step'] for r in all_results]
    optimal_steps = [r['optimal_step'] for r in all_results]

    ax3.bar(x - 0.2, baseline_steps, 0.35, label='Baseline', color='steelblue')
    ax3.bar(x + 0.2, optimal_steps, 0.35, label='Optimized', color='forestgreen')
    ax3.set_ylabel('Step Size')
    ax3.set_title('Step Size Adjustment')
    ax3.set_xticks(x)
    ax3.set_xticklabels(configs)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_yscale('log')

    # MSE convergence
    ax4 = axes[1, 1]
    r = all_results[0]
    window = 200
    mse_baseline = np.convolve(r['baseline_mse'], np.ones(window)/window, mode='valid')
    mse_optimized = np.convolve(r['optimized_mse'], np.ones(window)/window, mode='valid')
    t = np.arange(len(mse_baseline)) / 16000

    ax4.semilogy(t, mse_baseline, 'b-', label=f"Baseline", alpha=0.7)
    ax4.semilogy(t, mse_optimized, 'g-', label=f"Optimized", alpha=0.7)
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
    print("Phase 1 Refined: Filter Length + Step Size Optimization")
    print("=" * 70)

    output_dir = 'output/phase1_filter_length'
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for config_key, config in STEP7_CONFIGS.items():
        results = run_refined_comparison(config_key, config, duration=5.0)
        all_results.append(results)

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 1 REFINED SUMMARY")
    print("=" * 70)
    print(f"\n{'Config':<15} {'Baseline':<10} {'Filter Only':<12} {'Filter+Step':<12} {'Improvement':<12}")
    print("-" * 70)

    total_baseline = 0
    total_filter = 0
    total_both = 0

    for r in all_results:
        print(f"{r['config_name']:<15} {r['baseline_nr']:>7.1f} dB {r['filter_only_nr']:>9.1f} dB "
              f"{r['optimized_nr']:>9.1f} dB {r['improvement_both']:>+9.1f} dB")
        total_baseline += r['baseline_nr']
        total_filter += r['filter_only_nr']
        total_both += r['optimized_nr']

    n = len(all_results)
    avg_baseline = total_baseline / n
    avg_filter = total_filter / n
    avg_both = total_both / n
    avg_imp_filter = avg_filter - avg_baseline
    avg_imp_both = avg_both - avg_baseline

    print("-" * 70)
    print(f"{'AVERAGE':<15} {avg_baseline:>7.1f} dB {avg_filter:>9.1f} dB "
          f"{avg_both:>9.1f} dB {avg_imp_both:>+9.1f} dB")
    print("=" * 70)

    plot_refined_comparison(all_results, f'{output_dir}/phase1_refined_comparison.png')

    # Verdict
    print("\n" + "=" * 70)
    print("ANALYSIS:")
    print(f"  Filter length alone: {avg_imp_filter:+.1f} dB average")
    print(f"  Filter + step size:  {avg_imp_both:+.1f} dB average")

    if avg_imp_both > 1.0:
        print(f"\nVERDICT: Optimization provides significant improvement (+{avg_imp_both:.1f} dB)")
        print("         Ready for Phase 2")
    elif avg_imp_both > 0:
        print(f"\nVERDICT: Optimization provides modest improvement (+{avg_imp_both:.1f} dB)")
        print("         Consider investigating further or proceed to Phase 2")
    else:
        print(f"\nVERDICT: Optimization shows no clear benefit ({avg_imp_both:.1f} dB)")
        print("         Need to investigate root cause")
    print("=" * 70)

    return all_results


if __name__ == '__main__':
    main()
