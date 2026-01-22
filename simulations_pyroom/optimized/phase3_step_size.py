"""
Phase 3: Step Size Grid Search

Tests different step size values to find optimal adaptation rate.

Current Issue:
    Step sizes are set heuristically without systematic optimization.

Hypothesis:
    Finding the optimal step size can improve both convergence speed
    and final noise reduction.

Test Grid: [0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02]
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


STEP_SIZE_GRID = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02]


class Phase3ANC:
    """ANC with configurable step size."""

    def __init__(self, config: dict, step_size: float, fs: int = 16000):
        self.fs = fs
        self.config = config
        self.step_size = step_size

        room_cfg = config['room']
        pos = config['positions']
        fxlms_cfg = config['fxlms']

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

        self.room.add_source(pos['noise_source'])
        self.room.add_source(pos['speaker'])
        mic_array = np.array([pos['reference_mic'], pos['error_mic']]).T
        self.room.add_microphone_array(pra.MicrophoneArray(mic_array, fs=fs))

        self.room.compute_rir()

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
            step_size=step_size,  # KEY: Use specified step_size
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

        # Check for divergence
        final_mse = np.mean(mse[-1000:])
        diverged = final_mse > 1e6 or np.isnan(final_mse) or np.isinf(final_mse)

        steady_start = len(desired) // 2
        d_power = np.mean(np.array(desired[steady_start:])**2)
        e_power = np.mean(np.array(error[steady_start:])**2)

        if diverged or e_power > d_power * 10:
            nr_db = -10.0  # Mark as failed
        elif e_power > 1e-10:
            nr_db = 10 * np.log10(d_power / e_power)
        else:
            nr_db = 60.0

        # Calculate convergence time (time to reach 90% of final NR)
        window = 500
        mse_smooth = np.convolve(mse, np.ones(window)/window, mode='valid')
        final_mse_smooth = np.mean(mse_smooth[-1000:]) if len(mse_smooth) > 1000 else np.mean(mse_smooth)
        threshold = final_mse_smooth * 1.1

        conv_idx = np.where(mse_smooth < threshold)[0]
        if len(conv_idx) > 0:
            convergence_samples = conv_idx[0] + window
            convergence_time = convergence_samples / self.fs
        else:
            convergence_time = duration

        return {
            'noise_reduction_db': nr_db,
            'convergence_time': convergence_time,
            'diverged': diverged,
            'mse': np.array(mse),
        }


def run_grid_search(config_key: str, config: dict, duration: float = 5.0):
    """Run step size grid search for a configuration."""
    scenario = config.get('scenario', 'highway')
    baseline_step = config['fxlms']['step_size']

    print(f"\n{'='*60}")
    print(f"Configuration: {config['name']}")
    print(f"{'='*60}")
    print(f"Scenario: {scenario}")
    print(f"Baseline step size: {baseline_step}")
    print(f"\nTesting step sizes: {STEP_SIZE_GRID}")

    results = []
    baseline_nr = None

    for step_size in STEP_SIZE_GRID:
        print(f"\n  Testing μ={step_size}...", end=" ")
        anc = Phase3ANC(config, step_size)
        result = anc.run(duration, scenario)

        if step_size == baseline_step:
            baseline_nr = result['noise_reduction_db']

        status = "DIVERGED" if result['diverged'] else f"NR={result['noise_reduction_db']:.1f} dB"
        print(status)

        results.append({
            'step_size': step_size,
            'nr_db': result['noise_reduction_db'],
            'convergence_time': result['convergence_time'],
            'diverged': result['diverged'],
            'mse': result['mse'],
        })

    # Find best step size
    valid_results = [r for r in results if not r['diverged'] and r['nr_db'] > 0]
    if valid_results:
        best = max(valid_results, key=lambda x: x['nr_db'])
        best_step = best['step_size']
        best_nr = best['nr_db']
    else:
        best_step = baseline_step
        best_nr = baseline_nr if baseline_nr else 0

    # Calculate improvement
    improvement = best_nr - (baseline_nr if baseline_nr else 0)

    print(f"\n>>> Best step size: {best_step}")
    print(f">>> Best NR: {best_nr:.1f} dB")
    print(f">>> Improvement over baseline: {improvement:+.1f} dB")

    return {
        'config_name': config['name'],
        'scenario': scenario,
        'baseline_step': baseline_step,
        'baseline_nr': baseline_nr,
        'best_step': best_step,
        'best_nr': best_nr,
        'improvement': improvement,
        'all_results': results,
    }


def plot_results(all_results: list, save_path: str = None):
    """Generate comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase 3: Step Size Grid Search', fontsize=14, fontweight='bold')

    # NR vs step size for each config
    ax1 = axes[0, 0]
    for r in all_results:
        step_sizes = [x['step_size'] for x in r['all_results']]
        nrs = [x['nr_db'] for x in r['all_results']]
        ax1.semilogx(step_sizes, nrs, 'o-', label=r['config_name'], markersize=6)
        # Mark baseline
        ax1.axvline(x=r['baseline_step'], linestyle='--', alpha=0.3)

    ax1.set_xlabel('Step Size (μ)')
    ax1.set_ylabel('Noise Reduction (dB)')
    ax1.set_title('NR vs Step Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Best vs baseline comparison
    ax2 = axes[0, 1]
    configs = [r['config_name'] for r in all_results]
    baseline_nrs = [r['baseline_nr'] if r['baseline_nr'] else 0 for r in all_results]
    best_nrs = [r['best_nr'] for r in all_results]

    x = np.arange(len(configs))
    width = 0.35
    ax2.bar(x - width/2, baseline_nrs, width, label='Baseline', color='steelblue')
    ax2.bar(x + width/2, best_nrs, width, label='Optimized', color='forestgreen')
    ax2.set_ylabel('Noise Reduction (dB)')
    ax2.set_title('Baseline vs Best Step Size')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Improvement
    ax3 = axes[1, 0]
    improvements = [r['improvement'] for r in all_results]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax3.bar(configs, improvements, color=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_ylabel('Improvement (dB)')
    ax3.set_title('Step Size Optimization Impact')
    ax3.grid(True, alpha=0.3, axis='y')

    for i, imp in enumerate(improvements):
        ax3.annotate(f'{imp:+.1f}', xy=(i, imp),
                    xytext=(0, 5 if imp > 0 else -15),
                    textcoords='offset points', ha='center', fontsize=10, fontweight='bold')

    # Optimal step sizes
    ax4 = axes[1, 1]
    baseline_steps = [r['baseline_step'] for r in all_results]
    best_steps = [r['best_step'] for r in all_results]

    ax4.bar(x - width/2, baseline_steps, width, label='Baseline', color='steelblue')
    ax4.bar(x + width/2, best_steps, width, label='Optimized', color='forestgreen')
    ax4.set_ylabel('Step Size')
    ax4.set_title('Step Size: Baseline vs Optimal')
    ax4.set_xticks(x)
    ax4.set_xticklabels(configs)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_yscale('log')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {save_path}")
    plt.close()


def main():
    print("=" * 70)
    print("Phase 3: Step Size Grid Search")
    print("=" * 70)

    output_dir = 'output/phase3_step_size'
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    for config_key, config in STEP7_CONFIGS.items():
        results = run_grid_search(config_key, config, duration=5.0)
        all_results.append(results)

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 3 RESULTS: STEP SIZE OPTIMIZATION")
    print("=" * 70)
    print(f"\n{'Config':<15} {'Baseline NR':<12} {'Best NR':<10} {'Improvement':<12} {'Step':<20}")
    print("-" * 75)

    total_baseline = 0
    total_best = 0

    for r in all_results:
        baseline = r['baseline_nr'] if r['baseline_nr'] else 0
        print(f"{r['config_name']:<15} {baseline:>9.1f} dB {r['best_nr']:>7.1f} dB "
              f"{r['improvement']:>+9.1f} dB  {r['baseline_step']}->{r['best_step']}")
        total_baseline += baseline
        total_best += r['best_nr']

    n = len(all_results)
    avg_baseline = total_baseline / n
    avg_best = total_best / n
    avg_improvement = avg_best - avg_baseline

    print("-" * 75)
    print(f"{'AVERAGE':<15} {avg_baseline:>9.1f} dB {avg_best:>7.1f} dB {avg_improvement:>+9.1f} dB")
    print("=" * 70)

    plot_results(all_results, f'{output_dir}/phase3_comparison.png')

    # Save results
    with open(f'{output_dir}/phase3_results.txt', 'w') as f:
        f.write(f"Phase 3: Step Size Grid Search Results\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 60 + "\n\n")
        for r in all_results:
            f.write(f"{r['config_name']}:\n")
            f.write(f"  Baseline: μ={r['baseline_step']}, NR={r['baseline_nr']:.1f} dB\n")
            f.write(f"  Optimal:  μ={r['best_step']}, NR={r['best_nr']:.1f} dB\n")
            f.write(f"  Improvement: {r['improvement']:+.1f} dB\n\n")
        f.write(f"\nAverage improvement: {avg_improvement:+.1f} dB\n")

    print(f"\nResults saved to: {output_dir}/")

    # Verdict
    print("\n" + "=" * 70)
    if avg_improvement >= 1.0:
        print(f"SUCCESS: Phase 3 provides +{avg_improvement:.1f} dB improvement")
        print("         Recommend implementing step size optimization")
    elif avg_improvement > 0:
        print(f"MODEST: Phase 3 provides +{avg_improvement:.1f} dB improvement")
    else:
        print(f"NO IMPROVEMENT: Phase 3 shows {avg_improvement:.1f} dB change")
        print("         Skip this optimization")
    print("=" * 70)

    return all_results


if __name__ == '__main__':
    main()
