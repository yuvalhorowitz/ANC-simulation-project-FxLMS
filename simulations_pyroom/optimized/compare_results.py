"""
Compare Results: Baseline vs Optimized Configurations

Compares:
1. Baseline (original parameters) vs Optimized (tuned parameters)
2. Single speaker vs 4-speaker stereo configurations
3. Performance across different scenarios and car types

Generates comparison plots and summary statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results(results_path: str) -> Dict:
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def create_comparison_bar_chart(
    optimized_results: Dict,
    baseline_results: Optional[Dict] = None,
    save_path: str = None
):
    """
    Create bar chart comparing configurations.

    Args:
        optimized_results: Results from optimized simulation
        baseline_results: Optional baseline results for comparison
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('ANC Performance: Single Speaker vs 4-Speaker Stereo', fontsize=14, fontweight='bold')

    scenarios = ['idle', 'city', 'highway']
    configs = [k for k in optimized_results.keys() if k != 'timestamp']

    x = np.arange(len(configs))
    width = 0.35

    for ax_idx, scenario in enumerate(scenarios):
        ax = axes[ax_idx]

        single_vals = []
        quad_vals = []

        for config in configs:
            if config in optimized_results and 'single' in optimized_results[config]:
                if scenario in optimized_results[config]['single']:
                    single_vals.append(optimized_results[config]['single'][scenario]['noise_reduction_db'])
                else:
                    single_vals.append(0)
            else:
                single_vals.append(0)

            if config in optimized_results and 'quad_stereo' in optimized_results[config]:
                if scenario in optimized_results[config]['quad_stereo']:
                    quad_vals.append(optimized_results[config]['quad_stereo'][scenario]['noise_reduction_db'])
                else:
                    quad_vals.append(0)
            else:
                quad_vals.append(0)

        rects1 = ax.bar(x - width/2, single_vals, width, label='Single Speaker', color='steelblue')
        rects2 = ax.bar(x + width/2, quad_vals, width, label='4-Speaker Stereo', color='darkorange')

        ax.set_ylabel('Noise Reduction (dB)')
        ax.set_title(f'{scenario.title()} Scenario')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for rect in rects1:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(rect.get_x() + rect.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

        for rect in rects2:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(rect.get_x() + rect.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    else:
        plt.show()

    return fig


def create_speaker_comparison_heatmap(
    optimized_results: Dict,
    save_path: str = None
):
    """
    Create heatmap showing improvement from single to quad speakers.

    Args:
        optimized_results: Results from optimized simulation
        save_path: Path to save plot
    """
    configs = [k for k in optimized_results.keys() if k != 'timestamp']
    scenarios = ['idle', 'city', 'highway']

    # Calculate differences
    diff_matrix = np.zeros((len(configs), len(scenarios)))

    for i, config in enumerate(configs):
        for j, scenario in enumerate(scenarios):
            single_nr = optimized_results[config].get('single', {}).get(scenario, {}).get('noise_reduction_db', 0)
            quad_nr = optimized_results[config].get('quad_stereo', {}).get(scenario, {}).get('noise_reduction_db', 0)
            diff_matrix[i, j] = quad_nr - single_nr

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(diff_matrix, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=5)

    # Labels
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_yticks(np.arange(len(configs)))
    ax.set_xticklabels([s.title() for s in scenarios])
    ax.set_yticklabels(configs)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('NR Difference (dB): Quad - Single', rotation=-90, va="bottom")

    # Add text annotations
    for i in range(len(configs)):
        for j in range(len(scenarios)):
            text = ax.text(j, i, f'{diff_matrix[i, j]:+.1f}',
                          ha="center", va="center", color="black", fontweight='bold')

    ax.set_title('4-Speaker vs Single Speaker Improvement (dB)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    else:
        plt.show()

    return fig


def create_parameter_impact_chart(
    optimized_results: Dict,
    save_path: str = None
):
    """
    Create chart showing relationship between parameters and performance.

    Args:
        optimized_results: Results from optimized simulation
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Parameter Impact on Performance', fontsize=14, fontweight='bold')

    configs = [k for k in optimized_results.keys() if k != 'timestamp']

    # Collect data
    filter_lengths = []
    max_orders = []
    avg_nr_single = []
    avg_nr_quad = []

    for config in configs:
        # Get first scenario's parameters (they should be same across scenarios)
        params = list(optimized_results[config].get('single', {}).values())[0] if optimized_results[config].get('single') else {}

        filter_lengths.append(params.get('filter_length', 256))
        max_orders.append(params.get('max_order', 3))

        # Average NR across scenarios
        single_nrs = [r['noise_reduction_db'] for r in optimized_results[config].get('single', {}).values()]
        quad_nrs = [r['noise_reduction_db'] for r in optimized_results[config].get('quad_stereo', {}).values()]

        avg_nr_single.append(np.mean(single_nrs) if single_nrs else 0)
        avg_nr_quad.append(np.mean(quad_nrs) if quad_nrs else 0)

    # Filter length vs NR
    ax1 = axes[0]
    ax1.scatter(filter_lengths, avg_nr_single, s=100, c='steelblue', label='Single', marker='o')
    ax1.scatter(filter_lengths, avg_nr_quad, s=100, c='darkorange', label='Quad', marker='s')

    for i, config in enumerate(configs):
        ax1.annotate(config, (filter_lengths[i], avg_nr_single[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax1.set_xlabel('Filter Length (taps)')
    ax1.set_ylabel('Average Noise Reduction (dB)')
    ax1.set_title('Filter Length vs Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Max order vs NR
    ax2 = axes[1]
    ax2.scatter(max_orders, avg_nr_single, s=100, c='steelblue', label='Single', marker='o')
    ax2.scatter(max_orders, avg_nr_quad, s=100, c='darkorange', label='Quad', marker='s')

    for i, config in enumerate(configs):
        ax2.annotate(config, (max_orders[i], avg_nr_single[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax2.set_xlabel('Max Order (reflections)')
    ax2.set_ylabel('Average Noise Reduction (dB)')
    ax2.set_title('Max Order vs Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    else:
        plt.show()

    return fig


def generate_text_report(
    optimized_results: Dict,
    baseline_results: Optional[Dict] = None
) -> str:
    """
    Generate text report comparing results.

    Args:
        optimized_results: Results from optimized simulation
        baseline_results: Optional baseline results

    Returns:
        Formatted text report
    """
    lines = []
    lines.append("=" * 80)
    lines.append("OPTIMIZED ANC SIMULATION RESULTS REPORT")
    lines.append("=" * 80)
    lines.append("")

    if 'timestamp' in optimized_results:
        lines.append(f"Generated: {optimized_results['timestamp']}")
        lines.append("")

    configs = [k for k in optimized_results.keys() if k != 'timestamp']
    scenarios = ['idle', 'city', 'highway']

    # Summary table
    lines.append("-" * 80)
    lines.append("NOISE REDUCTION SUMMARY (dB)")
    lines.append("-" * 80)
    lines.append(f"{'Config':<15} {'Speaker':<12} {'Idle':>10} {'City':>10} {'Highway':>10} {'Average':>10}")
    lines.append("-" * 80)

    for config in configs:
        for speaker_type in ['single', 'quad_stereo']:
            if speaker_type in optimized_results[config]:
                nrs = []
                row = f"{config:<15} {speaker_type:<12}"
                for scenario in scenarios:
                    if scenario in optimized_results[config][speaker_type]:
                        nr = optimized_results[config][speaker_type][scenario]['noise_reduction_db']
                        nrs.append(nr)
                        row += f" {nr:>10.1f}"
                    else:
                        row += f" {'N/A':>10}"

                avg = np.mean(nrs) if nrs else 0
                row += f" {avg:>10.1f}"
                lines.append(row)

        lines.append("")

    # Speaker configuration comparison
    lines.append("-" * 80)
    lines.append("SPEAKER CONFIGURATION COMPARISON")
    lines.append("-" * 80)
    lines.append(f"{'Config':<15} {'Single Avg':>12} {'Quad Avg':>12} {'Improvement':>12}")
    lines.append("-" * 80)

    for config in configs:
        single_nrs = [r['noise_reduction_db']
                     for r in optimized_results[config].get('single', {}).values()]
        quad_nrs = [r['noise_reduction_db']
                   for r in optimized_results[config].get('quad_stereo', {}).values()]

        single_avg = np.mean(single_nrs) if single_nrs else 0
        quad_avg = np.mean(quad_nrs) if quad_nrs else 0
        improvement = quad_avg - single_avg

        lines.append(f"{config:<15} {single_avg:>12.1f} {quad_avg:>12.1f} {improvement:>+12.1f}")

    lines.append("")

    # Parameter summary
    lines.append("-" * 80)
    lines.append("OPTIMIZED PARAMETERS")
    lines.append("-" * 80)
    lines.append(f"{'Config':<15} {'Filter Len':>12} {'Max Order':>12} {'Step Size':>12}")
    lines.append("-" * 80)

    for config in configs:
        params = list(optimized_results[config].get('single', {}).values())[0] if optimized_results[config].get('single') else {}
        lines.append(f"{config:<15} "
                    f"{params.get('filter_length', 'N/A'):>12} "
                    f"{params.get('max_order', 'N/A'):>12} "
                    f"{params.get('step_size', 'N/A'):>12}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    """Main comparison function."""
    print("=" * 70)
    print("Results Comparison Tool")
    print("=" * 70)

    # Create output directory
    output_dir = 'output/optimized'
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    results_path = f'{output_dir}/results_summary.json'

    if not os.path.exists(results_path):
        print(f"\nResults file not found: {results_path}")
        print("Please run the optimized simulation first:")
        print("  python -m simulations_pyroom.optimized.run_optimized_simulation")
        return

    print(f"\nLoading results from: {results_path}")
    results = load_results(results_path)

    # Generate comparison charts
    print("\nGenerating comparison charts...")

    create_comparison_bar_chart(
        results,
        save_path=f'{output_dir}/comparison_bars.png'
    )

    create_speaker_comparison_heatmap(
        results,
        save_path=f'{output_dir}/speaker_comparison_heatmap.png'
    )

    create_parameter_impact_chart(
        results,
        save_path=f'{output_dir}/parameter_impact.png'
    )

    # Generate text report
    print("\nGenerating text report...")
    report = generate_text_report(results)

    report_path = f'{output_dir}/comparison_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved: {report_path}")

    # Print report to console
    print("\n" + report)

    print("\n" + "=" * 70)
    print("Comparison complete!")
    print("=" * 70)
    print("\nOutput files:")
    print(f"  {output_dir}/comparison_bars.png")
    print(f"  {output_dir}/speaker_comparison_heatmap.png")
    print(f"  {output_dir}/parameter_impact.png")
    print(f"  {output_dir}/comparison_report.txt")


if __name__ == '__main__':
    main()
