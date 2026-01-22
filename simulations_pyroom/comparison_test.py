#!/usr/bin/env python3
"""
ANC Configuration Comparison Test Script

Compares ANC performance across 3 configurations and 4 driving scenarios:
- Configurations: 1ref/1spk, 1ref/4spk, 4ref/4spk
- Scenarios: Highway, City, Acceleration, Idle

Outputs all visualization plots including car interior layout for each combination.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add paths for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "playground"))

from playground.presets import (
    ROOM_PRESETS, SCENARIO_PRESETS,
    FOUR_SPEAKER_CONFIG, FOUR_REF_MIC_CONFIG,
    NOISE_SOURCE_POSITIONS, SCENARIO_NOISE_POSITIONS
)
from playground.simulation.runner import run_simulation
from playground.components.plots import (
    plot_before_after, plot_spectrum, plot_convergence, plot_filter_coefficients,
    plot_noise_source_time, plot_noise_source_freq,
    plot_error_mic_time, plot_error_mic_freq,
    plot_ref_mic_signals_time, plot_ref_mic_signals_freq
)
from playground.components.room_interactive import create_interactive_room_diagram

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "output" / "comparison_test"

# Scenarios to test (excluding Custom)
SCENARIOS = ['Highway', 'City', 'Acceleration', 'Idle']


def get_noise_position_for_scenario(scenario: str) -> list:
    """Get the appropriate noise source position for a scenario."""
    noise_pos_key = SCENARIO_NOISE_POSITIONS.get(scenario, 'Combined (Dashboard)')
    return NOISE_SOURCE_POSITIONS[noise_pos_key].copy()


def build_base_params(scenario: str) -> dict:
    """Build base parameters from Sedan preset."""
    room_preset = ROOM_PRESETS['Sedan']

    params = {
        'dimensions': room_preset['dimensions'].copy(),
        'absorption': room_preset['absorption'],
        'max_order': room_preset['max_order'],
        'positions': {k: v.copy() for k, v in room_preset['positions'].items()},
        'filter_length': 256,
        'step_size': 0.005,
        'sample_rate': 16000,
        'duration': 5.0,
        'scenario': scenario.lower(),
    }

    # Set noise source position based on scenario
    params['positions']['noise_source'] = get_noise_position_for_scenario(scenario)

    return params


def build_params_1ref_1spk(scenario: str) -> dict:
    """Build parameters for 1 ref mic, 1 speaker configuration."""
    params = build_base_params(scenario)
    params['speaker_mode'] = 'Single Speaker'
    params['ref_mic_mode'] = 'Single Reference Mic'
    return params


def build_params_1ref_4spk(scenario: str) -> dict:
    """Build parameters for 1 ref mic, 4 speakers configuration."""
    params = build_base_params(scenario)
    params['speaker_mode'] = '4-Speaker System'
    params['ref_mic_mode'] = 'Single Reference Mic'
    params['speakers'] = {k: v.copy() for k, v in FOUR_SPEAKER_CONFIG.items()}
    return params


def build_params_4ref_4spk(scenario: str) -> dict:
    """Build parameters for 4 ref mics, 4 speakers configuration."""
    params = build_base_params(scenario)
    params['speaker_mode'] = '4-Speaker System'
    params['ref_mic_mode'] = '4-Reference Mic System'
    params['speakers'] = {k: v.copy() for k, v in FOUR_SPEAKER_CONFIG.items()}
    params['ref_mics'] = {k: v.copy() for k, v in FOUR_REF_MIC_CONFIG.items()}
    return params


def save_all_plots(results: dict, params: dict, output_dir: Path):
    """Save all visualization plots for a simulation run."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Basic plots
    plots = [
        ('before_after', plot_before_after),
        ('spectrum', plot_spectrum),
        ('convergence', plot_convergence),
        ('filter_coefficients', plot_filter_coefficients),
        ('noise_source_time', plot_noise_source_time),
        ('noise_source_freq', plot_noise_source_freq),
        ('error_mic_time', plot_error_mic_time),
        ('error_mic_freq', plot_error_mic_freq),
    ]

    for name, plot_func in plots:
        try:
            fig = plot_func(results)
            filepath = output_dir / f"{name}.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved: {name}.png")
        except Exception as e:
            print(f"    Error saving {name}: {e}")

    # Reference mic plots (only for 4-ref-mic mode)
    if results.get('ref_mic_mode') == '4-Reference Mic System':
        ref_plots = [
            ('ref_mic_signals_time', plot_ref_mic_signals_time),
            ('ref_mic_signals_freq', plot_ref_mic_signals_freq),
        ]
        for name, plot_func in ref_plots:
            try:
                fig = plot_func(results)
                filepath = output_dir / f"{name}.png"
                fig.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"    Saved: {name}.png")
            except Exception as e:
                print(f"    Error saving {name}: {e}")


def save_room_layout(params: dict, output_dir: Path):
    """Save the room layout diagram."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dimensions = params['dimensions']
    positions = params['positions']

    # Get multi-speaker/multi-ref-mic positions if applicable
    speakers_4ch = params.get('speakers') if params.get('speaker_mode') == '4-Speaker System' else None
    ref_mics_4ch = params.get('ref_mics') if params.get('ref_mic_mode') == '4-Reference Mic System' else None

    fig = create_interactive_room_diagram(
        dimensions,
        positions,
        speakers_4ch=speakers_4ch,
        ref_mics_4ch=ref_mics_4ch,
        is_car=True
    )

    # Save as PNG (requires kaleido)
    try:
        filepath = output_dir / "room_layout.png"
        fig.write_image(str(filepath), width=1000, height=700, scale=2)
        print(f"    Saved: room_layout.png")
    except Exception as e:
        print(f"    Error saving room_layout.png: {e}")
        print(f"    (Install kaleido: pip install kaleido)")
        # Fallback: save as HTML
        filepath = output_dir / "room_layout.html"
        fig.write_html(str(filepath))
        print(f"    Saved fallback: room_layout.html")


def run_single_test(config_name: str, config_builder, scenario: str) -> dict:
    """Run a single test configuration."""
    print(f"\n  Running {config_name} / {scenario}...")

    params = config_builder(scenario)

    def progress_callback(progress, mse):
        if progress >= 0.99:
            print(f"    Progress: 100%")

    results = run_simulation(params, progress_callback=progress_callback)

    if results['success']:
        print(f"    Noise Reduction: {results['noise_reduction_db']:.2f} dB")
    else:
        print(f"    FAILED: {results['error_message']}")

    return results


def run_comparison_test():
    """Run the full comparison test across all configurations and scenarios."""
    print("=" * 70)
    print("ANC CONFIGURATION COMPARISON TEST")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Configurations: 1ref_1spk, 1ref_4spk, 4ref_4spk")
    print(f"Scenarios: {', '.join(SCENARIOS)}")
    print(f"Total runs: {3 * len(SCENARIOS)}")

    # Configuration definitions
    configs = [
        ('1ref_1spk', build_params_1ref_1spk, '1 Ref Mic / 1 Speaker'),
        ('1ref_4spk', build_params_1ref_4spk, '1 Ref Mic / 4 Speakers'),
        ('4ref_4spk', build_params_4ref_4spk, '4 Ref Mics / 4 Speakers'),
    ]

    # Results storage
    all_results = {}

    for config_name, config_builder, config_desc in configs:
        print(f"\n{'=' * 70}")
        print(f"Configuration: {config_desc}")
        print('=' * 70)

        all_results[config_name] = {}

        for scenario in SCENARIOS:
            # Run simulation
            results = run_single_test(config_name, config_builder, scenario)
            all_results[config_name][scenario] = results

            if results['success']:
                # Create output directory
                output_dir = OUTPUT_DIR / config_name / scenario

                # Save plots
                print(f"  Saving plots...")
                save_all_plots(results, results['params'], output_dir)
                save_room_layout(results['params'], output_dir)

    # Generate summary report
    generate_summary_report(all_results)

    print(f"\n{'=' * 70}")
    print("COMPARISON TEST COMPLETE")
    print(f"{'=' * 70}")
    print(f"Results saved to: {OUTPUT_DIR}")


def generate_summary_report(all_results: dict):
    """Generate a summary report of all test results."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Console output
    print(f"\n{'=' * 70}")
    print("SUMMARY REPORT")
    print(f"{'=' * 70}")

    # Table header
    header = f"{'Configuration':<20}"
    for scenario in SCENARIOS:
        header += f"{scenario:<15}"
    print(header)
    print("-" * 70)

    # Table rows
    for config_name in all_results:
        row = f"{config_name:<20}"
        for scenario in SCENARIOS:
            result = all_results[config_name][scenario]
            if result['success']:
                nr_db = result['noise_reduction_db']
                row += f"{nr_db:>6.2f} dB{'':<6}"
            else:
                row += f"{'FAILED':<15}"
        print(row)

    # Save to file
    summary_file = OUTPUT_DIR / "summary_report.txt"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(summary_file, 'w') as f:
        f.write("ANC CONFIGURATION COMPARISON TEST - SUMMARY REPORT\n")
        f.write(f"Generated: {timestamp}\n")
        f.write("=" * 70 + "\n\n")

        f.write("CONFIGURATIONS:\n")
        f.write("  1ref_1spk: 1 Reference Mic / 1 Speaker (baseline)\n")
        f.write("  1ref_4spk: 1 Reference Mic / 4 Speakers\n")
        f.write("  4ref_4spk: 4 Reference Mics / 4 Speakers\n\n")

        f.write("SCENARIOS:\n")
        for scenario in SCENARIOS:
            desc = SCENARIO_PRESETS[scenario]['description']
            f.write(f"  {scenario}: {desc}\n")
        f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("NOISE REDUCTION RESULTS (dB)\n")
        f.write("=" * 70 + "\n\n")

        # Table
        header = f"{'Configuration':<20}"
        for scenario in SCENARIOS:
            header += f"{scenario:<15}"
        f.write(header + "\n")
        f.write("-" * 70 + "\n")

        for config_name in all_results:
            row = f"{config_name:<20}"
            for scenario in SCENARIOS:
                result = all_results[config_name][scenario]
                if result['success']:
                    nr_db = result['noise_reduction_db']
                    row += f"{nr_db:>6.2f} dB{'':<6}"
                else:
                    row += f"{'FAILED':<15}"
            f.write(row + "\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("=" * 70 + "\n\n")

        for config_name in all_results:
            f.write(f"\n{config_name}:\n")
            f.write("-" * 40 + "\n")
            for scenario in SCENARIOS:
                result = all_results[config_name][scenario]
                if result['success']:
                    f.write(f"  {scenario}:\n")
                    f.write(f"    Noise Reduction: {result['noise_reduction_db']:.2f} dB\n")
                    f.write(f"    Final MSE: {result['mse'][-1]:.2e}\n")
                    f.write(f"    Filter Taps: {len(result['weights'])}\n")
                else:
                    f.write(f"  {scenario}: FAILED - {result['error_message']}\n")

    print(f"\nSummary saved to: {summary_file}")


if __name__ == '__main__':
    run_comparison_test()
