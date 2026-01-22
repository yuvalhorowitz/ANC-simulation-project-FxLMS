"""
Step 8: Single Speaker vs 4-Speaker ANC Comparison

Answers the question: Is 4-speaker better than a well-placed single speaker?

Compares:
1. Single headrest speaker (optimal single-speaker location)
2. 4-speaker stereo system

Tests across: engine, road, and highway noise scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, RegularPolygon
import sys
import os
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyroomacoustics as pra
from src.acoustic.path_generator import FIRPath
from src.core.fxlms import FxNLMS
from src.noise.noise_mixer import NoiseMixer


# =============================================================================
# Car Configuration
# =============================================================================

SEDAN_DIMS = [4.5, 1.85, 1.2]  # Length, Width, Height (m)

SEDAN_MATERIALS = {
    'ceiling': 0.38, 'floor': 0.52,
    'east': 0.14, 'west': 0.14,
    'north': 0.20, 'south': 0.30,
}

# Key positions
DRIVER_HEAD = [1.8, 0.55, 1.0]
REF_MIC = [0.3, 0.92, 0.5]
ERROR_MIC = DRIVER_HEAD.copy()

# Speaker configurations to compare
CONFIGS = {
    'single_headrest': {
        'name': 'Single Headrest Speaker',
        'speakers': {'headrest': [1.9, 0.55, 1.0]},
        'color': '#3498db',
    },
    'quad_stereo': {
        'name': '4-Speaker Stereo',
        'speakers': {
            'door_L': [1.5, 0.1, 0.5],
            'door_R': [1.5, 1.75, 0.5],
            'dash_L': [0.8, 0.35, 0.9],
            'dash_R': [0.8, 1.50, 0.9],
        },
        'color': '#2ecc71',
    },
}

# Noise scenarios
SCENARIOS = {
    'engine': {'name': 'Engine (Idle)', 'source': [0.3, 0.92, 0.4], 'type': 'idle'},
    'road': {'name': 'Road (City)', 'source': [1.0, 0.92, 0.15], 'type': 'city'},
    'highway': {'name': 'Highway (Combined)', 'source': [0.5, 0.92, 0.5], 'type': 'highway'},
}


# =============================================================================
# ANC Simulator
# =============================================================================

class ANCSimulator:
    """Simple ANC simulator for speaker comparison."""

    def __init__(self, speakers: dict, noise_source: list, fs: int = 16000):
        self.fs = fs
        self.speakers = speakers
        self.speaker_names = list(speakers.keys())

        # Create room
        materials = {k: pra.Material(v) for k, v in SEDAN_MATERIALS.items()}

        self.room = pra.ShoeBox(
            SEDAN_DIMS, fs=fs, materials=materials,
            max_order=3, air_absorption=True
        )

        # Add sources: noise (0), then speakers (1, 2, ...)
        self.room.add_source(noise_source)
        for name in self.speaker_names:
            self.room.add_source(speakers[name])

        # Add mics: reference (0), error (1)
        mic_array = np.array([REF_MIC, ERROR_MIC]).T
        self.room.add_microphone_array(pra.MicrophoneArray(mic_array, fs=fs))

        self.room.compute_rir()

        # Extract paths
        max_len = 512
        self.H_primary = self.room.rir[1][0][:max_len]
        self.H_reference = self.room.rir[0][0][:max_len]

        # Combined secondary path (all speakers)
        self.H_secondary = np.zeros(max_len)
        self.H_secondary_per_speaker = {}
        for i, name in enumerate(self.speaker_names):
            h = self.room.rir[1][i + 1][:max_len]
            self.H_secondary_per_speaker[name] = h
            self.H_secondary[:len(h)] += h

        self.H_secondary_est = self.H_secondary * (1 + 0.05 * np.random.randn(max_len))

        # Create FIR filters
        self.primary_path = FIRPath(self.H_primary)
        self.reference_path = FIRPath(self.H_reference)
        self.secondary_paths = {
            name: FIRPath(self.H_secondary_per_speaker[name])
            for name in self.speaker_names
        }

    def run(self, noise_signal: np.ndarray) -> dict:
        """Run ANC simulation."""
        n_samples = len(noise_signal)

        fxlms = FxNLMS(
            filter_length=256,
            step_size=0.005,
            secondary_path_estimate=self.H_secondary_est,
            regularization=1e-4
        )

        # Reset paths
        self.primary_path.reset()
        self.reference_path.reset()
        for path in self.secondary_paths.values():
            path.reset()

        desired = np.zeros(n_samples)
        error = np.zeros(n_samples)

        for i in range(n_samples):
            sample = noise_signal[i]
            x = self.reference_path.filter_sample(sample)
            d = self.primary_path.filter_sample(sample)
            desired[i] = d

            y = fxlms.generate_antinoise(x)

            y_at_error = sum(
                self.secondary_paths[name].filter_sample(y)
                for name in self.speaker_names
            )

            e = d + y_at_error
            error[i] = e

            fxlms.filter_reference(x)
            fxlms.update_weights(e)

        # Calculate noise reduction (steady state)
        steady_start = n_samples // 2
        d_power = np.mean(desired[steady_start:]**2)
        e_power = np.mean(error[steady_start:]**2)

        nr_db = 10 * np.log10(d_power / e_power) if e_power > 1e-10 else 60.0

        return {
            'noise_reduction_db': nr_db,
            'desired': desired,
            'error': error,
        }


# =============================================================================
# Visualization
# =============================================================================

def plot_car_interior(config_key: str, config: dict, noise_source: list,
                      scenario_name: str, nr_db: float, save_path: str):
    """Plot car interior with component positions."""

    fig, ax = plt.subplots(figsize=(12, 8))

    length, width, height = SEDAN_DIMS

    # Car outline
    car = plt.Rectangle((0, 0), length, width, fill=False,
                         edgecolor='#333', linewidth=3)
    ax.add_patch(car)

    # Windshield (front)
    windshield = plt.Polygon(
        [[0, 0.15], [0, width-0.15], [0.35, width-0.25], [0.35, 0.25]],
        facecolor='#87CEEB', edgecolor='#333', alpha=0.5, linewidth=2
    )
    ax.add_patch(windshield)
    ax.text(0.17, width/2, 'FRONT', ha='center', va='center', fontsize=10,
            color='#333', rotation=90, fontweight='bold')

    # Rear window
    rear = plt.Polygon(
        [[length, 0.2], [length, width-0.2], [length-0.25, width-0.3], [length-0.25, 0.3]],
        facecolor='#87CEEB', edgecolor='#333', alpha=0.5, linewidth=2
    )
    ax.add_patch(rear)

    # Dashboard
    dashboard = FancyBboxPatch((0.4, 0.2), 0.55, width-0.4,
                                boxstyle="round,pad=0.02",
                                facecolor='#555', edgecolor='#333', alpha=0.5)
    ax.add_patch(dashboard)
    ax.text(0.67, width/2, 'Dashboard', ha='center', va='center',
            fontsize=9, color='white', fontweight='bold')

    # Driver seat
    driver_seat = FancyBboxPatch((1.2, 0.15), 0.9, 0.7,
                                  boxstyle="round,pad=0.03",
                                  facecolor='#8B4513', edgecolor='#5D3A1A',
                                  alpha=0.7, linewidth=2)
    ax.add_patch(driver_seat)
    ax.text(1.65, 0.5, 'Driver\nSeat', ha='center', va='center',
            fontsize=9, color='white', fontweight='bold')

    # Passenger seat
    pass_seat = FancyBboxPatch((1.2, 1.0), 0.9, 0.7,
                                boxstyle="round,pad=0.03",
                                facecolor='#8B4513', edgecolor='#5D3A1A',
                                alpha=0.7, linewidth=2)
    ax.add_patch(pass_seat)
    ax.text(1.65, 1.35, 'Passenger\nSeat', ha='center', va='center',
            fontsize=9, color='white', fontweight='bold')

    # Center console
    console = FancyBboxPatch((1.0, 0.78), 1.2, 0.29,
                              boxstyle="round,pad=0.02",
                              facecolor='#444', edgecolor='#333', alpha=0.5)
    ax.add_patch(console)

    # Rear seats
    rear_seats = FancyBboxPatch((2.9, 0.2), 1.0, width-0.4,
                                 boxstyle="round,pad=0.03",
                                 facecolor='#A0522D', edgecolor='#5D3A1A',
                                 alpha=0.5, linewidth=2)
    ax.add_patch(rear_seats)
    ax.text(3.4, width/2, 'Rear Seats', ha='center', va='center',
            fontsize=9, color='white', fontweight='bold')

    # Steering wheel
    steering = Circle((0.85, 0.5), 0.15, facecolor='#333',
                       edgecolor='#222', linewidth=2)
    ax.add_patch(steering)

    # --- Plot components ---

    # Noise source (red square)
    ax.scatter(noise_source[0], noise_source[1], s=400, c='#e74c3c', marker='s',
               edgecolors='white', linewidths=2, zorder=20, label='Noise Source')
    ax.annotate('NOISE\nSOURCE', (noise_source[0], noise_source[1]),
                textcoords="offset points", xytext=(25, 0),
                ha='left', fontsize=9, color='#c0392b', fontweight='bold')

    # Reference mic (blue triangle)
    ax.scatter(REF_MIC[0], REF_MIC[1], s=300, c='#3498db', marker='^',
               edgecolors='white', linewidths=2, zorder=20, label='Reference Mic')
    ax.annotate('REF\nMIC', (REF_MIC[0], REF_MIC[1]),
                textcoords="offset points", xytext=(0, -25),
                ha='center', fontsize=9, color='#2980b9', fontweight='bold')

    # Error mic / Driver ear (yellow star)
    ax.scatter(ERROR_MIC[0], ERROR_MIC[1], s=400, c='#f1c40f', marker='*',
               edgecolors='#333', linewidths=1.5, zorder=25, label='Error Mic (Driver Ear)')
    ax.annotate('ERROR MIC\n(Driver Ear)', (ERROR_MIC[0], ERROR_MIC[1]),
                textcoords="offset points", xytext=(20, 15),
                ha='left', fontsize=9, color='#333', fontweight='bold')

    # Speakers (green circles)
    speakers = config['speakers']
    for i, (name, pos) in enumerate(speakers.items()):
        ax.scatter(pos[0], pos[1], s=350, c='#2ecc71', marker='o',
                   edgecolors='white', linewidths=2, zorder=20,
                   label='Speaker' if i == 0 else None)

        # Label position adjustment
        if 'door_L' in name or 'headrest' in name:
            offset = (0, -20)
        elif 'door_R' in name:
            offset = (0, 20)
        elif 'dash_L' in name:
            offset = (-15, -20)
        elif 'dash_R' in name:
            offset = (-15, 20)
        else:
            offset = (0, 15)

        label = name.upper().replace('_', '\n')
        ax.annotate(label, (pos[0], pos[1]),
                    textcoords="offset points", xytext=offset,
                    ha='center', fontsize=8, color='#27ae60', fontweight='bold')

    # Draw signal paths (dashed lines)
    # Noise to error mic (primary path)
    ax.annotate('', xy=(ERROR_MIC[0], ERROR_MIC[1]),
                xytext=(noise_source[0], noise_source[1]),
                arrowprops=dict(arrowstyle='->', color='#e74c3c',
                               lw=2, ls='--', alpha=0.6))

    # Speakers to error mic (secondary path)
    for pos in speakers.values():
        ax.annotate('', xy=(ERROR_MIC[0], ERROR_MIC[1]),
                    xytext=(pos[0], pos[1]),
                    arrowprops=dict(arrowstyle='->', color='#2ecc71',
                                   lw=1.5, ls='--', alpha=0.5))

    # Legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    # Labels and title
    ax.set_xlim(-0.3, length + 0.3)
    ax.set_ylim(-0.3, width + 0.3)
    ax.set_aspect('equal')
    ax.set_xlabel('Length (m)', fontsize=11)
    ax.set_ylabel('Width (m)', fontsize=11)

    title = f"{config['name']} - {scenario_name}\n"
    title += f"Noise Reduction: {nr_db:.1f} dB"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # Add info box
    n_speakers = len(speakers)
    info_text = f"Speakers: {n_speakers}\n"
    info_text += f"Filter: 256 taps\n"
    info_text += f"Step size: 0.005"

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_comparison_bar(results: dict, save_path: str):
    """Plot bar chart comparing configurations across scenarios."""

    fig, ax = plt.subplots(figsize=(12, 6))

    scenarios = list(SCENARIOS.keys())
    configs = list(CONFIGS.keys())

    x = np.arange(len(scenarios))
    width = 0.35

    for i, config_key in enumerate(configs):
        config = CONFIGS[config_key]
        values = [results[s][config_key]['nr_db'] for s in scenarios]
        offset = width * (i - 0.5)
        bars = ax.bar(x + offset, values, width, label=config['name'],
                      color=config['color'], edgecolor='white', linewidth=1.5)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Noise Scenario', fontsize=12)
    ax.set_ylabel('Noise Reduction (dB)', fontsize=12)
    ax.set_title('Single Headrest Speaker vs 4-Speaker Stereo\nNoise Reduction Comparison',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIOS[s]['name'] for s in scenarios], fontsize=11)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max([max([results[s][c]['nr_db'] for c in configs]) for s in scenarios]) + 3)

    # Add winner annotation
    for i, scenario in enumerate(scenarios):
        single = results[scenario]['single_headrest']['nr_db']
        quad = results[scenario]['quad_stereo']['nr_db']
        winner = "Single" if single > quad else "4-Speaker"
        diff = abs(single - quad)

        y_pos = max(single, quad) + 1.5
        ax.text(i, y_pos, f'{winner}\nwins by\n{diff:.1f} dB',
                ha='center', va='bottom', fontsize=8, color='#333',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_summary(results: dict, save_path: str):
    """Plot summary answering the main question."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Average comparison
    ax1 = axes[0]

    avg_single = np.mean([results[s]['single_headrest']['nr_db'] for s in SCENARIOS])
    avg_quad = np.mean([results[s]['quad_stereo']['nr_db'] for s in SCENARIOS])

    bars = ax1.bar(['Single Headrest\nSpeaker', '4-Speaker\nStereo'],
                   [avg_single, avg_quad],
                   color=[CONFIGS['single_headrest']['color'], CONFIGS['quad_stereo']['color']],
                   edgecolor='white', linewidth=2, width=0.5)

    for bar, val in zip(bars, [avg_single, avg_quad]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f} dB', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax1.set_ylabel('Average Noise Reduction (dB)', fontsize=12)
    ax1.set_title('Average Performance\n(Across All Scenarios)', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, max(avg_single, avg_quad) + 3)
    ax1.grid(axis='y', alpha=0.3)

    # Right: Answer the question
    ax2 = axes[1]
    ax2.axis('off')

    diff = avg_single - avg_quad

    if diff > 0.5:
        answer = "SINGLE HEADREST SPEAKER\nIS BETTER"
        answer_color = CONFIGS['single_headrest']['color']
        explanation = f"By {diff:.1f} dB on average"
    elif diff < -0.5:
        answer = "4-SPEAKER STEREO\nIS BETTER"
        answer_color = CONFIGS['quad_stereo']['color']
        explanation = f"By {abs(diff):.1f} dB on average"
    else:
        answer = "ROUGHLY EQUAL\nPERFORMANCE"
        answer_color = '#95a5a6'
        explanation = f"Difference: {abs(diff):.1f} dB"

    ax2.text(0.5, 0.7, "Is 4-speaker better than\na well-placed single speaker?",
             ha='center', va='center', fontsize=14, transform=ax2.transAxes)

    ax2.text(0.5, 0.4, answer, ha='center', va='center', fontsize=20,
             fontweight='bold', color=answer_color, transform=ax2.transAxes)

    ax2.text(0.5, 0.2, explanation, ha='center', va='center', fontsize=12,
             color='#666', transform=ax2.transAxes)

    # Per-scenario breakdown
    breakdown = ""
    for s in SCENARIOS:
        single = results[s]['single_headrest']['nr_db']
        quad = results[s]['quad_stereo']['nr_db']
        winner = "Single" if single > quad else "4-Spk"
        breakdown += f"{SCENARIOS[s]['name']}: {winner} (+{abs(single-quad):.1f} dB)\n"

    ax2.text(0.5, 0.02, breakdown, ha='center', va='bottom', fontsize=10,
             color='#444', transform=ax2.transAxes, family='monospace')

    plt.suptitle('Step 8: Speaker Configuration Comparison',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("Step 8: Single Speaker vs 4-Speaker ANC Comparison")
    print("=" * 70)
    print("\nQuestion: Is 4-speaker better than a well-placed single speaker?")
    print("\nConfigurations:")
    print("  1. Single headrest speaker (optimal single-speaker location)")
    print("  2. 4-speaker stereo (door + dashboard, both sides)")
    print("\nScenarios: Engine (idle), Road (city), Highway (combined)")

    # Create output directories
    save_dir = 'output/plots'
    data_dir = 'output/data'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Run simulations
    results = {}
    noise_gen = NoiseMixer(16000)
    duration = 3.0

    for scenario_key, scenario in SCENARIOS.items():
        print(f"\n{'='*50}")
        print(f"Scenario: {scenario['name']}")
        print(f"{'='*50}")

        results[scenario_key] = {}
        noise_signal = noise_gen.generate_scenario(duration, scenario['type'])

        for config_key, config in CONFIGS.items():
            print(f"\n  Testing: {config['name']}...")

            sim = ANCSimulator(
                speakers=config['speakers'],
                noise_source=scenario['source'],
            )
            result = sim.run(noise_signal)

            nr_db = result['noise_reduction_db']
            results[scenario_key][config_key] = {'nr_db': nr_db}
            print(f"    Noise Reduction: {nr_db:.1f} dB")

            # Plot car interior
            plot_car_interior(
                config_key, config, scenario['source'],
                scenario['name'], nr_db,
                f"{save_dir}/step8_{scenario_key}_{config_key}.png"
            )

    # Generate comparison plots
    print(f"\n{'='*50}")
    print("Generating comparison plots...")
    print(f"{'='*50}")

    plot_comparison_bar(results, f"{save_dir}/step8_comparison_bar.png")
    plot_summary(results, f"{save_dir}/step8_summary.png")

    # Print results
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Scenario':<20} {'Single Headrest':<18} {'4-Speaker':<15} {'Winner':<15}")
    print("-" * 70)

    total_single = 0
    total_quad = 0

    for scenario_key in SCENARIOS:
        single = results[scenario_key]['single_headrest']['nr_db']
        quad = results[scenario_key]['quad_stereo']['nr_db']
        winner = "Single" if single > quad else "4-Speaker"
        diff = abs(single - quad)

        print(f"{SCENARIOS[scenario_key]['name']:<20} {single:>10.1f} dB      {quad:>8.1f} dB      "
              f"{winner} (+{diff:.1f})")

        total_single += single
        total_quad += quad

    avg_single = total_single / len(SCENARIOS)
    avg_quad = total_quad / len(SCENARIOS)

    print("-" * 70)
    print(f"{'AVERAGE':<20} {avg_single:>10.1f} dB      {avg_quad:>8.1f} dB")
    print("=" * 70)

    # Answer the question
    print("\n" + "=" * 70)
    print("ANSWER: Is 4-speaker better than a well-placed single speaker?")
    print("=" * 70)

    diff = avg_single - avg_quad
    if diff > 0.5:
        print(f"\n  NO - Single headrest speaker is BETTER by {diff:.1f} dB on average")
        print("  The headrest speaker's proximity to the ear provides an advantage.")
    elif diff < -0.5:
        print(f"\n  YES - 4-speaker stereo is BETTER by {abs(diff):.1f} dB on average")
        print("  Multiple speakers provide better spatial coverage.")
    else:
        print(f"\n  ROUGHLY EQUAL - Difference is only {abs(diff):.1f} dB")
        print("  A well-placed single speaker can match 4-speaker performance.")

    print("\n" + "=" * 70)

    # Save results to JSON
    json_results = {
        'timestamp': datetime.now().isoformat(),
        'question': 'Is 4-speaker better than a well-placed single speaker?',
        'answer': 'Yes' if diff < -0.5 else ('No' if diff > 0.5 else 'Equal'),
        'average_difference_db': float(diff),
        'results': results,
    }

    with open(f'{data_dir}/step8_comparison_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {data_dir}/step8_comparison_results.json")

    # List output files
    print("\nOutput files:")
    print(f"  {save_dir}/step8_<scenario>_<config>.png  - Car interior diagrams")
    print(f"  {save_dir}/step8_comparison_bar.png       - Bar chart comparison")
    print(f"  {save_dir}/step8_summary.png              - Summary with answer")

    return results


if __name__ == '__main__':
    results = main()
