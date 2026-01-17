"""
Step 8: Microphone Placement Optimization for Car ANC

Goal: Find optimal reference and error microphone locations when using
      existing car stereo speakers for anti-noise generation.

This simulation:
1. Tests multiple speaker positions (existing car stereo speakers)
2. Tests multiple reference microphone locations
3. Tests multiple error microphone locations
4. Runs across different driving scenarios
5. Generates heatmaps and recommendations

The key question: Given a car's existing stereo speakers, where should we
place the reference and error microphones for best ANC performance?

Target frequency range: 20-300 Hz
Expected output: Optimal placement recommendations per scenario
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyroomacoustics as pra
from src.acoustic.room_builder import RoomBuilder
from src.acoustic.path_generator import AcousticPathGenerator, FIRPath
from src.core.fxlms import FxNLMS
from src.noise.noise_mixer import NoiseMixer
from src.utils.audio import save_wav, save_comparison_wav
from src.placement.microphone_config import (
    SPEAKER_POSITIONS,
    REF_MIC_POSITIONS,
    ERROR_MIC_POSITIONS,
    DRIVING_SCENARIOS,
    SEDAN_DIMENSIONS,
    SEDAN_MATERIALS,
    get_placement_config,
    get_subset_combinations,
    get_all_placement_combinations,
    calculate_distances,
    validate_placement,
)


class PlacementTester:
    """
    Tests ANC performance across different placement configurations.
    """

    def __init__(self, fs: int = 16000, duration: float = 3.0):
        """
        Initialize placement tester.

        Args:
            fs: Sample rate
            duration: Test duration per configuration (shorter for sweep)
        """
        self.fs = fs
        self.duration = duration
        self.results = []
        self.noise_gen = NoiseMixer(fs)

    def run_single_test(self, config: dict, verbose: bool = False) -> dict:
        """
        Run ANC simulation for a single placement configuration.

        Args:
            config: Placement configuration from get_placement_config()
            verbose: Print progress

        Returns:
            Results dictionary with performance metrics
        """
        room_cfg = config['room']
        pos = config['positions']
        fxlms_cfg = config['fxlms']
        scenario = config.get('scenario', 'highway')

        # Validate placement
        is_valid, msg = validate_placement(config)
        if not is_valid:
            if verbose:
                print(f"  Skipping invalid config: {msg}")
            return {
                'success': False,
                'error': msg,
                'noise_reduction_db': -999,
                'convergence_time_s': -1,
                'final_mse': -1,
            }

        # Create room
        pra_materials = {
            'ceiling': pra.Material(room_cfg['materials']['ceiling']),
            'floor': pra.Material(room_cfg['materials']['floor']),
            'east': pra.Material(room_cfg['materials']['east']),
            'west': pra.Material(room_cfg['materials']['west']),
            'north': pra.Material(room_cfg['materials']['north']),
            'south': pra.Material(room_cfg['materials']['south']),
        }

        room = pra.ShoeBox(
            room_cfg['dimensions'],
            fs=self.fs,
            materials=pra_materials,
            max_order=room_cfg['max_order'],
            air_absorption=True
        )

        # Add sources and mics
        room.add_source(pos['noise_source'])
        room.add_source(pos['speaker'])
        mic_array = np.array([pos['reference_mic'], pos['error_mic']]).T
        room.add_microphone_array(pra.MicrophoneArray(mic_array, fs=self.fs))

        # Compute RIRs
        room.compute_rir()

        # Extract paths
        path_gen = AcousticPathGenerator(room)
        paths = path_gen.get_all_anc_paths(modeling_error=0.05)

        max_len = 512
        H_primary = paths['primary'][:max_len]
        H_secondary = paths['secondary'][:max_len]
        H_secondary_est = paths['secondary_estimate'][:max_len]
        H_reference = paths['reference'][:max_len]

        # Create filters
        primary_path = FIRPath(H_primary)
        secondary_path = FIRPath(H_secondary)
        reference_path = FIRPath(H_reference)

        fxlms = FxNLMS(
            filter_length=fxlms_cfg['filter_length'],
            step_size=fxlms_cfg['step_size'],
            secondary_path_estimate=H_secondary_est,
            regularization=1e-4
        )

        # Generate noise
        noise_source = self.noise_gen.generate_scenario(self.duration, scenario)
        n_samples = len(noise_source)

        # Run simulation
        desired = []
        error = []
        mse = []

        for i in range(n_samples):
            sample = noise_source[i]

            x = reference_path.filter_sample(sample)
            d = primary_path.filter_sample(sample)
            desired.append(d)

            y = fxlms.generate_antinoise(x)
            y_at_error = secondary_path.filter_sample(y)

            e = d + y_at_error
            error.append(e)
            mse.append(e ** 2)

            fxlms.filter_reference(x)
            fxlms.update_weights(e)

        # Calculate metrics
        desired = np.array(desired)
        error = np.array(error)
        mse = np.array(mse)

        # Steady-state noise reduction (last 50%)
        steady_start = len(desired) // 2
        d_power = np.mean(desired[steady_start:]**2)
        e_power = np.mean(error[steady_start:]**2)

        if e_power > 1e-10:
            nr_db = 10 * np.log10(d_power / e_power)
        else:
            nr_db = 60.0

        # Convergence time (time to reach 90% of final reduction)
        window = 500
        mse_smooth = np.convolve(mse, np.ones(window)/window, mode='valid')
        final_mse = np.mean(mse_smooth[-window:])
        target_mse = final_mse + 0.1 * (mse_smooth[0] - final_mse)  # 90% of improvement

        conv_idx = np.argmax(mse_smooth < target_mse)
        conv_time = conv_idx / self.fs if conv_idx > 0 else self.duration

        # Distance metrics
        distances = calculate_distances(config)

        return {
            'success': True,
            'noise_reduction_db': nr_db,
            'convergence_time_s': conv_time,
            'final_mse': final_mse,
            'time_budget_ms': distances['time_budget_ms'],
            'primary_distance_m': distances['primary_distance_m'],
            'secondary_distance_m': distances['secondary_distance_m'],
        }

    def run_sweep(
        self,
        combinations: List[Tuple[str, str, str, str]],
        verbose: bool = True,
        save_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Run ANC tests for all placement combinations.

        Args:
            combinations: List of (speaker, ref_mic, error_mic, scenario) tuples
            verbose: Print progress
            save_progress: Save results incrementally

        Returns:
            DataFrame with all results
        """
        total = len(combinations)
        print(f"\nRunning placement sweep: {total} configurations")
        print("=" * 60)

        results = []
        start_time = datetime.now()

        for i, (speaker, ref_mic, error_mic, scenario) in enumerate(combinations):
            if verbose and i % 10 == 0:
                elapsed = (datetime.now() - start_time).seconds
                rate = (i + 1) / max(elapsed, 1)
                remaining = (total - i - 1) / rate if rate > 0 else 0
                print(f"[{i+1}/{total}] {speaker} | {ref_mic} | {error_mic} | {scenario}")
                print(f"    Elapsed: {elapsed}s, Remaining: ~{remaining:.0f}s")

            config = get_placement_config(speaker, ref_mic, error_mic, scenario)
            result = self.run_single_test(config, verbose=False)

            result.update({
                'speaker': speaker,
                'ref_mic': ref_mic,
                'error_mic': error_mic,
                'scenario': scenario,
            })

            results.append(result)

            # Save progress periodically
            if save_progress and (i + 1) % 50 == 0:
                df = pd.DataFrame(results)
                df.to_csv('output/data/pyroom_step8_sweep_progress.csv', index=False)

        self.results = results
        df = pd.DataFrame(results)

        # Save final results
        os.makedirs('output/data', exist_ok=True)
        df.to_csv('output/data/pyroom_step8_sweep_results.csv', index=False)

        print(f"\nSweep complete! Tested {len(df)} configurations")
        print(f"Results saved to: output/data/pyroom_step8_sweep_results.csv")

        return df

    def analyze_results(self, df: pd.DataFrame = None) -> dict:
        """
        Analyze sweep results to find optimal placements.

        Returns:
            Analysis dictionary with rankings and recommendations
        """
        if df is None:
            df = pd.DataFrame(self.results)

        # Filter successful results
        df_valid = df[df['success'] == True].copy()

        if len(df_valid) == 0:
            return {'error': 'No valid results to analyze'}

        analysis = {}

        # Best overall
        best_idx = df_valid['noise_reduction_db'].idxmax()
        best = df_valid.loc[best_idx]
        analysis['best_overall'] = {
            'speaker': best['speaker'],
            'ref_mic': best['ref_mic'],
            'error_mic': best['error_mic'],
            'scenario': best['scenario'],
            'noise_reduction_db': best['noise_reduction_db'],
        }

        # Best per scenario
        analysis['best_per_scenario'] = {}
        for scenario in df_valid['scenario'].unique():
            df_scenario = df_valid[df_valid['scenario'] == scenario]
            best_idx = df_scenario['noise_reduction_db'].idxmax()
            best = df_scenario.loc[best_idx]
            analysis['best_per_scenario'][scenario] = {
                'speaker': best['speaker'],
                'ref_mic': best['ref_mic'],
                'error_mic': best['error_mic'],
                'noise_reduction_db': best['noise_reduction_db'],
            }

        # Average performance by speaker
        speaker_avg = df_valid.groupby('speaker')['noise_reduction_db'].mean().sort_values(ascending=False)
        analysis['speaker_ranking'] = speaker_avg.to_dict()

        # Average performance by ref_mic
        ref_mic_avg = df_valid.groupby('ref_mic')['noise_reduction_db'].mean().sort_values(ascending=False)
        analysis['ref_mic_ranking'] = ref_mic_avg.to_dict()

        # Average performance by error_mic
        error_mic_avg = df_valid.groupby('error_mic')['noise_reduction_db'].mean().sort_values(ascending=False)
        analysis['error_mic_ranking'] = error_mic_avg.to_dict()

        return analysis


def plot_results_heatmap(df: pd.DataFrame, scenario: str = 'highway', save_path: str = None):
    """
    Create heatmap of ANC performance for speaker vs microphone positions.
    """
    df_scenario = df[(df['scenario'] == scenario) & (df['success'] == True)]

    if len(df_scenario) == 0:
        print(f"No valid results for scenario: {scenario}")
        return

    # Pivot table: speaker x ref_mic (averaging over error_mic)
    pivot = df_scenario.pivot_table(
        values='noise_reduction_db',
        index='speaker',
        columns='ref_mic',
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')

    # Labels
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel('Reference Microphone')
    ax.set_ylabel('Speaker')
    ax.set_title(f'Noise Reduction (dB) - {scenario.title()} Scenario\n(Averaged over error mic positions)')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Noise Reduction (dB)')

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = 'white' if val < pivot.values.mean() else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', color=text_color, fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_speaker_comparison(df: pd.DataFrame, save_path: str = None):
    """
    Bar chart comparing average performance by speaker position.
    """
    df_valid = df[df['success'] == True]

    speaker_perf = df_valid.groupby(['speaker', 'scenario'])['noise_reduction_db'].mean().unstack()

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(speaker_perf.index))
    width = 0.2
    scenarios = speaker_perf.columns

    for i, scenario in enumerate(scenarios):
        offset = (i - len(scenarios)/2 + 0.5) * width
        bars = ax.bar(x + offset, speaker_perf[scenario], width, label=scenario.title())

    ax.set_xlabel('Speaker Position')
    ax.set_ylabel('Noise Reduction (dB)')
    ax.set_title('ANC Performance by Speaker Position')
    ax.set_xticks(x)
    ax.set_xticklabels(speaker_perf.index, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_top_configurations(df: pd.DataFrame, n_top: int = 10, save_path: str = None):
    """
    Bar chart of top N best performing configurations.
    """
    df_valid = df[df['success'] == True].copy()
    df_top = df_valid.nlargest(n_top, 'noise_reduction_db')

    fig, ax = plt.subplots(figsize=(12, 6))

    labels = [f"{row['speaker']}\n{row['ref_mic']}\n{row['error_mic']}\n({row['scenario']})"
              for _, row in df_top.iterrows()]

    colors = plt.cm.viridis(np.linspace(0.8, 0.2, n_top))

    bars = ax.barh(range(n_top), df_top['noise_reduction_db'], color=colors)

    ax.set_yticks(range(n_top))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Noise Reduction (dB)')
    ax.set_title(f'Top {n_top} Placement Configurations')
    ax.invert_yaxis()

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df_top['noise_reduction_db'])):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f} dB', va='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()


def main():
    """
    Run complete placement optimization study.
    """
    print("=" * 70)
    print("Step 8: Microphone Placement Optimization")
    print("=" * 70)
    print()

    # Use same output structure as other steps
    os.makedirs('output/plots', exist_ok=True)
    os.makedirs('output/audio', exist_ok=True)
    os.makedirs('output/data', exist_ok=True)

    # Create tester
    tester = PlacementTester(fs=16000, duration=3.0)

    # Get subset of combinations for reasonable runtime
    # Full sweep would be: 9 speakers × 9 ref_mics × 9 error_mics × 4 scenarios = 2916 tests
    # Subset: 4 speakers × 3 ref_mics × 2 error_mics × 3 scenarios = 72 tests
    print("\nConfiguration:")
    print(f"  Speakers to test: 4 (most promising)")
    print(f"  Reference mics: 3 (best positions)")
    print(f"  Error mics: 2 (driver positions)")
    print(f"  Scenarios: 3 (highway, city, acceleration)")
    print()

    combinations = get_subset_combinations(
        n_speakers=4,
        n_ref_mics=3,
        n_error_mics=2,
        scenarios=['highway', 'city', 'acceleration']
    )

    print(f"Total configurations to test: {len(combinations)}")
    print()

    # Run sweep
    df = tester.run_sweep(combinations, verbose=True)

    # Analyze results
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)

    analysis = tester.analyze_results(df)

    if 'error' not in analysis:
        print("\n📊 BEST OVERALL CONFIGURATION:")
        best = analysis['best_overall']
        print(f"  Speaker: {best['speaker']}")
        print(f"  Reference Mic: {best['ref_mic']}")
        print(f"  Error Mic: {best['error_mic']}")
        print(f"  Scenario: {best['scenario']}")
        print(f"  Noise Reduction: {best['noise_reduction_db']:.1f} dB")

        print("\n📊 BEST PER SCENARIO:")
        for scenario, config in analysis['best_per_scenario'].items():
            print(f"\n  {scenario.upper()}:")
            print(f"    Speaker: {config['speaker']}")
            print(f"    Reference Mic: {config['ref_mic']}")
            print(f"    Error Mic: {config['error_mic']}")
            print(f"    Reduction: {config['noise_reduction_db']:.1f} dB")

        print("\n📊 SPEAKER RANKING (by average reduction):")
        for i, (speaker, nr) in enumerate(analysis['speaker_ranking'].items(), 1):
            print(f"  {i}. {speaker}: {nr:.1f} dB")

        print("\n📊 REFERENCE MIC RANKING:")
        for i, (mic, nr) in enumerate(analysis['ref_mic_ranking'].items(), 1):
            print(f"  {i}. {mic}: {nr:.1f} dB")

    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    for scenario in ['highway', 'city', 'acceleration']:
        plot_results_heatmap(df, scenario, f'output/plots/pyroom_step8_heatmap_{scenario}.png')

    plot_speaker_comparison(df, 'output/plots/pyroom_step8_speaker_comparison.png')
    plot_top_configurations(df, 10, 'output/plots/pyroom_step8_top_configurations.png')

    # Save analysis
    with open('output/data/pyroom_step8_analysis.json', 'w') as f:
        # Convert numpy types for JSON serialization
        analysis_json = json.loads(json.dumps(analysis, default=str))
        json.dump(analysis_json, f, indent=2)
    print("\nAnalysis saved to: output/data/pyroom_step8_analysis.json")

    print("\n" + "=" * 70)
    print("PLACEMENT OPTIMIZATION COMPLETE")
    print("=" * 70)
    print("\nOutput files:")
    print("  Results: output/data/pyroom_step8_sweep_results.csv")
    print("  Analysis: output/data/pyroom_step8_analysis.json")
    print("  Heatmaps: output/plots/pyroom_step8_heatmap_*.png")
    print("  Rankings: output/plots/pyroom_step8_speaker_comparison.png")
    print("  Top configs: output/plots/pyroom_step8_top_configurations.png")

    return df, analysis


if __name__ == '__main__':
    df, analysis = main()
