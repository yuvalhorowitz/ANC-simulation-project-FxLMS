"""
Run Optimized FxLMS ANC Simulation

Main entry point for running optimized simulations with:
- Optimal filter length (based on RT60)
- Optimal max_order (based on room dimensions)
- Both single speaker AND 4-speaker stereo configurations

Each scenario is tested with both speaker configurations for comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pyroomacoustics as pra
from src.acoustic.path_generator import AcousticPathGenerator, FIRPath
from src.core.fxlms import FxNLMS
from src.noise.noise_mixer import NoiseMixer
from src.utils.audio import save_wav, save_comparison_wav

from .optimized_configurations import (
    get_config, get_all_configs, SpeakerConfig, OptimizedConfig,
    get_speaker_positions, print_config_summary
)
from .rir_optimization import truncate_rir, optimal_rir_length
from .low_freq_materials import create_pra_material, MATERIAL_COEFFICIENTS


class OptimizedCarANC:
    """
    Optimized car interior ANC system with configurable speaker setup.

    Supports both single speaker and 4-speaker stereo configurations.
    """

    def __init__(
        self,
        config: OptimizedConfig,
        speaker_config: SpeakerConfig = SpeakerConfig.SINGLE,
        fs: int = 16000
    ):
        """
        Initialize optimized ANC system.

        Args:
            config: OptimizedConfig from optimized_configurations
            speaker_config: SINGLE or QUAD_STEREO
            fs: Sample rate
        """
        self.fs = fs
        self.config = config
        self.speaker_config = speaker_config

        print(f"\n{'='*60}")
        print(f"Initializing Optimized ANC: {config.name}")
        print(f"Speaker Configuration: {speaker_config.value}")
        print(f"{'='*60}")

        # Create room
        self.room = self._create_room()

        # Get speaker positions based on configuration
        speaker_positions = get_speaker_positions(config, speaker_config)

        # Add noise source
        self.room.add_source(config.noise_source)

        # Add speakers (one or four depending on config)
        for sp_pos in speaker_positions:
            self.room.add_source(sp_pos)

        # Add microphones
        mic_positions = np.array([
            config.reference_mic.position,
            config.error_mic.position
        ]).T
        self.room.add_microphone_array(pra.MicrophoneArray(mic_positions, fs=fs))

        # Compute RIRs
        self.room.compute_rir()

        # Extract and optimize paths
        self._setup_acoustic_paths()

        # Create FxNLMS filter
        # For quad_stereo: use single filter with combined secondary path
        # (more stable than 4 independent filters)
        self.fxlms = FxNLMS(
            filter_length=config.filter_length,
            step_size=config.step_size,
            secondary_path_estimate=self.H_secondary_est,
            regularization=1e-4
        )

        # Noise generator
        self.noise_gen = NoiseMixer(fs)

        # Results storage
        self.results = {}

        # Print summary
        self._print_setup_summary()

    def _create_room(self) -> pra.ShoeBox:
        """Create room with optimized materials."""
        dims = self.config.room.dimensions

        # Create pra materials
        pra_materials = {}

        # Map our material names to pra wall names
        wall_mapping = {
            'ceiling': 'ceiling',
            'floor': 'floor',
            'front': 'east',
            'back': 'west',
            'left': 'south',
            'right': 'north',
        }

        # Octave band center frequencies for frequency-dependent materials
        center_freqs = [125, 250, 500, 1000, 2000, 4000]

        for our_name, pra_name in wall_mapping.items():
            mat_name = self.config.room.materials.get(our_name, 'dashboard_plastic')
            if mat_name in MATERIAL_COEFFICIENTS:
                mat_data = MATERIAL_COEFFICIENTS[mat_name]
                # pyroomacoustics expects dict format for frequency-dependent absorption
                pra_materials[pra_name] = pra.Material(
                    energy_absorption={
                        'coeffs': mat_data['absorption'],
                        'center_freqs': center_freqs
                    },
                    scattering=mat_data['scattering']
                )
            else:
                # Fallback to simple absorption
                pra_materials[pra_name] = pra.Material(0.1)

        room = pra.ShoeBox(
            dims,
            fs=self.fs,
            materials=pra_materials,
            max_order=self.config.max_order,
            air_absorption=True
        )

        return room

    def _setup_acoustic_paths(self):
        """Setup and optimize acoustic paths."""
        # Source 0 = Noise source
        # Source 1+ = Speakers

        # Primary path: Noise source -> Error mic (mic index 1)
        rir_primary = self.room.rir[1][0]
        opt_len = optimal_rir_length(rir_primary)
        self.H_primary = truncate_rir(rir_primary, opt_len)

        # Reference path: Noise source -> Reference mic (mic index 0)
        rir_reference = self.room.rir[0][0]
        self.H_reference = truncate_rir(rir_reference, opt_len)

        if self.speaker_config == SpeakerConfig.SINGLE:
            # Single speaker: Source 1 -> Error mic
            rir_secondary = self.room.rir[1][1]
            self.H_secondary = truncate_rir(rir_secondary, opt_len)
            self.H_secondary_est = self.H_secondary * (1.0 + 0.05 * np.random.randn(len(self.H_secondary)))

        else:
            # Quad stereo: Sources 1-4 -> Error mic
            # Combined secondary path (sum of all speakers)
            self.H_secondary_channels = []
            self.H_secondary_est_channels = []

            combined_secondary = np.zeros(opt_len)

            for i in range(4):
                rir = self.room.rir[1][i + 1]  # Sources 1-4
                h = truncate_rir(rir, opt_len)
                self.H_secondary_channels.append(h)

                # Estimate with modeling error
                h_est = h * (1.0 + 0.05 * np.random.randn(len(h)))
                self.H_secondary_est_channels.append(h_est)

                combined_secondary += h

            # Normalize combined
            self.H_secondary = combined_secondary / 4.0
            self.H_secondary_est = self.H_secondary * (1.0 + 0.05 * np.random.randn(len(self.H_secondary)))

        # Create FIR filter objects
        self.primary_path = FIRPath(self.H_primary)
        self.secondary_path = FIRPath(self.H_secondary)
        self.reference_path = FIRPath(self.H_reference)

        if self.speaker_config == SpeakerConfig.QUAD_STEREO:
            self.secondary_paths = [FIRPath(h) for h in self.H_secondary_channels]

    def _print_setup_summary(self):
        """Print configuration summary."""
        print(f"\nSystem Configuration:")
        print(f"  Room: {self.config.room.dimensions} m")
        print(f"  RT60: {self.config.room.rt60:.3f} s")
        print(f"  Max order: {self.config.max_order}")
        print(f"  Filter length: {self.config.filter_length} taps ({self.config.filter_length/self.fs*1000:.1f} ms)")
        print(f"  Step size: {self.config.step_size}")
        print(f"  Primary path: {len(self.H_primary)} taps")
        print(f"  Secondary path: {len(self.H_secondary)} taps")

        n_speakers = 1 if self.speaker_config == SpeakerConfig.SINGLE else 4
        print(f"  Speakers: {n_speakers}")

    def run_scenario(
        self,
        duration: float = 5.0,
        scenario: str = 'highway',
        verbose: bool = True
    ) -> dict:
        """
        Run ANC simulation for a driving scenario.

        Args:
            duration: Simulation duration in seconds
            scenario: Noise scenario ('idle', 'city', 'highway')
            verbose: Print progress

        Returns:
            Dictionary with simulation results
        """
        print(f"\nRunning {scenario} scenario ({duration}s)...")
        print(f"Speaker config: {self.speaker_config.value}")

        # Generate noise
        noise_source = self.noise_gen.generate_scenario(duration, scenario)
        n_samples = len(noise_source)

        # Reset filters
        self.fxlms.reset()
        self.primary_path.reset()
        self.secondary_path.reset()
        self.reference_path.reset()

        if self.speaker_config == SpeakerConfig.QUAD_STEREO:
            for path in self.secondary_paths:
                path.reset()

        # Storage
        reference = []
        desired = []
        antinoise = []
        error = []
        mse = []

        for i in range(n_samples):
            sample = noise_source[i]

            # Reference signal
            x = self.reference_path.filter_sample(sample)
            reference.append(x)

            # Noise at error mic
            d = self.primary_path.filter_sample(sample)
            desired.append(d)

            # Generate anti-noise (same filter for both configs)
            y = self.fxlms.generate_antinoise(x)
            antinoise.append(y)

            if self.speaker_config == SpeakerConfig.SINGLE:
                # Single speaker mode
                y_at_error = self.secondary_path.filter_sample(y)
            else:
                # Quad stereo mode: same signal to all 4 speakers
                # Combined effect at error mic is sum of all paths
                y_at_error = 0.0
                for path in self.secondary_paths:
                    y_at_error += path.filter_sample(y)

            # Error signal
            e = d + y_at_error
            error.append(e)
            mse.append(e ** 2)

            # Update weights (same for both configs)
            self.fxlms.filter_reference(x)
            self.fxlms.update_weights(e)

            # Progress
            if verbose and (i + 1) % (n_samples // 10) == 0:
                progress = (i + 1) / n_samples * 100
                current_mse = np.mean(mse[-1000:]) if len(mse) > 1000 else np.mean(mse)
                print(f"  Progress: {progress:.0f}% | MSE: {current_mse:.6f}")

        # Calculate noise reduction
        steady_start = len(desired) // 2
        d_power = np.mean(np.array(desired[steady_start:])**2)
        e_power = np.mean(np.array(error[steady_start:])**2)

        if e_power > 1e-10:
            nr_db = 10 * np.log10(d_power / e_power)
        else:
            nr_db = 60.0

        # Store results
        self.results = {
            'reference': np.array(reference),
            'desired': np.array(desired),
            'antinoise': np.array(antinoise),
            'error': np.array(error),
            'mse': np.array(mse),
            'scenario': scenario,
            'duration': duration,
            'noise_reduction_db': nr_db,
            'speaker_config': self.speaker_config.value,
            'config_name': self.config.name,
            'filter_length': self.config.filter_length,
            'max_order': self.config.max_order,
            'step_size': self.config.step_size,
        }

        print(f"\nNoise Reduction: {nr_db:.1f} dB")
        return self.results

    def plot_results(self, save_path: str = None):
        """Generate comprehensive result plots."""
        if not self.results:
            print("No results to plot.")
            return

        t = np.arange(len(self.results['reference'])) / self.fs

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(
            f"{self.config.name} - {self.speaker_config.value} - {self.results['scenario']}",
            fontsize=14, fontweight='bold'
        )

        # Time domain (last 100ms)
        show_samples = int(0.1 * self.fs)
        t_show = t[-show_samples:] * 1000

        # Before/After comparison
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(t_show, self.results['desired'][-show_samples:], 'r-', linewidth=0.8, alpha=0.7, label='Noise')
        ax1.plot(t_show, self.results['error'][-show_samples:], 'g-', linewidth=0.8, alpha=0.7, label='With ANC')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'Time Domain (Reduction: {self.results["noise_reduction_db"]:.1f} dB)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # MSE convergence
        ax2 = fig.add_subplot(2, 2, 2)
        window = 200
        mse_smooth = np.convolve(self.results['mse'], np.ones(window)/window, mode='valid')
        ax2.semilogy(np.arange(len(mse_smooth)) / self.fs, mse_smooth)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('MSE')
        ax2.set_title('Convergence')
        ax2.grid(True, alpha=0.3)

        # Frequency spectrum
        ax3 = fig.add_subplot(2, 2, 3)
        from scipy import signal as scipy_signal

        steady_start = len(self.results['desired']) // 2
        f_d, psd_d = scipy_signal.welch(self.results['desired'][steady_start:], self.fs, nperseg=2048)
        f_e, psd_e = scipy_signal.welch(self.results['error'][steady_start:], self.fs, nperseg=2048)

        ax3.semilogy(f_d, psd_d, 'r-', linewidth=1.5, alpha=0.7, label='Noise')
        ax3.semilogy(f_e, psd_e, 'g-', linewidth=1.5, alpha=0.7, label='With ANC')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('PSD')
        ax3.set_title('Spectrum (20-300 Hz target)')
        ax3.set_xlim(0, 350)
        ax3.axvline(x=20, color='gray', linestyle='--', alpha=0.5)
        ax3.axvline(x=300, color='gray', linestyle='--', alpha=0.5)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Configuration info
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')

        info_text = f"""
Configuration Summary
---------------------
Room: {self.config.room.name}
Dimensions: {self.config.room.dimensions} m
RT60: {self.config.room.rt60:.3f} s

FxLMS Parameters:
  Filter length: {self.config.filter_length} taps
  Step size: {self.config.step_size}
  Max order: {self.config.max_order}

Speaker Config: {self.speaker_config.value}
{'  Single headrest speaker' if self.speaker_config == SpeakerConfig.SINGLE else '  4 door speakers (FL, FR, RL, RR)'}

Results:
  Scenario: {self.results['scenario']}
  Duration: {self.results['duration']:.1f}s
  Noise Reduction: {self.results['noise_reduction_db']:.1f} dB
"""
        ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {save_path}")
        else:
            plt.close()

        return fig


def run_all_configurations(
    scenarios: List[str] = None,
    duration: float = 5.0,
    output_dir: str = 'output/optimized'
) -> Dict:
    """
    Run all configurations with both speaker setups.

    Args:
        scenarios: List of scenarios to test (default: ['idle', 'city', 'highway'])
        duration: Simulation duration in seconds
        output_dir: Output directory for plots and audio

    Returns:
        Dictionary with all results
    """
    if scenarios is None:
        scenarios = ['idle', 'city', 'highway']

    # Create output directories
    os.makedirs(f'{output_dir}/plots', exist_ok=True)
    os.makedirs(f'{output_dir}/audio', exist_ok=True)

    all_results = {}

    # Get all configurations
    configs = get_all_configs()

    for config_name, config in configs.items():
        print(f"\n{'#'*70}")
        print(f"# Configuration: {config_name}")
        print(f"{'#'*70}")

        all_results[config_name] = {}

        # Test both speaker configurations
        for speaker_config in [SpeakerConfig.SINGLE, SpeakerConfig.QUAD_STEREO]:
            speaker_key = speaker_config.value
            all_results[config_name][speaker_key] = {}

            # Create ANC system
            anc = OptimizedCarANC(config, speaker_config)

            # Run each scenario
            for scenario in scenarios:
                results = anc.run_scenario(duration=duration, scenario=scenario)
                all_results[config_name][speaker_key][scenario] = results

                # Save plot
                plot_name = f"{config_name}_{speaker_key}_{scenario}"
                anc.plot_results(f'{output_dir}/plots/{plot_name}.png')

                # Save audio
                save_comparison_wav(
                    f'optimized_{plot_name}',
                    results['desired'],
                    results['error'],
                    16000,
                    f'{output_dir}/audio'
                )

    return all_results


def print_results_summary(all_results: Dict):
    """Print a comprehensive summary of all results."""
    print("\n" + "=" * 80)
    print("OPTIMIZED SIMULATION RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n{'Config':<15} {'Speaker':<12} {'Scenario':<10} {'Filter':<8} {'MaxOrd':<8} {'NR (dB)':>10}")
    print("-" * 80)

    for config_name, speaker_results in all_results.items():
        for speaker_type, scenario_results in speaker_results.items():
            for scenario, results in scenario_results.items():
                print(f"{config_name:<15} {speaker_type:<12} {scenario:<10} "
                      f"{results['filter_length']:<8} {results['max_order']:<8} "
                      f"{results['noise_reduction_db']:>10.1f}")

    # Summary statistics
    print("\n" + "-" * 80)
    print("SPEAKER CONFIGURATION COMPARISON")
    print("-" * 80)

    for config_name, speaker_results in all_results.items():
        single_avg = np.mean([r['noise_reduction_db']
                             for r in speaker_results.get('single', {}).values()])
        quad_avg = np.mean([r['noise_reduction_db']
                           for r in speaker_results.get('quad_stereo', {}).values()])

        diff = quad_avg - single_avg
        print(f"{config_name:<15} Single: {single_avg:>6.1f} dB | Quad: {quad_avg:>6.1f} dB | Diff: {diff:>+5.1f} dB")


def save_results_json(all_results: Dict, output_path: str):
    """Save results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        else:
            return obj

    # Create summary (without large arrays)
    summary = {}
    for config_name, speaker_results in all_results.items():
        summary[config_name] = {}
        for speaker_type, scenario_results in speaker_results.items():
            summary[config_name][speaker_type] = {}
            for scenario, results in scenario_results.items():
                summary[config_name][speaker_type][scenario] = {
                    'noise_reduction_db': results['noise_reduction_db'],
                    'filter_length': results['filter_length'],
                    'max_order': results['max_order'],
                    'step_size': results['step_size'],
                    'duration': results['duration'],
                }

    summary['timestamp'] = datetime.now().isoformat()

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    """Main entry point for optimized simulation."""
    print("=" * 70)
    print("Optimized FxLMS ANC Simulation")
    print("Testing Single Speaker vs 4-Speaker Stereo Configurations")
    print("=" * 70)

    # Run all configurations
    all_results = run_all_configurations(
        scenarios=['idle', 'city', 'highway'],
        duration=5.0,
        output_dir='output/optimized'
    )

    # Print summary
    print_results_summary(all_results)

    # Save JSON results
    save_results_json(all_results, 'output/optimized/results_summary.json')

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print("\nOutput files:")
    print("  Plots: output/optimized/plots/")
    print("  Audio: output/optimized/audio/")
    print("  Results: output/optimized/results_summary.json")

    return all_results


if __name__ == '__main__':
    main()
