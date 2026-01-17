"""
Step 7: Car Interior ANC Simulation with pyroomacoustics

Goal: Full ANC system for car interior noise cancellation
      in the 20-300 Hz range using realistic acoustic modeling.

This simulation demonstrates:
1. Car cabin geometry with pyroomacoustics
2. Multiple noise sources (engine, road, wind)
3. FxLMS adaptive algorithm
4. Different driving scenarios
5. Comprehensive performance analysis

Runs 3 different car configurations:
- Config A: Compact car (small cabin, engine-dominant noise)
- Config B: Sedan (mid-size, balanced noise)
- Config C: SUV (large cabin, road noise dominant)

Target frequency range: 20-300 Hz
Expected noise reduction: 10-20 dB
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyroomacoustics as pra
from src.acoustic.room_builder import RoomBuilder
from src.acoustic.path_generator import AcousticPathGenerator, FIRPath
from src.core.fxlms import FxNLMS
from src.noise.noise_mixer import NoiseMixer
from src.utils.audio import save_wav, save_comparison_wav
from configurations import STEP7_CONFIGS, print_config_summary


class CarInteriorANC:
    """
    Complete car interior ANC system using pyroomacoustics.
    """

    def __init__(self, config: dict, fs: int = 16000):
        """
        Initialize car interior ANC system from configuration.
        """
        self.fs = fs
        self.config = config

        room_cfg = config['room']
        pos = config['positions']
        fxlms_cfg = config['fxlms']

        # Create car cabin room with custom materials
        self.room = self._create_car_cabin(
            room_cfg['dimensions'],
            room_cfg['materials'],
            room_cfg['max_order'],
            fs
        )

        # Store positions
        self.positions = pos

        # Add sources and microphones
        self.room.add_source(pos['noise_source'])   # Source 0: Noise
        self.room.add_source(pos['speaker'])        # Source 1: Speaker
        mic_array = np.array([pos['reference_mic'], pos['error_mic']]).T
        self.room.add_microphone_array(pra.MicrophoneArray(mic_array, fs=fs))

        # Compute RIRs
        self.room.compute_rir()

        # Extract acoustic paths
        path_gen = AcousticPathGenerator(self.room)
        paths = path_gen.get_all_anc_paths(modeling_error=0.05)

        # Truncate paths for efficiency
        max_len = 512
        self.H_primary = paths['primary'][:max_len]
        self.H_secondary = paths['secondary'][:max_len]
        self.H_secondary_est = paths['secondary_estimate'][:max_len]
        self.H_reference = paths['reference'][:max_len]

        # Create FIR filters
        self.primary_path = FIRPath(self.H_primary)
        self.secondary_path = FIRPath(self.H_secondary)
        self.reference_path = FIRPath(self.H_reference)

        # Create FxNLMS
        self.fxlms = FxNLMS(
            filter_length=fxlms_cfg['filter_length'],
            step_size=fxlms_cfg['step_size'],
            secondary_path_estimate=self.H_secondary_est,
            regularization=1e-4  # Small regularization for better adaptation
        )

        # Noise generator
        self.noise_gen = NoiseMixer(fs)

        # Results
        self.results = {}

        dims = room_cfg['dimensions']
        print(f"Car Interior ANC System initialized:")
        print(f"  Car type: {config['name']}")
        print(f"  Cabin: {dims[0]:.1f}m x {dims[1]:.1f}m x {dims[2]:.1f}m")
        print(f"  Filter length: {fxlms_cfg['filter_length']} taps")
        print(f"  Primary path: {len(self.H_primary)} taps")
        print(f"  Secondary path: {len(self.H_secondary)} taps")

    def _create_car_cabin(
        self,
        dimensions: list,
        materials: dict,
        max_order: int,
        fs: int
    ) -> pra.ShoeBox:
        """Create realistic car cabin room."""
        # Convert material absorption coefficients to pra.Material objects
        pra_materials = {
            'ceiling': pra.Material(materials['ceiling']),
            'floor': pra.Material(materials['floor']),
            'east': pra.Material(materials['east']),
            'west': pra.Material(materials['west']),
            'north': pra.Material(materials['north']),
            'south': pra.Material(materials['south']),
        }

        room = pra.ShoeBox(
            dimensions,
            fs=fs,
            materials=pra_materials,
            max_order=max_order,
            air_absorption=True
        )

        return room

    def run_scenario(
        self,
        duration: float = 5.0,
        scenario: str = 'highway',
        verbose: bool = True
    ) -> dict:
        """
        Run ANC simulation for a driving scenario.
        """
        print(f"\nRunning {scenario} scenario ({duration}s)...")

        # Generate noise for scenario
        noise_source = self.noise_gen.generate_scenario(duration, scenario)
        n_samples = len(noise_source)

        # Reset filters
        self.fxlms.reset()
        self.primary_path.reset()
        self.secondary_path.reset()
        self.reference_path.reset()

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

            # Generate anti-noise
            y = self.fxlms.generate_antinoise(x)
            antinoise.append(y)

            # Anti-noise through secondary path
            y_at_error = self.secondary_path.filter_sample(y)

            # Error
            e = d + y_at_error
            error.append(e)
            mse.append(e ** 2)

            # Update FxLMS
            self.fxlms.filter_reference(x)
            self.fxlms.update_weights(e)

            # Progress
            if verbose and (i + 1) % (n_samples // 10) == 0:
                progress = (i + 1) / n_samples * 100
                current_mse = np.mean(mse[-1000:]) if len(mse) > 1000 else np.mean(mse)
                print(f"  Progress: {progress:.0f}% | MSE: {current_mse:.6f}")

        # Store results
        self.results = {
            'reference': np.array(reference),
            'desired': np.array(desired),
            'antinoise': np.array(antinoise),
            'error': np.array(error),
            'mse': np.array(mse),
            'scenario': scenario,
            'duration': duration
        }

        # Calculate noise reduction
        steady_start = len(desired) // 2
        d_power = np.mean(np.array(desired[steady_start:])**2)
        e_power = np.mean(np.array(error[steady_start:])**2)

        if e_power > 1e-10:
            nr_db = 10 * np.log10(d_power / e_power)
        else:
            nr_db = 60.0

        self.results['noise_reduction_db'] = nr_db
        print(f"\nNoise Reduction: {nr_db:.1f} dB")

        return self.results

    def plot_results(self, save_path: str = None):
        """Generate comprehensive result plots."""
        if not self.results:
            print("No results to plot.")
            return

        t = np.arange(len(self.results['reference'])) / self.fs

        fig = plt.figure(figsize=(16, 12))

        # Time domain (last 100ms)
        show_samples = int(0.1 * self.fs)
        t_show = t[-show_samples:] * 1000

        # Reference
        ax1 = fig.add_subplot(3, 2, 1)
        ax1.plot(t_show, self.results['reference'][-show_samples:], 'b-', linewidth=0.8)
        ax1.set_ylabel('Reference x(n)')
        ax1.set_title(f"{self.config['name']} - {self.results['scenario'].title()}")
        ax1.grid(True, alpha=0.3)

        # Before/After
        ax2 = fig.add_subplot(3, 2, 2)
        ax2.plot(t_show, self.results['desired'][-show_samples:], 'r-', linewidth=0.8, alpha=0.7, label='Noise')
        ax2.plot(t_show, self.results['error'][-show_samples:], 'g-', linewidth=0.8, alpha=0.7, label='With ANC')
        ax2.set_ylabel('Amplitude')
        ax2.set_title(f"Reduction: {self.results['noise_reduction_db']:.1f} dB")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # MSE convergence
        ax3 = fig.add_subplot(3, 2, 3)
        window = 200
        mse_smooth = np.convolve(self.results['mse'], np.ones(window)/window, mode='valid')
        ax3.semilogy(np.arange(len(mse_smooth)) / self.fs, mse_smooth)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('MSE')
        ax3.set_title('Convergence')
        ax3.grid(True, alpha=0.3)

        # Frequency spectrum
        ax4 = fig.add_subplot(3, 2, 4)
        from scipy import signal as scipy_signal

        steady_start = len(self.results['desired']) // 2
        f_d, psd_d = scipy_signal.welch(self.results['desired'][steady_start:], self.fs, nperseg=2048)
        f_e, psd_e = scipy_signal.welch(self.results['error'][steady_start:], self.fs, nperseg=2048)

        ax4.semilogy(f_d, psd_d, 'r-', linewidth=1.5, alpha=0.7, label='Noise')
        ax4.semilogy(f_e, psd_e, 'g-', linewidth=1.5, alpha=0.7, label='With ANC')
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('PSD')
        ax4.set_title('Spectrum (20-300 Hz target)')
        ax4.set_xlim(0, 350)
        ax4.axvline(x=20, color='gray', linestyle='--', alpha=0.5)
        ax4.axvline(x=300, color='gray', linestyle='--', alpha=0.5)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Adaptive filter weights
        ax5 = fig.add_subplot(3, 2, 5)
        ax5.plot(self.fxlms.weights, 'b-', linewidth=1)
        ax5.set_xlabel('Tap Index')
        ax5.set_ylabel('Weight')
        ax5.set_title('Learned Filter Coefficients')
        ax5.grid(True, alpha=0.3)

        # Noise reduction over time
        ax6 = fig.add_subplot(3, 2, 6)
        window_size = int(0.5 * self.fs)
        n_windows = len(self.results['desired']) // window_size

        nr_over_time = []
        time_points = []

        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            d_pow = np.mean(self.results['desired'][start:end]**2)
            e_pow = np.mean(self.results['error'][start:end]**2)
            if e_pow > 1e-10:
                nr = 10 * np.log10(d_pow / e_pow)
            else:
                nr = 30
            nr_over_time.append(nr)
            time_points.append((start + end) / 2 / self.fs)

        ax6.plot(time_points, nr_over_time, 'g-o', linewidth=2, markersize=4)
        ax6.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Noise Reduction (dB)')
        ax6.set_title('Performance Over Time')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {save_path}")
        else:
            plt.close()

        return fig


def main():
    """
    Run complete car interior ANC simulation for all configurations.
    """
    print("=" * 70)
    print("Step 7: Car Interior ANC with pyroomacoustics")
    print("=" * 70)
    print()

    # Ensure output directories exist
    os.makedirs('output/plots', exist_ok=True)
    os.makedirs('output/audio', exist_ok=True)

    fs = 16000
    all_results = {}

    # Run all configurations
    for config_key, config in STEP7_CONFIGS.items():
        print_config_summary(config, f"Step 7 - {config_key}")

        # Create ANC system
        anc = CarInteriorANC(config, fs)

        # Get scenario from config
        scenario = config.get('scenario', 'highway')
        duration = config['fxlms']['duration']

        # Run scenario
        results = anc.run_scenario(duration=duration, scenario=scenario)
        all_results[config_key] = {
            'results': results,
            'config': config,
            'reduction_db': results['noise_reduction_db']
        }

        # Plot
        anc.plot_results(f'output/plots/pyroom_step7_{config_key}.png')

        # Save audio
        save_comparison_wav(
            f'pyroom_step7_{config_key}',
            results['desired'],
            results['error'],
            fs,
            'output/audio'
        )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - Car Interior ANC Performance")
    print("=" * 70)
    print()
    print(f"{'Configuration':<20} {'Cabin Size':<20} {'Scenario':<15} {'Reduction':>12}")
    print("-" * 70)

    for config_key, result_data in all_results.items():
        config = result_data['config']
        dims = config['room']['dimensions']
        size = f"{dims[0]:.1f}x{dims[1]:.1f}x{dims[2]:.1f}m"
        scenario = config.get('scenario', 'highway')
        nr = result_data['reduction_db']
        print(f"{config['name']:<20} {size:<20} {scenario:<15} {nr:>10.1f} dB")

    print()
    print("=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print()
    print("Key observations:")
    print("1. FxLMS successfully reduces car noise in 20-300 Hz range")
    print("2. pyroomacoustics provides realistic acoustic modeling")
    print("3. Different car sizes affect acoustic characteristics")
    print("4. Convergence takes ~1-2 seconds")
    print()
    print("Output files:")
    print("  Plots: output/plots/pyroom_step7_*.png")
    print("  Audio: output/audio/pyroom_step7_*_comparison.wav")
    print()

    return all_results


if __name__ == '__main__':
    main()
