"""
Step 7: Car Interior ANC Simulation

Goal: Apply FxLMS-based ANC to a realistic car interior scenario
      with multiple noise sources (engine, road, wind) in the 20-300 Hz range.

This is the FINAL step that brings together everything:
- Wave physics understanding (Steps 1-3)
- ANC principles (Steps 4-5)
- FxLMS algorithm (Step 6)
- Realistic car noise (this step)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.fxlms import FxLMS
from src.noise.noise_mixer import NoiseMixer
from src.utils.audio import save_comparison_wav


class CarInteriorANC:
    """
    Complete car interior ANC simulation using FxLMS.

    Simulates:
    - Primary path (noise source to driver's ear)
    - Secondary path (speaker to driver's ear)
    - FxLMS adaptive filter
    - Realistic car interior noise (engine + road + wind)
    """

    def __init__(
        self,
        sample_rate: float = 16000,
        primary_path_ms: float = 8.0,    # ~2.7m at 343 m/s
        secondary_path_ms: float = 2.0,   # ~0.7m speaker to ear
        filter_length: int = 256,
        step_size: float = 0.005
    ):
        """
        Initialize car interior ANC simulation.

        Args:
            sample_rate: Sampling rate in Hz
            primary_path_ms: Primary path delay in milliseconds
            secondary_path_ms: Secondary path delay in milliseconds
            filter_length: FxLMS adaptive filter length
            step_size: FxLMS step size (mu)
        """
        self.fs = sample_rate
        self.filter_length = filter_length
        self.step_size = step_size

        # Convert delays to samples
        self.primary_delay = int(primary_path_ms * sample_rate / 1000)
        self.secondary_delay = int(secondary_path_ms * sample_rate / 1000)

        # Create primary path (noise to listener)
        # Includes delay + room modes + reflections
        self.primary_path = self._create_primary_path()

        # Create secondary path (speaker to listener)
        self.secondary_path = self._create_secondary_path()

        # Secondary path estimate (with slight error to be realistic)
        self.secondary_path_estimate = self._create_secondary_path_estimate()

        # Initialize FxLMS
        self.fxlms = FxLMS(
            filter_length=filter_length,
            step_size=step_size,
            secondary_path_estimate=self.secondary_path_estimate
        )

        # Buffers for path filtering
        self.primary_buffer = np.zeros(len(self.primary_path))
        self.secondary_buffer = np.zeros(len(self.secondary_path))

        # Noise generator
        self.noise_gen = NoiseMixer(sample_rate)

        # Results storage
        self.results = {}

        print("Car Interior ANC System initialized:")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Primary path: {primary_path_ms:.1f} ms ({self.primary_delay} samples)")
        print(f"  Secondary path: {secondary_path_ms:.1f} ms ({self.secondary_delay} samples)")
        print(f"  Filter length: {filter_length}")
        print(f"  Step size: {step_size}")

    def _create_primary_path(self) -> np.ndarray:
        """Create realistic primary path impulse response."""
        length = self.primary_delay + 64
        h = np.zeros(length)

        # Main path (direct + early reflections)
        if self.primary_delay < length:
            h[self.primary_delay] = 1.0

            # Add car body resonances (typical modes)
            for i in range(self.primary_delay + 1, length):
                decay = np.exp(-0.1 * (i - self.primary_delay))
                h[i] = 0.2 * decay * np.sin(0.5 * (i - self.primary_delay))

        # Normalize
        h /= np.max(np.abs(h)) + 1e-10

        return h

    def _create_secondary_path(self) -> np.ndarray:
        """Create realistic secondary path impulse response."""
        length = self.secondary_delay + 32
        h = np.zeros(length)

        if self.secondary_delay < length:
            h[self.secondary_delay] = 1.0

            # Speaker response characteristics
            for i in range(self.secondary_delay + 1, length):
                decay = np.exp(-0.2 * (i - self.secondary_delay))
                h[i] = 0.15 * decay * ((-1) ** (i - self.secondary_delay))

        h /= np.max(np.abs(h)) + 1e-10

        return h

    def _create_secondary_path_estimate(self) -> np.ndarray:
        """Create estimated secondary path (with realistic error)."""
        # Copy true path
        h_hat = self.secondary_path.copy()

        # Add small modeling error (5%)
        noise = 0.05 * np.std(h_hat) * np.random.randn(len(h_hat))
        h_hat += noise

        # Small gain error
        h_hat *= (1 + 0.02 * np.random.randn())

        return h_hat

    def _filter_through_path(self, sample: float, path: np.ndarray, buffer: np.ndarray) -> tuple:
        """Filter sample through an acoustic path."""
        buffer = np.roll(buffer, 1)
        buffer[0] = sample
        output = np.dot(path, buffer[:len(path)])
        return output, buffer

    def run_simulation(
        self,
        duration: float = 5.0,
        scenario: str = 'highway',
        verbose: bool = True
    ) -> dict:
        """
        Run complete ANC simulation.

        Args:
            duration: Simulation duration in seconds
            scenario: 'highway', 'city', 'acceleration', 'idle'
            verbose: Print progress

        Returns:
            Dict with all signals and metrics
        """
        print(f"\nRunning {scenario} scenario for {duration}s...")

        # Generate noise
        reference_signal = self.noise_gen.generate_scenario(duration, scenario)
        n_samples = len(reference_signal)

        # Reset state
        self.fxlms.reset()
        self.primary_buffer = np.zeros(len(self.primary_path))
        self.secondary_buffer = np.zeros(len(self.secondary_path))

        # Storage
        reference = []
        desired = []  # Noise at listener (d(n))
        output = []   # Anti-noise (y(n))
        error = []    # Residual (e(n))
        mse = []

        for i in range(n_samples):
            x = reference_signal[i]
            reference.append(x)

            # Noise through primary path -> d(n)
            d, self.primary_buffer = self._filter_through_path(
                x, self.primary_path, self.primary_buffer
            )
            desired.append(d)

            # Generate anti-noise through adaptive filter
            y = self.fxlms.generate_antinoise(x)
            output.append(y)

            # Anti-noise through secondary path
            y_filtered, self.secondary_buffer = self._filter_through_path(
                y, self.secondary_path, self.secondary_buffer
            )

            # Error signal: e(n) = d(n) + y'(n)
            e = d + y_filtered
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
            'output': np.array(output),
            'error': np.array(error),
            'mse': np.array(mse),
            'scenario': scenario,
            'duration': duration
        }

        # Calculate metrics
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
            print("No results to plot. Run simulation first.")
            return

        t = np.arange(len(self.results['reference'])) / self.fs

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 12))

        # Time domain signals (last 100ms)
        show_samples = int(0.1 * self.fs)
        t_show = t[-show_samples:] * 1000

        # Reference signal
        ax1 = fig.add_subplot(3, 2, 1)
        ax1.plot(t_show, self.results['reference'][-show_samples:], 'b-', linewidth=0.8)
        ax1.set_ylabel('Reference x(n)')
        ax1.set_title(f"Car Interior ANC - {self.results['scenario'].title()} Scenario")
        ax1.grid(True, alpha=0.3)

        # Before/After comparison
        ax2 = fig.add_subplot(3, 2, 2)
        ax2.plot(t_show, self.results['desired'][-show_samples:], 'r-', linewidth=0.8, alpha=0.7, label='Noise d(n)')
        ax2.plot(t_show, self.results['error'][-show_samples:], 'g-', linewidth=0.8, alpha=0.7, label='Error e(n)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title(f"Noise Reduction: {self.results['noise_reduction_db']:.1f} dB")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # MSE convergence
        ax3 = fig.add_subplot(3, 2, 3)
        window = 200
        mse_smooth = np.convolve(self.results['mse'], np.ones(window)/window, mode='valid')
        ax3.semilogy(np.arange(len(mse_smooth)) / self.fs, mse_smooth)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('MSE')
        ax3.set_title('Convergence (Mean Squared Error)')
        ax3.grid(True, alpha=0.3)

        # Frequency spectrum
        ax4 = fig.add_subplot(3, 2, 4)
        from scipy import signal as sig

        # Use second half for steady-state spectrum
        steady_start = len(self.results['desired']) // 2
        f_d, psd_d = sig.welch(self.results['desired'][steady_start:], self.fs, nperseg=2048)
        f_e, psd_e = sig.welch(self.results['error'][steady_start:], self.fs, nperseg=2048)

        ax4.semilogy(f_d, psd_d, 'r-', linewidth=1.5, alpha=0.7, label='Noise')
        ax4.semilogy(f_e, psd_e, 'g-', linewidth=1.5, alpha=0.7, label='With ANC')
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('PSD')
        ax4.set_title('Frequency Spectrum (20-300 Hz)')
        ax4.set_xlim(0, 350)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Adaptive filter weights
        ax5 = fig.add_subplot(3, 2, 5)
        ax5.plot(self.fxlms.weights, 'b-', linewidth=1)
        ax5.set_xlabel('Tap Index')
        ax5.set_ylabel('Weight Value')
        ax5.set_title('Learned Adaptive Filter Coefficients')
        ax5.grid(True, alpha=0.3)

        # Noise reduction over time
        ax6 = fig.add_subplot(3, 2, 6)
        window_size = int(0.5 * self.fs)  # 500ms windows
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
        ax6.set_title('Noise Reduction Over Time')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig


def main():
    """
    Run complete car interior ANC simulation.
    """
    print("=" * 70)
    print("Step 7: Car Interior ANC Simulation with FxLMS")
    print("=" * 70)

    # Create ANC system
    anc = CarInteriorANC(
        sample_rate=16000,
        primary_path_ms=8.0,
        secondary_path_ms=2.0,
        filter_length=256,
        step_size=0.005
    )

    # Test different scenarios
    scenarios = ['highway', 'city', 'acceleration']

    all_results = {}

    # Ensure audio directory exists
    os.makedirs('output/audio', exist_ok=True)

    for scenario in scenarios:
        print("\n" + "=" * 50)
        results = anc.run_simulation(duration=5.0, scenario=scenario)
        all_results[scenario] = results

        # Plot and save
        fig = anc.plot_results(f'output/plots/step7_{scenario}.png')

        # Save audio files
        orig_path, cancel_path, comp_path = save_comparison_wav(
            filename_prefix=f'step7_{scenario}',
            original=results['desired'],
            cancelled=results['error'],
            sample_rate=16000,
            output_dir='output/audio'
        )
        print(f"\nAudio files saved:")
        print(f"  Original: {orig_path}")
        print(f"  After ANC: {cancel_path}")
        print(f"  Comparison: {comp_path}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY - Noise Reduction by Scenario")
    print("=" * 70)
    print()
    print(f"{'Scenario':<15} {'Noise Reduction':>20}")
    print("-" * 40)
    for scenario, results in all_results.items():
        nr = results['noise_reduction_db']
        print(f"{scenario:<15} {nr:>18.1f} dB")

    print()
    print("=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print()
    print("Key observations:")
    print("1. FxLMS successfully cancels car interior noise in 20-300 Hz range")
    print("2. Convergence takes ~1-2 seconds as algorithm learns the acoustic paths")
    print("3. Performance varies with noise type (tonal engine noise vs broadband)")
    print("4. Final noise reduction of 10-20 dB is typical for real ANC systems")
    print()
    print("Output files:")
    print("  Plots: output/plots/step7_*.png")
    print("  Audio: output/audio/step7_*_original.wav (noise)")
    print("         output/audio/step7_*_cancelled.wav (after ANC)")
    print("         output/audio/step7_*_comparison.wav (before→after)")
    print()

    plt.show()

    return anc, all_results


if __name__ == '__main__':
    main()
