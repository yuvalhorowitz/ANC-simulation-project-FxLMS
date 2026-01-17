"""
Step 6: FxLMS Algorithm - The Solution

Goal: Use the FxLMS (Filtered-x LMS) adaptive algorithm to achieve
      noise cancellation even with secondary path latency.

Why FxLMS works:
1. It LEARNS the optimal filter weights to cancel noise
2. It compensates for the secondary path by filtering the reference
   signal through an ESTIMATE of the secondary path
3. It ADAPTS over time, continuously improving cancellation

This step brings together everything we've learned:
- Wave propagation (Step 1)
- Microphone measurement (Step 2)
- Speaker interaction and superposition (Step 3)
- The ANC goal (Step 4)
- The latency problem (Step 5)
- The FxLMS solution (THIS STEP)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.fxlms import FxLMS
from src.utils.audio import save_comparison_wav


class SimpleSecondaryPath:
    """
    Simple model of the secondary path S(z).

    The secondary path includes everything from controller output
    to the error microphone: DAC, amplifier, speaker, acoustics.

    For simplicity, we model it as a delay + low-pass filter.
    """

    def __init__(self, delay_samples: int, num_taps: int = 32):
        """
        Create simple secondary path model.

        Args:
            delay_samples: Pure delay in samples
            num_taps: Length of impulse response
        """
        self.delay = delay_samples
        self.num_taps = num_taps

        # Create impulse response: delay + exponential decay
        self.impulse_response = np.zeros(num_taps)

        if delay_samples < num_taps:
            # Main impulse at delay position
            self.impulse_response[delay_samples] = 1.0

            # Add some decay (speaker/room response)
            for i in range(delay_samples + 1, num_taps):
                decay = np.exp(-0.3 * (i - delay_samples))
                self.impulse_response[i] = 0.2 * decay * ((-1) ** (i - delay_samples))

        # Normalize
        self.impulse_response /= np.sum(np.abs(self.impulse_response)) + 1e-10

        # Buffer for filtering
        self.buffer = np.zeros(num_taps)

    def filter(self, sample: float) -> float:
        """
        Filter a single sample through the secondary path.

        Args:
            sample: Input sample

        Returns:
            Filtered output
        """
        # Shift buffer and insert new sample
        self.buffer = np.roll(self.buffer, 1)
        self.buffer[0] = sample

        # Convolve
        output = np.dot(self.impulse_response, self.buffer)
        return output

    def reset(self):
        """Reset filter state."""
        self.buffer = np.zeros(self.num_taps)


class SimplePrimaryPath:
    """
    Simple model of the primary path P(z).

    The primary path is the acoustic path from noise source
    to error microphone (through the room/enclosure).
    """

    def __init__(self, delay_samples: int, num_taps: int = 64):
        """
        Create simple primary path model.

        Args:
            delay_samples: Acoustic delay in samples
            num_taps: Length of impulse response
        """
        self.delay = delay_samples
        self.num_taps = num_taps

        # Create impulse response
        self.impulse_response = np.zeros(num_taps)

        if delay_samples < num_taps:
            # Main impulse
            self.impulse_response[delay_samples] = 1.0

            # Add some reflections (room acoustics)
            for i in range(delay_samples + 1, num_taps):
                if (i - delay_samples) % 5 == 0:  # Some reflections
                    decay = np.exp(-0.1 * (i - delay_samples))
                    self.impulse_response[i] = 0.3 * decay

        self.buffer = np.zeros(num_taps)

    def filter(self, sample: float) -> float:
        """Filter a single sample through the primary path."""
        self.buffer = np.roll(self.buffer, 1)
        self.buffer[0] = sample
        return np.dot(self.impulse_response, self.buffer)

    def reset(self):
        """Reset filter state."""
        self.buffer = np.zeros(self.num_taps)


class FxLMSANCSystem:
    """
    Complete FxLMS-based Active Noise Control system.

    This implements the feedforward ANC architecture:

    x(n) ──┬──────────────────────────────────────┐
           │                                       │
           │                                       ▼
           │                               ┌──────────────┐
           │                               │ Primary Path │
           │                               │     P(z)     │
           │                               └──────┬───────┘
           │                                      │ d(n)
           ▼                                      ▼
    ┌──────────────┐                       ┌──────────┐
    │   Adaptive   │──── y(n) ────────────►│    +     │───► e(n)
    │   Filter     │                       └──────────┘
    │     W(z)     │                              ▲
    └──────────────┘                              │
           ▲                               ┌──────┴───────┐
           │                               │Secondary Path│
           │                               │     S(z)     │
           │          ┌──────────┐         └──────────────┘
           │          │ Ŝ(z)     │                │
           └──────────│ (estimate)│◄───── y(n) ───┘
                      └──────────┘
                           │
                      f(n) │  (filtered reference)
                           ▼
                    ┌──────────────┐
                    │     LMS      │◄──── e(n)
                    │    Update    │
                    └──────────────┘
    """

    def __init__(
        self,
        sample_rate: float,
        filter_length: int,
        step_size: float,
        secondary_path: SimpleSecondaryPath,
        primary_path: SimplePrimaryPath,
        secondary_path_estimate: np.ndarray = None
    ):
        """
        Initialize FxLMS ANC system.

        Args:
            sample_rate: Sampling rate in Hz
            filter_length: Length of adaptive filter W(z)
            step_size: LMS step size (mu)
            secondary_path: Actual secondary path S(z)
            primary_path: Primary path P(z)
            secondary_path_estimate: Estimate of S(z) for FxLMS
        """
        self.fs = sample_rate
        self.L = filter_length
        self.mu = step_size

        # Acoustic paths
        self.S = secondary_path
        self.P = primary_path

        # Secondary path estimate (if not provided, use perfect estimate)
        if secondary_path_estimate is None:
            self.S_hat = secondary_path.impulse_response.copy()
        else:
            self.S_hat = secondary_path_estimate

        # Create FxLMS adaptive filter
        self.fxlms = FxLMS(
            filter_length=filter_length,
            step_size=step_size,
            secondary_path_estimate=self.S_hat
        )

        # Buffer for secondary path output
        self.y_buffer = np.zeros(len(secondary_path.impulse_response))

        # History
        self.reference_history = []
        self.desired_history = []  # d(n) - noise at error mic
        self.output_history = []   # y(n) - anti-noise
        self.error_history = []    # e(n) - residual
        self.mse_history = []

        print(f"FxLMS ANC System initialized:")
        print(f"  Filter length: {filter_length}")
        print(f"  Step size: {step_size}")
        print(f"  Sample rate: {sample_rate} Hz")

    def process_sample(self, x: float) -> float:
        """
        Process one sample through the ANC system.

        Args:
            x: Reference signal sample (noise picked up by ref mic)

        Returns:
            Error signal e(n) (residual noise at error mic)
        """
        # Store reference
        self.reference_history.append(x)

        # Noise travels through primary path to error mic
        d = self.P.filter(x)
        self.desired_history.append(d)

        # Generate anti-noise through adaptive filter
        y = self.fxlms.generate_antinoise(x)
        self.output_history.append(y)

        # Anti-noise travels through secondary path
        y_filtered = self.S.filter(y)

        # Error signal at error microphone
        # e(n) = d(n) + y'(n)  where y'(n) is anti-noise after secondary path
        # For cancellation, we want y'(n) = -d(n), so e(n) = 0
        e = d + y_filtered
        self.error_history.append(e)

        # Update adaptive filter using FxLMS
        # First, filter reference through secondary path estimate
        f_n = self.fxlms.filter_reference(x)

        # Then update weights
        self.fxlms.update_weights(e)

        # Store MSE
        self.mse_history.append(e ** 2)

        return e

    def run(self, reference_signal: np.ndarray, verbose: bool = True) -> dict:
        """
        Run complete simulation.

        Args:
            reference_signal: Input noise signal x(n)
            verbose: Print progress

        Returns:
            Dict with all signals
        """
        n_samples = len(reference_signal)

        for i, x in enumerate(reference_signal):
            self.process_sample(x)

            if verbose and (i + 1) % (n_samples // 10) == 0:
                progress = (i + 1) / n_samples * 100
                current_mse = np.mean(self.mse_history[-1000:]) if len(self.mse_history) > 1000 else np.mean(self.mse_history)
                print(f"  Progress: {progress:.0f}% | MSE: {current_mse:.6f}")

        return {
            'reference': np.array(self.reference_history),
            'desired': np.array(self.desired_history),
            'output': np.array(self.output_history),
            'error': np.array(self.error_history),
            'mse': np.array(self.mse_history)
        }

    def get_noise_reduction_db(self, window: int = 1000) -> float:
        """Calculate noise reduction in dB."""
        if len(self.desired_history) < window:
            return 0.0

        d_power = np.mean(np.array(self.desired_history[-window:])**2)
        e_power = np.mean(np.array(self.error_history[-window:])**2)

        if e_power < 1e-10:
            return 60.0

        return 10 * np.log10(d_power / e_power)

    def reset(self):
        """Reset system state."""
        self.fxlms.reset()
        self.S.reset()
        self.P.reset()
        self.reference_history = []
        self.desired_history = []
        self.output_history = []
        self.error_history = []
        self.mse_history = []


def main():
    """
    Demonstrate FxLMS-based ANC.
    """
    print("=" * 60)
    print("Step 6: FxLMS Algorithm - Active Noise Cancellation")
    print("=" * 60)
    print()

    # System parameters
    sample_rate = 16000  # Hz
    duration = 2.0       # seconds
    n_samples = int(sample_rate * duration)

    # Create acoustic paths
    # Primary path: ~5ms delay (about 1.7m at 343 m/s)
    primary_delay_samples = int(0.005 * sample_rate)
    primary_path = SimplePrimaryPath(primary_delay_samples, num_taps=64)

    # Secondary path: ~2ms delay (speaker to ear)
    secondary_delay_samples = int(0.002 * sample_rate)
    secondary_path = SimpleSecondaryPath(secondary_delay_samples, num_taps=32)

    print(f"Primary path delay: {primary_delay_samples} samples ({primary_delay_samples/sample_rate*1000:.1f} ms)")
    print(f"Secondary path delay: {secondary_delay_samples} samples ({secondary_delay_samples/sample_rate*1000:.1f} ms)")
    print()

    # FxLMS parameters
    filter_length = 128
    step_size = 0.01  # This is crucial - too large = unstable, too small = slow

    # Create ANC system
    anc = FxLMSANCSystem(
        sample_rate=sample_rate,
        filter_length=filter_length,
        step_size=step_size,
        secondary_path=secondary_path,
        primary_path=primary_path
    )

    print()

    # =========================================
    # Test 1: Single frequency noise (100 Hz)
    # =========================================
    print("Test 1: Single Frequency Noise (100 Hz)")
    print("-" * 40)

    frequency = 100  # Hz
    t = np.arange(n_samples) / sample_rate
    reference_signal = np.sin(2 * np.pi * frequency * t)

    print("Running FxLMS simulation...")
    results = anc.run(reference_signal)

    # Calculate final noise reduction
    nr_db = anc.get_noise_reduction_db()
    print(f"\nFinal noise reduction: {nr_db:.1f} dB")

    # =========================================
    # Save audio files
    # =========================================
    os.makedirs('output/audio', exist_ok=True)

    # Save comparison audio (noise vs cancelled)
    orig_path, cancel_path, comp_path = save_comparison_wav(
        filename_prefix='step6_100hz',
        original=results['desired'],
        cancelled=results['error'],
        sample_rate=sample_rate,
        output_dir='output/audio'
    )
    print(f"\nAudio files saved:")
    print(f"  Original noise: {orig_path}")
    print(f"  After ANC: {cancel_path}")
    print(f"  Comparison (before→after): {comp_path}")

    # =========================================
    # Plot results
    # =========================================

    # Time domain signals
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # Show last 50ms for clarity
    show_samples = int(0.05 * sample_rate)
    t_show = t[-show_samples:] * 1000

    axes[0].plot(t_show, results['reference'][-show_samples:], 'b-', linewidth=1)
    axes[0].set_ylabel('Reference\nx(n)')
    axes[0].set_title('FxLMS ANC: Time Domain Signals (last 50ms)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_show, results['desired'][-show_samples:], 'r-', linewidth=1)
    axes[1].set_ylabel('Noise at ear\nd(n)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_show, results['output'][-show_samples:], 'g-', linewidth=1)
    axes[2].set_ylabel('Anti-noise\ny(n)')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t_show, results['error'][-show_samples:], 'purple', linewidth=1)
    axes[3].set_ylabel('Error\ne(n)')
    axes[3].set_xlabel('Time (ms)')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/plots/step6_fxlms_signals.png', dpi=150)
    print("\nSaved: output/plots/step6_fxlms_signals.png")

    # MSE convergence
    fig, ax = plt.subplots(figsize=(12, 5))

    # Smooth MSE for plotting
    window = 100
    mse_smooth = np.convolve(results['mse'], np.ones(window)/window, mode='valid')

    ax.semilogy(np.arange(len(mse_smooth)) / sample_rate, mse_smooth)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title(f'FxLMS Convergence (Final Reduction: {nr_db:.1f} dB)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/plots/step6_convergence.png', dpi=150)
    print("Saved: output/plots/step6_convergence.png")

    # Before/After comparison
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # Before (early in simulation)
    early_samples = slice(int(0.01 * sample_rate), int(0.06 * sample_rate))
    t_early = t[early_samples] * 1000

    axes[0].plot(t_early, results['desired'][early_samples], 'r-', linewidth=1.5, label='Noise d(n)')
    axes[0].plot(t_early, results['error'][early_samples], 'b-', linewidth=1.5, label='Error e(n)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('BEFORE Adaptation (first 50ms) - Algorithm Still Learning')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # After (end of simulation)
    axes[1].plot(t_show, results['desired'][-show_samples:], 'r-', linewidth=1.5, label='Noise d(n)')
    axes[1].plot(t_show, results['error'][-show_samples:], 'g-', linewidth=1.5, label='Error e(n)')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title(f'AFTER Adaptation (last 50ms) - {nr_db:.1f} dB Reduction!')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/plots/step6_before_after.png', dpi=150)
    print("Saved: output/plots/step6_before_after.png")

    # =========================================
    # Test 2: Multi-frequency noise
    # =========================================
    print()
    print("Test 2: Multi-Frequency Noise (50 + 120 + 200 Hz)")
    print("-" * 40)

    anc.reset()

    # Create multi-frequency noise (more realistic)
    reference_multi = (
        0.5 * np.sin(2 * np.pi * 50 * t) +
        0.3 * np.sin(2 * np.pi * 120 * t) +
        0.2 * np.sin(2 * np.pi * 200 * t)
    )

    print("Running FxLMS simulation...")
    results_multi = anc.run(reference_multi)
    nr_multi = anc.get_noise_reduction_db()
    print(f"\nFinal noise reduction: {nr_multi:.1f} dB")

    # Save multi-frequency audio
    orig_path, cancel_path, comp_path = save_comparison_wav(
        filename_prefix='step6_multi_freq',
        original=results_multi['desired'],
        cancelled=results_multi['error'],
        sample_rate=sample_rate,
        output_dir='output/audio'
    )
    print(f"\nAudio files saved:")
    print(f"  Original noise: {orig_path}")
    print(f"  After ANC: {cancel_path}")
    print(f"  Comparison (before→after): {comp_path}")

    # Plot spectrum comparison
    fig, ax = plt.subplots(figsize=(12, 5))

    # Compute spectra of last portion
    from scipy import signal as sig
    f_d, psd_d = sig.welch(results_multi['desired'][-n_samples//2:], sample_rate, nperseg=1024)
    f_e, psd_e = sig.welch(results_multi['error'][-n_samples//2:], sample_rate, nperseg=1024)

    ax.semilogy(f_d, psd_d, 'r-', linewidth=2, label='Noise (without ANC)')
    ax.semilogy(f_e, psd_e, 'g-', linewidth=2, label='Error (with ANC)')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title(f'Frequency Spectrum: Noise Reduction at Multiple Frequencies')
    ax.set_xlim(0, 300)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark the frequencies
    for freq in [50, 120, 200]:
        ax.axvline(x=freq, color='gray', linestyle='--', alpha=0.5)
        ax.text(freq, ax.get_ylim()[1], f'{freq} Hz', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('output/plots/step6_spectrum.png', dpi=150)
    print("Saved: output/plots/step6_spectrum.png")

    # =========================================
    # Summary
    # =========================================
    print()
    print("=" * 60)
    print("FxLMS ALGORITHM - KEY POINTS")
    print("=" * 60)
    print()
    print("1. ADAPTIVE LEARNING:")
    print("   FxLMS continuously adjusts filter weights to minimize error.")
    print("   No need to know the exact acoustic paths in advance!")
    print()
    print("2. SECONDARY PATH COMPENSATION:")
    print("   The reference signal is filtered through Ŝ(z) before the LMS update.")
    print("   This compensates for the delay and frequency response of S(z).")
    print()
    print("3. CONVERGENCE:")
    print("   - Step size (mu) controls speed vs. stability tradeoff")
    print("   - Too large: unstable (oscillations)")
    print("   - Too small: slow convergence")
    print()
    print("4. RESULTS:")
    print(f"   - Single frequency (100 Hz): {nr_db:.1f} dB reduction")
    print(f"   - Multi-frequency: {nr_multi:.1f} dB reduction")
    print()
    print("NEXT: Apply this to car interior with realistic noise sources!")
    print()

    plt.show()

    return anc, results


if __name__ == '__main__':
    main()
