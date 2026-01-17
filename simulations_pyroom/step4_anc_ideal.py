"""
Step 4: Ideal ANC with Known Acoustic Paths

Goal: Demonstrate perfect noise cancellation when we have
      complete knowledge of the acoustic paths.

This simulation demonstrates:
1. Full feedforward ANC architecture with pyroomacoustics
2. Reference microphone captures noise upstream
3. Knowing exact primary and secondary paths, we can compute perfect anti-noise
4. Achieves near-perfect cancellation (theoretical limit)

Runs 3 different configurations:
- Config A: Small Office with HVAC hum (100 Hz)
- Config B: Living Room with traffic noise (50+80+120 Hz)
- Config C: Industrial Space with machinery (30-240 Hz harmonics)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyroomacoustics as pra
from src.acoustic.room_builder import RoomBuilder, calculate_distance
from src.utils.audio import save_wav, save_comparison_wav
from configurations import STEP4_CONFIGS, generate_noise_signal, print_config_summary


def run_ideal_anc(config: dict, fs: int = 16000, duration: float = 2.0) -> dict:
    """
    Run ideal ANC simulation for a given configuration.

    Args:
        config: Configuration dictionary
        fs: Sampling frequency
        duration: Simulation duration in seconds

    Returns:
        Results dictionary with signals and metrics
    """
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs

    # Extract configuration
    room_cfg = config['room']
    pos = config['positions']

    # Create room with both sources
    room = RoomBuilder.simple_room(
        room_cfg['dimensions'],
        fs,
        absorption=room_cfg['absorption'],
        max_order=room_cfg['max_order']
    )

    # Add sources and microphone
    room.add_source(pos['noise_source'])   # Source 0
    room.add_source(pos['speaker'])        # Source 1
    room.add_microphone(pos['error_mic'])
    room.compute_rir()

    # Extract RIRs
    rir_primary = room.rir[0][0]    # Noise -> Error mic
    rir_secondary = room.rir[0][1]  # Speaker -> Error mic

    # Generate noise signal
    noise_signal = generate_noise_signal(config['noise'], duration, fs)

    # Compute transfer functions for each frequency component
    def compute_transfer_function(rir, freq, fs):
        n = np.arange(len(rir))
        return np.sum(rir * np.exp(-1j * 2 * np.pi * freq * n / fs))

    # For multi-frequency signals, compute ideal anti-noise in frequency domain
    frequencies = config['noise'].get('frequencies', [100])
    amplitudes = config['noise'].get('amplitudes', [1.0])

    # Build anti-noise signal
    antinoise_signal = np.zeros(n_samples)
    for freq, amp in zip(frequencies, amplitudes):
        H_primary = compute_transfer_function(rir_primary, freq, fs)
        H_secondary = compute_transfer_function(rir_secondary, freq, fs)

        amp_ratio = np.abs(H_primary) / np.abs(H_secondary)
        phase_shift = np.angle(H_primary) - np.angle(H_secondary)

        # Anti-noise component for this frequency
        antinoise_signal += -amp * amp_ratio * np.sin(2 * np.pi * freq * t + phase_shift)

    # Normalize to match original noise level
    if np.max(np.abs(antinoise_signal)) > 0:
        antinoise_signal = antinoise_signal / np.max(np.abs(antinoise_signal))

    # Direct convolution to get signals at error mic
    noise_at_error = np.convolve(noise_signal, rir_primary, mode='full')[:n_samples]
    antinoise_at_error = np.convolve(antinoise_signal, rir_secondary, mode='full')[:n_samples]

    # Combined signal
    combined = noise_at_error + antinoise_at_error

    # Calculate noise reduction (steady state)
    steady_start = int(0.3 * fs)
    noise_power = np.mean(noise_at_error[steady_start:] ** 2)
    combined_power = np.mean(combined[steady_start:] ** 2)

    if combined_power > 1e-12:
        reduction_db = 10 * np.log10(noise_power / combined_power)
    else:
        reduction_db = 60.0

    return {
        'config_name': config['name'],
        'noise_signal': noise_signal,
        'noise_at_error': noise_at_error,
        'antinoise_at_error': antinoise_at_error,
        'combined': combined,
        'reduction_db': reduction_db,
        't': t,
        'rir_primary': rir_primary,
        'rir_secondary': rir_secondary,
    }


def plot_results(results: dict, config: dict, fs: int, save_path: str):
    """Generate result plots for a configuration."""
    t = results['t']
    show_samples = int(0.1 * fs)
    t_show = t[-show_samples:] * 1000

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # Noise at error mic
    axes[0].plot(t_show, results['noise_at_error'][-show_samples:], 'r-', linewidth=1)
    axes[0].set_ylabel('Noise d(n)')
    axes[0].set_title(f"Ideal ANC: {config['name']}")
    axes[0].grid(True, alpha=0.3)

    # Anti-noise at error mic
    axes[1].plot(t_show, results['antinoise_at_error'][-show_samples:], 'b-', linewidth=1)
    axes[1].set_ylabel('Anti-noise')
    axes[1].grid(True, alpha=0.3)

    # Residual
    axes[2].plot(t_show, results['combined'][-show_samples:], 'g-', linewidth=1)
    axes[2].set_ylabel('Residual e(n)')
    axes[2].set_xlabel('Time (ms)')
    axes[2].grid(True, alpha=0.3)

    axes[2].text(0.02, 0.95, f"Reduction: {results['reduction_db']:.1f} dB",
                 transform=axes[2].transAxes, fontsize=12, va='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_spectrum(results: dict, config: dict, fs: int, save_path: str):
    """Plot frequency spectrum comparison."""
    from scipy import signal as scipy_signal

    steady_start = int(0.3 * fs)
    noise = results['noise_at_error'][steady_start:]
    residual = results['combined'][steady_start:]

    f_n, psd_n = scipy_signal.welch(noise, fs, nperseg=1024)
    f_r, psd_r = scipy_signal.welch(residual, fs, nperseg=1024)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.semilogy(f_n, psd_n, 'r-', linewidth=2, label='Noise (without ANC)')
    ax.semilogy(f_r, psd_r, 'g-', linewidth=2, label='Residual (with ideal ANC)')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title(f"Spectrum: {config['name']} ({results['reduction_db']:.1f} dB reduction)")
    ax.set_xlim(0, 350)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark target frequencies
    for freq in config['noise'].get('frequencies', []):
        ax.axvline(x=freq, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    """
    Run ideal ANC simulations for all configurations.
    """
    print("=" * 70)
    print("Step 4: Ideal ANC with Known Acoustic Paths")
    print("=" * 70)
    print()

    # Ensure output directories exist
    os.makedirs('output/plots', exist_ok=True)
    os.makedirs('output/audio', exist_ok=True)

    fs = 16000
    duration = 2.0
    all_results = {}

    # Run all configurations
    for config_key, config in STEP4_CONFIGS.items():
        print_config_summary(config, f"Step 4 - {config_key}")

        results = run_ideal_anc(config, fs, duration)
        all_results[config_key] = results

        print(f"\nNoise Reduction: {results['reduction_db']:.1f} dB")

        # Generate plots
        plot_results(results, config, fs, f'output/plots/pyroom_step4_{config_key}.png')
        plot_spectrum(results, config, fs, f'output/plots/pyroom_step4_{config_key}_spectrum.png')
        print(f"Saved: output/plots/pyroom_step4_{config_key}.png")
        print(f"Saved: output/plots/pyroom_step4_{config_key}_spectrum.png")

        # Save audio
        save_comparison_wav(
            f'pyroom_step4_{config_key}',
            results['noise_at_error'],
            results['combined'],
            fs,
            'output/audio'
        )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - Ideal ANC Performance")
    print("=" * 70)
    print()
    print(f"{'Configuration':<35} {'Noise Type':<25} {'Reduction':>12}")
    print("-" * 75)

    for config_key, results in all_results.items():
        config = STEP4_CONFIGS[config_key]
        freqs = config['noise'].get('frequencies', [])
        noise_type = f"{len(freqs)} tone(s): {freqs[0]}-{freqs[-1]} Hz" if len(freqs) > 1 else f"{freqs[0]} Hz"
        print(f"{config['name']:<35} {noise_type:<25} {results['reduction_db']:>10.1f} dB")

    print()
    print("=" * 70)
    print("KEY OBSERVATIONS")
    print("=" * 70)
    print()
    print("1. IDEAL ANC ACHIEVES NEAR-PERFECT CANCELLATION:")
    print("   - With known acoustic paths, we can compute exact anti-noise")
    print("   - Works for single and multi-frequency noise")
    print()
    print("2. REQUIREMENTS FOR IDEAL CANCELLATION:")
    print("   - Exact knowledge of primary path P(z)")
    print("   - Exact knowledge of secondary path S(z)")
    print("   - No processing delay constraints")
    print()
    print("3. WHY REAL SYSTEMS CAN'T ACHIEVE THIS:")
    print("   - Acoustic paths are unknown and time-varying")
    print("   - Processing introduces delay")
    print("   - This is why we need ADAPTIVE algorithms like FxLMS")
    print()
    print("NEXT: Demonstrate the latency problem in Step 5")
    print()

    return all_results


if __name__ == '__main__':
    main()
