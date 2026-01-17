"""
Step 3: Superposition and Destructive Interference

Goal: Demonstrate the fundamental principle behind ANC -
      destructive interference between noise and anti-noise.

This simulation demonstrates:
1. Two sound sources (noise + speaker) propagating through a room
2. Signals add linearly at the microphone (superposition principle)
3. When speaker signal = -noise signal (with correct timing), they CANCEL
4. This is the CORE of Active Noise Cancellation!

Key equation:
    p_total = p_noise + p_antinoise
    If p_antinoise = -p_noise, then p_total = 0 (silence!)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyroomacoustics as pra
from src.acoustic.room_builder import RoomBuilder, calculate_distance
from src.acoustic.path_generator import AcousticPathGenerator
from src.utils.audio import save_wav, save_comparison_wav


def main():
    """
    Demonstrate superposition and destructive interference.
    """
    print("=" * 60)
    print("Step 3: Superposition and Destructive Interference")
    print("=" * 60)
    print()

    # Ensure output directory exists
    os.makedirs('output/plots', exist_ok=True)
    os.makedirs('output/audio', exist_ok=True)

    # Parameters
    fs = 16000
    duration = 1.0  # seconds
    n_samples = int(duration * fs)

    # =========================================
    # Setup: Room with Noise Source and Speaker
    # =========================================
    print("Setting up room with noise source and control speaker...")
    print("-" * 50)

    # Create room
    room_dim = [5, 3, 2.5]
    room = RoomBuilder.simple_room(room_dim, fs, absorption=0.3, max_order=3)

    # Positions
    noise_pos = [0.5, 1.5, 1.2]      # Noise source (e.g., engine)
    speaker_pos = [3.0, 1.5, 1.2]   # Control speaker
    error_mic_pos = [4.0, 1.5, 1.2] # Error microphone (listener)

    print(f"Room: {room_dim[0]}m x {room_dim[1]}m x {room_dim[2]}m")
    print(f"Noise source: {noise_pos}")
    print(f"Control speaker: {speaker_pos}")
    print(f"Error microphone: {error_mic_pos}")

    # Calculate distances
    dist_noise_to_mic = calculate_distance(noise_pos, error_mic_pos)
    dist_speaker_to_mic = calculate_distance(speaker_pos, error_mic_pos)

    print(f"\nNoise to mic distance: {dist_noise_to_mic:.2f} m")
    print(f"Speaker to mic distance: {dist_speaker_to_mic:.2f} m")

    # =========================================
    # Setup: Create single room with both sources
    # =========================================
    # IMPORTANT: Using a single room ensures consistent RIRs
    # and using direct convolution avoids pyroomacoustics simulation artifacts

    room = RoomBuilder.simple_room(room_dim, fs, absorption=0.3, max_order=3)
    room.add_source(noise_pos)   # Source 0: Noise
    room.add_source(speaker_pos) # Source 1: Speaker
    room.add_microphone(error_mic_pos)
    room.compute_rir()

    # Extract RIRs from the same room for consistency
    rir_noise = room.rir[0][0]   # Mic 0, Source 0 (noise)
    rir_speaker = room.rir[0][1] # Mic 0, Source 1 (speaker)

    print(f"RIR lengths: noise={len(rir_noise)}, speaker={len(rir_speaker)}")

    # =========================================
    # Experiment 1: Noise Only
    # =========================================
    print()
    print("Experiment 1: Noise Source Only")
    print("-" * 50)

    # Generate noise signal (100 Hz tone)
    t = np.arange(n_samples) / fs
    frequency = 100  # Hz
    noise_signal = np.sin(2 * np.pi * frequency * t)

    # Noise at mic via direct convolution with RIR
    noise_at_mic = np.convolve(noise_signal, rir_noise, mode='full')[:n_samples]

    print(f"Noise signal: {frequency} Hz sine wave")
    print(f"Signal at mic (RMS): {np.sqrt(np.mean(noise_at_mic**2)):.4f}")

    # =========================================
    # Experiment 2: Speaker Only (Anti-noise)
    # =========================================
    print()
    print("Experiment 2: Speaker Only (Anti-noise)")
    print("-" * 50)

    # For accurate cancellation of a single-frequency signal, we need to compute
    # the EXACT transfer function at that frequency from each RIR
    # H(f) = sum(h[n] * exp(-j*2*pi*f*n/fs))

    def compute_transfer_function(rir, freq, fs):
        """Compute complex transfer function at a specific frequency."""
        n = np.arange(len(rir))
        # Discrete-time Fourier transform at the specific frequency
        H = np.sum(rir * np.exp(-1j * 2 * np.pi * freq * n / fs))
        return H

    H_noise = compute_transfer_function(rir_noise, frequency, fs)
    H_speaker = compute_transfer_function(rir_speaker, frequency, fs)

    # Magnitude and phase at target frequency
    gain_noise = np.abs(H_noise)
    phase_noise = np.angle(H_noise)
    gain_speaker = np.abs(H_speaker)
    phase_speaker = np.angle(H_speaker)

    print(f"Noise path at {frequency} Hz:")
    print(f"  Gain: {gain_noise:.4f}, Phase: {np.degrees(phase_noise):.1f}°")
    print(f"Speaker path at {frequency} Hz:")
    print(f"  Gain: {gain_speaker:.4f}, Phase: {np.degrees(phase_speaker):.1f}°")

    # For perfect cancellation at the mic:
    # noise_signal * H_noise + antinoise_signal * H_speaker = 0
    # Therefore: antinoise_signal = -noise_signal * H_noise / H_speaker
    #
    # In time domain for a sine wave:
    # noise at source: sin(2*pi*f*t)
    # noise at mic: gain_noise * sin(2*pi*f*t + phase_noise)
    # We need speaker at mic: -gain_noise * sin(2*pi*f*t + phase_noise)
    # Speaker signal: (-gain_noise/gain_speaker) * sin(2*pi*f*t + phase_noise - phase_speaker)

    amp_ratio = gain_noise / gain_speaker
    phase_shift = phase_noise - phase_speaker

    print(f"\nFor cancellation:")
    print(f"  Amplitude ratio: {amp_ratio:.4f}")
    print(f"  Phase shift: {np.degrees(phase_shift):.1f}°")

    # Create anti-noise: inverted, amplitude-matched, phase-shifted
    # The original noise signal is sin(2*pi*f*t)
    # Anti-noise should be: -amp_ratio * sin(2*pi*f*t + phase_shift)
    antinoise_signal = -amp_ratio * np.sin(2 * np.pi * frequency * t + phase_shift)

    # Anti-noise at mic via direct convolution
    antinoise_at_mic = np.convolve(antinoise_signal, rir_speaker, mode='full')[:n_samples]

    # =========================================
    # Experiment 3: Both Sources - Superposition
    # =========================================
    print()
    print("Experiment 3: Superposition - Noise + Anti-noise")
    print("-" * 50)

    # Superposition: signals add linearly
    combined_at_mic = noise_at_mic + antinoise_at_mic

    # Calculate noise reduction in steady state (skip first 200ms for transients)
    steady_start = int(0.2 * fs)
    noise_power = np.mean(noise_at_mic[steady_start:] ** 2)
    combined_power = np.mean(combined_at_mic[steady_start:] ** 2)

    if combined_power > 1e-10:
        reduction_db = 10 * np.log10(noise_power / combined_power)
    else:
        reduction_db = 60.0

    print(f"Noise only RMS (steady state): {np.sqrt(noise_power):.4f}")
    print(f"Combined RMS (steady state): {np.sqrt(combined_power):.4f}")
    print(f"Noise reduction: {reduction_db:.1f} dB")

    # =========================================
    # Plot Results
    # =========================================

    # Time domain comparison
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # Show last 50ms for clarity
    show_samples = int(0.05 * fs)
    t_show = t[-show_samples:] * 1000

    axes[0].plot(t_show, noise_at_mic[-show_samples:], 'r-', linewidth=1)
    axes[0].set_ylabel('Noise only')
    axes[0].set_title('Superposition: Noise + Anti-noise = Cancellation')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_show, antinoise_at_mic[-show_samples:], 'b-', linewidth=1)
    axes[1].set_ylabel('Anti-noise only')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_show, noise_at_mic[-show_samples:], 'r-', linewidth=1, alpha=0.5, label='Noise')
    axes[2].plot(t_show, antinoise_at_mic[-show_samples:], 'b-', linewidth=1, alpha=0.5, label='Anti-noise')
    axes[2].set_ylabel('Both signals')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t_show, combined_at_mic[-show_samples:], 'g-', linewidth=1)
    axes[3].set_ylabel('Combined\n(Cancelled)')
    axes[3].set_xlabel('Time (ms)')
    axes[3].grid(True, alpha=0.3)

    # Add reduction annotation
    axes[3].text(0.02, 0.95, f'Reduction: {reduction_db:.1f} dB',
                transform=axes[3].transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat'))

    plt.tight_layout()
    plt.savefig('output/plots/pyroom_step3_superposition.png', dpi=150)
    print("\nSaved: output/plots/pyroom_step3_superposition.png")

    # =========================================
    # Experiment 4: Effect of Phase Error
    # =========================================
    print()
    print("Experiment 4: Effect of Phase/Timing Error on Cancellation")
    print("-" * 50)

    # Test different timing errors
    timing_errors_samples = [0, 1, 2, 5, 10, 20]
    reductions = []

    for timing_error in timing_errors_samples:
        # Create time-shifted anti-noise
        shifted_antinoise = np.zeros_like(noise_signal)
        if timing_error == 0:
            shifted_antinoise = antinoise_signal.copy()
        else:
            shifted_antinoise[timing_error:] = antinoise_signal[:-timing_error]

        # Direct convolution (consistent with main experiment)
        shifted_antinoise_at_mic = np.convolve(shifted_antinoise, rir_speaker, mode='full')[:n_samples]
        result = noise_at_mic + shifted_antinoise_at_mic

        result_power = np.mean(result[steady_start:] ** 2)

        if result_power > 1e-10:
            red = 10 * np.log10(noise_power / result_power)
        else:
            red = 60.0

        reductions.append(red)

        timing_ms = timing_error / fs * 1000
        print(f"  Timing error: {timing_error:>3} samples ({timing_ms:>5.2f} ms) -> Reduction: {red:>6.1f} dB")

    # Plot phase error effect
    fig, ax = plt.subplots(figsize=(10, 5))

    timing_ms = np.array(timing_errors_samples) / fs * 1000
    ax.plot(timing_ms, reductions, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Timing Error (ms)')
    ax.set_ylabel('Noise Reduction (dB)')
    ax.set_title('Effect of Timing Error on Cancellation Performance')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # Add annotation
    ax.annotate('Perfect timing', xy=(0, reductions[0]),
               xytext=(0.3, reductions[0] - 5),
               arrowprops=dict(arrowstyle='->', color='green'),
               fontsize=10, color='green')

    plt.tight_layout()
    plt.savefig('output/plots/pyroom_step3_phase_error.png', dpi=150)
    print("\nSaved: output/plots/pyroom_step3_phase_error.png")

    # =========================================
    # Experiment 5: Anechoic Room (Near-Perfect Cancellation)
    # =========================================
    print()
    print("Experiment 5: Anechoic Room (Ideal Case)")
    print("-" * 50)

    # Create anechoic room (max_order=0 means no reflections)
    room_anechoic = RoomBuilder.simple_room(room_dim, fs, absorption=0.99, max_order=0)
    room_anechoic.add_source(noise_pos)   # Source 0
    room_anechoic.add_source(speaker_pos) # Source 1
    room_anechoic.add_microphone(error_mic_pos)
    room_anechoic.compute_rir()

    # Get anechoic RIRs
    rir_noise_anechoic = room_anechoic.rir[0][0]
    rir_speaker_anechoic = room_anechoic.rir[0][1]

    # Compute transfer functions in anechoic case
    H_noise_anechoic = compute_transfer_function(rir_noise_anechoic, frequency, fs)
    H_speaker_anechoic = compute_transfer_function(rir_speaker_anechoic, frequency, fs)

    amp_ratio_anechoic = np.abs(H_noise_anechoic) / np.abs(H_speaker_anechoic)
    phase_shift_anechoic = np.angle(H_noise_anechoic) - np.angle(H_speaker_anechoic)

    antinoise_anechoic = -amp_ratio_anechoic * np.sin(2 * np.pi * frequency * t + phase_shift_anechoic)

    # Direct convolution in anechoic room
    noise_anechoic = np.convolve(noise_signal, rir_noise_anechoic, mode='full')[:n_samples]
    antinoise_anechoic_at_mic = np.convolve(antinoise_anechoic, rir_speaker_anechoic, mode='full')[:n_samples]
    combined_anechoic = noise_anechoic + antinoise_anechoic_at_mic

    # Calculate reduction
    noise_power_anechoic = np.mean(noise_anechoic[steady_start:] ** 2)
    combined_power_anechoic = np.mean(combined_anechoic[steady_start:] ** 2)

    if combined_power_anechoic > 1e-10:
        reduction_anechoic = 10 * np.log10(noise_power_anechoic / combined_power_anechoic)
    else:
        reduction_anechoic = 60.0

    print(f"Anechoic room (no reflections):")
    print(f"  Noise reduction: {reduction_anechoic:.1f} dB")
    print(f"\nComparison:")
    print(f"  Reverberant room: {reduction_db:.1f} dB")
    print(f"  Anechoic room: {reduction_anechoic:.1f} dB")
    print(f"\nReflections add {reduction_anechoic - reduction_db:.1f} dB of difficulty!")

    # =========================================
    # Save Audio Files
    # =========================================
    print()
    print("Saving audio files...")

    save_wav('output/audio/pyroom_step3_noise.wav', noise_at_mic, fs)
    save_wav('output/audio/pyroom_step3_antinoise.wav', antinoise_at_mic, fs)
    save_wav('output/audio/pyroom_step3_cancelled.wav', combined_at_mic, fs)

    # Create comparison audio
    save_comparison_wav(
        'pyroom_step3',
        noise_at_mic,
        combined_at_mic,
        fs,
        'output/audio'
    )

    print("Audio files saved to output/audio/")

    # =========================================
    # Key Observations
    # =========================================
    print()
    print("=" * 60)
    print("KEY OBSERVATIONS")
    print("=" * 60)
    print()
    print("1. SUPERPOSITION PRINCIPLE:")
    print("   - Sound waves add linearly at any point in space")
    print("   - p_total(t) = p_noise(t) + p_speaker(t)")
    print()
    print("2. DESTRUCTIVE INTERFERENCE:")
    print("   - When two waves are equal in amplitude but OPPOSITE in phase,")
    print("   - they cancel: (+A) + (-A) = 0")
    print()
    print("3. TIMING IS CRITICAL:")
    print("   - Anti-noise must arrive at the EXACT same time as noise")
    print("   - Even small timing errors reduce cancellation significantly")
    print(f"   - A 1-sample error ({1/fs*1000:.3f} ms) can degrade performance!")
    print()
    print("4. THE CHALLENGE:")
    print("   - In real systems, we don't know the noise signal in advance")
    print("   - We need to ADAPT the anti-noise in real-time")
    print("   - This is why we need adaptive algorithms like FxLMS!")
    print()
    print("NEXT: Implement ideal ANC with known acoustic paths")
    print()

    plt.show()

    return reduction_db


if __name__ == '__main__':
    main()
