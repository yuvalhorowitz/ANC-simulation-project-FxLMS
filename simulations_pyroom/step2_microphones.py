"""
Step 2: Microphone Placement and Delay Measurement

Goal: Understand how distance affects signal arrival time and
      how to measure acoustic delays using pyroomacoustics.

This simulation demonstrates:
1. Placing multiple microphones at different distances
2. Computing RIRs for each microphone position
3. Measuring arrival time differences
4. Understanding phase relationships
5. The concept of reference microphone (upstream) vs error microphone (downstream)

Key concepts for ANC:
- Reference mic: Captures noise BEFORE it reaches the listener
- Error mic: Measures residual noise at the listener position
- The TIME DIFFERENCE between ref and error mics is critical for ANC
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyroomacoustics as pra
from src.acoustic.room_builder import RoomBuilder, calculate_distance, calculate_delay_samples
from src.utils.audio import save_wav


def find_first_arrival(rir: np.ndarray, threshold: float = 0.1) -> int:
    """
    Find the sample index of first significant arrival in RIR.

    Args:
        rir: Room impulse response
        threshold: Detection threshold relative to peak

    Returns:
        Sample index of first arrival
    """
    peak = np.max(np.abs(rir))
    indices = np.where(np.abs(rir) > threshold * peak)[0]
    return indices[0] if len(indices) > 0 else 0


def main():
    """
    Demonstrate microphone placement and delay measurement.
    """
    print("=" * 60)
    print("Step 2: Microphone Placement and Delay Measurement")
    print("=" * 60)
    print()

    # Ensure output directory exists
    os.makedirs('output/plots', exist_ok=True)
    os.makedirs('output/audio', exist_ok=True)

    # Parameters
    fs = 16000
    c = 343.0  # Speed of sound

    # =========================================
    # Experiment 1: Multiple Microphones at Different Distances
    # =========================================
    print("Experiment 1: Microphones at Different Distances")
    print("-" * 50)

    # Create a moderately reverberant room
    room_dim = [6, 4, 3]
    room = RoomBuilder.simple_room(room_dim, fs, absorption=0.3, max_order=3)

    # Source at one end
    source_pos = [0.5, 2.0, 1.5]

    # Microphones at different distances along the room
    mic_positions = [
        [1.0, 2.0, 1.5],   # 0.5m from source
        [2.0, 2.0, 1.5],   # 1.5m from source
        [3.0, 2.0, 1.5],   # 2.5m from source
        [4.0, 2.0, 1.5],   # 3.5m from source
        [5.0, 2.0, 1.5],   # 4.5m from source
    ]

    # Add source
    room.add_source(source_pos)

    # Add all microphones
    mic_array = np.array(mic_positions).T  # Shape: (3, n_mics)
    room.add_microphone_array(pra.MicrophoneArray(mic_array, fs=fs))

    # Compute RIRs
    room.compute_rir()

    # Analyze each microphone
    print(f"\nSource position: {source_pos}")
    print(f"{'Mic':>4} {'Position':>20} {'Distance':>10} {'Expected':>12} {'Measured':>12} {'Error':>8}")
    print("-" * 72)

    measured_delays = []
    expected_delays = []

    for i, mic_pos in enumerate(mic_positions):
        rir = room.rir[i][0]

        # Calculate expected delay
        distance = calculate_distance(source_pos, mic_pos)
        expected_samples = calculate_delay_samples(distance, fs, c)
        expected_ms = distance / c * 1000

        # Measure actual delay from RIR
        measured_samples = find_first_arrival(rir)
        measured_ms = measured_samples / fs * 1000

        error_ms = measured_ms - expected_ms

        measured_delays.append(measured_samples)
        expected_delays.append(expected_samples)

        print(f"{i:>4} {str(mic_pos):>20} {distance:>8.2f}m {expected_ms:>10.2f}ms {measured_ms:>10.2f}ms {error_ms:>+6.2f}ms")

    # Plot RIRs
    fig, axes = plt.subplots(len(mic_positions), 1, figsize=(12, 2*len(mic_positions)), sharex=True)

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(mic_positions)))

    for i, mic_pos in enumerate(mic_positions):
        rir = room.rir[i][0]
        t = np.arange(len(rir)) / fs * 1000  # ms

        distance = calculate_distance(source_pos, mic_pos)

        axes[i].plot(t, rir, color=colors[i], linewidth=0.8)
        axes[i].axvline(x=measured_delays[i]/fs*1000, color='r', linestyle='--',
                       alpha=0.7, label='First arrival')
        axes[i].set_ylabel(f'{distance:.1f}m')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(0, 50)

    axes[0].set_title('RIR at Different Distances from Source')
    axes[-1].set_xlabel('Time (ms)')

    plt.tight_layout()
    plt.savefig('output/plots/pyroom_step2_distances.png', dpi=150)
    print("\nSaved: output/plots/pyroom_step2_distances.png")

    # =========================================
    # Experiment 2: Reference vs Error Microphone
    # =========================================
    print()
    print("Experiment 2: Reference vs Error Microphone (ANC Setup)")
    print("-" * 50)

    # Create ANC-like configuration
    room = RoomBuilder.simple_room([5, 3, 2.5], fs, absorption=0.25, max_order=3)

    # Noise source at front
    noise_pos = [0.5, 1.5, 1.2]

    # Reference mic (upstream, close to source)
    ref_mic_pos = [1.0, 1.5, 1.2]

    # Error mic (downstream, listener position)
    error_mic_pos = [4.0, 1.5, 1.2]

    # Add components
    room.add_source(noise_pos)
    mic_array = np.array([ref_mic_pos, error_mic_pos]).T
    room.add_microphone_array(pra.MicrophoneArray(mic_array, fs=fs))

    room.compute_rir()

    rir_ref = room.rir[0][0]
    rir_error = room.rir[1][0]

    # Calculate delays
    dist_to_ref = calculate_distance(noise_pos, ref_mic_pos)
    dist_to_error = calculate_distance(noise_pos, error_mic_pos)

    delay_ref_samples = find_first_arrival(rir_ref)
    delay_error_samples = find_first_arrival(rir_error)

    delay_difference = delay_error_samples - delay_ref_samples
    delay_difference_ms = delay_difference / fs * 1000

    print(f"Noise source: {noise_pos}")
    print(f"Reference mic: {ref_mic_pos} (distance: {dist_to_ref:.2f}m)")
    print(f"Error mic: {error_mic_pos} (distance: {dist_to_error:.2f}m)")
    print()
    print(f"Delay to reference mic: {delay_ref_samples} samples ({delay_ref_samples/fs*1000:.2f}ms)")
    print(f"Delay to error mic: {delay_error_samples} samples ({delay_error_samples/fs*1000:.2f}ms)")
    print(f"DELAY DIFFERENCE: {delay_difference} samples ({delay_difference_ms:.2f}ms)")
    print()
    print("This delay difference is CRITICAL for ANC:")
    print("  - It tells us how much TIME we have to compute anti-noise")
    print("  - If processing takes longer than this, cancellation fails!")

    # Plot comparison
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    t = np.arange(max(len(rir_ref), len(rir_error))) / fs * 1000

    axes[0].plot(t[:len(rir_ref)], rir_ref, 'b-', linewidth=1, label='Reference mic RIR')
    axes[0].axvline(x=delay_ref_samples/fs*1000, color='r', linestyle='--', label='First arrival')
    axes[0].set_ylabel('Reference Mic')
    axes[0].set_title('ANC Configuration: Reference vs Error Microphone')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 50)

    axes[1].plot(t[:len(rir_error)], rir_error, 'g-', linewidth=1, label='Error mic RIR')
    axes[1].axvline(x=delay_error_samples/fs*1000, color='r', linestyle='--', label='First arrival')
    axes[1].set_ylabel('Error Mic')
    axes[1].set_xlabel('Time (ms)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Add annotation showing delay difference
    axes[1].annotate('', xy=(delay_ref_samples/fs*1000, 0.5),
                    xytext=(delay_error_samples/fs*1000, 0.5),
                    arrowprops=dict(arrowstyle='<->', color='red'))
    axes[1].text((delay_ref_samples + delay_error_samples)/2/fs*1000, 0.6,
                f'Δt = {delay_difference_ms:.1f}ms',
                ha='center', fontsize=12, color='red')

    plt.tight_layout()
    plt.savefig('output/plots/pyroom_step2_anc_mics.png', dpi=150)
    print("\nSaved: output/plots/pyroom_step2_anc_mics.png")

    # =========================================
    # Experiment 3: Simulate Signal at Both Microphones
    # =========================================
    print()
    print("Experiment 3: Signal Arrival at Both Microphones")
    print("-" * 50)

    # Create a 100 Hz test tone
    duration = 0.5  # seconds
    t_signal = np.arange(int(duration * fs)) / fs
    frequency = 100  # Hz
    test_signal = np.sin(2 * np.pi * frequency * t_signal)

    # Convolve with RIRs
    signal_at_ref = np.convolve(test_signal, rir_ref, mode='full')[:len(test_signal)]
    signal_at_error = np.convolve(test_signal, rir_error, mode='full')[:len(test_signal)]

    # Save audio
    save_wav('output/audio/pyroom_step2_at_ref.wav', signal_at_ref, fs)
    save_wav('output/audio/pyroom_step2_at_error.wav', signal_at_error, fs)
    print("Saved audio files for comparison")

    # Plot signals
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    show_samples = int(0.05 * fs)  # First 50ms
    t_show = np.arange(show_samples) / fs * 1000

    axes[0].plot(t_show, test_signal[:show_samples], 'k-', linewidth=1)
    axes[0].set_ylabel('Source')
    axes[0].set_title('100 Hz Signal: Source vs Microphone Recordings')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_show, signal_at_ref[:show_samples], 'b-', linewidth=1)
    axes[1].set_ylabel('At Ref Mic')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_show, signal_at_error[:show_samples], 'g-', linewidth=1)
    axes[2].set_ylabel('At Error Mic')
    axes[2].set_xlabel('Time (ms)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/plots/pyroom_step2_signals.png', dpi=150)
    print("\nSaved: output/plots/pyroom_step2_signals.png")

    # =========================================
    # Experiment 4: Phase Difference at Microphones
    # =========================================
    print()
    print("Experiment 4: Phase Difference Between Microphones")
    print("-" * 50)

    # Calculate phase difference due to delay
    wavelength = c / frequency  # meters
    path_diff = dist_to_error - dist_to_ref
    phase_diff_deg = (path_diff / wavelength) * 360

    print(f"Frequency: {frequency} Hz")
    print(f"Wavelength: {wavelength:.2f} m")
    print(f"Path difference: {path_diff:.2f} m")
    print(f"Phase difference: {phase_diff_deg:.1f} degrees")
    print()
    print("This phase relationship is key to understanding how")
    print("anti-noise must be timed to achieve cancellation!")

    # =========================================
    # Key Observations
    # =========================================
    print()
    print("=" * 60)
    print("KEY OBSERVATIONS")
    print("=" * 60)
    print()
    print("1. DELAY = DISTANCE / SPEED OF SOUND:")
    print(f"   - Sound travels at ~343 m/s")
    print(f"   - 1 meter ≈ 2.9 ms delay ≈ {int(343/fs*1000)} samples at {fs} Hz")
    print()
    print("2. REFERENCE vs ERROR MICROPHONE:")
    print("   - Reference mic is UPSTREAM (closer to noise source)")
    print("   - Error mic is DOWNSTREAM (at listener position)")
    print(f"   - Time difference: {delay_difference_ms:.2f} ms")
    print()
    print("3. ANC TIMING CONSTRAINT:")
    print("   - We measure noise at reference mic")
    print("   - Must generate anti-noise BEFORE it arrives at error mic")
    print(f"   - Available time budget: {delay_difference_ms:.2f} ms")
    print()
    print("4. PHASE RELATIONSHIP:")
    print("   - Different frequencies have different phase at each mic")
    print("   - This affects how we design the cancellation filter")
    print()
    print("NEXT: Demonstrate destructive interference with speaker")
    print()

    plt.show()

    return room


if __name__ == '__main__':
    main()
