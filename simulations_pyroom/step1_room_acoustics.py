"""
Step 1: Room Acoustics Basics with pyroomacoustics

Goal: Understand how pyroomacoustics models sound propagation using
      the Image Source Method (ISM).

This simulation demonstrates:
1. Creating a room with pyroomacoustics
2. Computing Room Impulse Responses (RIRs)
3. Visualizing the RIR: direct sound, early reflections, late reverb
4. How absorption affects the acoustic response
5. The relationship between distance and delay

Key concepts:
- Room Impulse Response (RIR): The acoustic "fingerprint" of a room
- Direct sound: First arrival (shortest path from source to mic)
- Early reflections: First few bounces off walls
- Late reverberation: Dense overlapping reflections
- Absorption: How much energy walls absorb (vs reflect)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyroomacoustics as pra
from src.acoustic.room_builder import RoomBuilder, calculate_distance
from src.utils.audio import save_wav


def main():
    """
    Main demonstration of room acoustics with pyroomacoustics.
    """
    print("=" * 60)
    print("Step 1: Room Acoustics Basics with pyroomacoustics")
    print("=" * 60)
    print()

    # Ensure output directory exists
    os.makedirs('output/plots', exist_ok=True)
    os.makedirs('output/audio', exist_ok=True)

    # Parameters
    fs = 16000  # Sample rate

    # =========================================
    # Experiment 1: Simple Room and RIR
    # =========================================
    print("Experiment 1: Computing Room Impulse Response (RIR)")
    print("-" * 50)

    # Create a simple room (5m x 4m x 3m)
    room_dim = [5, 4, 3]
    absorption = 0.2  # 20% absorption (moderately reflective)

    room = RoomBuilder.simple_room(
        dimensions=room_dim,
        fs=fs,
        absorption=absorption,
        max_order=3  # Include up to 3rd order reflections
    )

    # Place source and microphone
    source_pos = [1.0, 2.0, 1.5]  # Near one wall
    mic_pos = [4.0, 2.0, 1.5]     # 3 meters away

    room.add_source(source_pos)
    room.add_microphone(mic_pos)

    # Compute RIR
    room.compute_rir()
    rir = room.rir[0][0]

    # Calculate expected direct path delay
    distance = calculate_distance(source_pos, mic_pos)
    expected_delay_samples = int(distance / 343.0 * fs)
    expected_delay_ms = distance / 343.0 * 1000

    print(f"Room dimensions: {room_dim[0]}m x {room_dim[1]}m x {room_dim[2]}m")
    print(f"Source position: {source_pos}")
    print(f"Microphone position: {mic_pos}")
    print(f"Distance: {distance:.2f} m")
    print(f"Expected direct path delay: {expected_delay_ms:.2f} ms ({expected_delay_samples} samples)")
    print(f"RIR length: {len(rir)} samples ({len(rir)/fs*1000:.1f} ms)")

    # Plot RIR
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Full RIR
    t_rir = np.arange(len(rir)) / fs * 1000  # Convert to ms
    axes[0].plot(t_rir, rir, 'b-', linewidth=0.5)
    axes[0].axvline(x=expected_delay_ms, color='r', linestyle='--',
                    label=f'Direct path ({expected_delay_ms:.1f} ms)')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Room Impulse Response (RIR)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Zoomed to first 50ms (early reflections)
    zoom_samples = int(0.05 * fs)
    axes[1].plot(t_rir[:zoom_samples], rir[:zoom_samples], 'b-', linewidth=1)
    axes[1].axvline(x=expected_delay_ms, color='r', linestyle='--',
                    label='Direct path')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Early Part of RIR (first 50ms) - Direct Sound + Early Reflections')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Annotate direct sound
    direct_idx = np.argmax(np.abs(rir[:zoom_samples]))
    axes[1].annotate('Direct\nSound', xy=(t_rir[direct_idx], rir[direct_idx]),
                     xytext=(t_rir[direct_idx] + 5, rir[direct_idx] * 0.8),
                     arrowprops=dict(arrowstyle='->', color='red'),
                     fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig('output/plots/pyroom_step1_rir.png', dpi=150)
    print("\nSaved: output/plots/pyroom_step1_rir.png")

    # =========================================
    # Experiment 2: Effect of Absorption
    # =========================================
    print()
    print("Experiment 2: Effect of Absorption on RIR")
    print("-" * 50)

    absorptions = [0.05, 0.2, 0.5, 0.9]
    labels = ['Very reflective (5%)', 'Moderate (20%)',
              'Absorbent (50%)', 'Nearly anechoic (90%)']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (abs_coef, label) in enumerate(zip(absorptions, labels)):
        # Create room with different absorption
        room = RoomBuilder.simple_room(
            dimensions=room_dim,
            fs=fs,
            absorption=abs_coef,
            max_order=5  # More reflections for comparison
        )
        room.add_source(source_pos)
        room.add_microphone(mic_pos)
        room.compute_rir()
        rir = room.rir[0][0]

        # Plot
        t = np.arange(len(rir)) / fs * 1000
        axes[i].plot(t, rir, 'b-', linewidth=0.5)
        axes[i].set_xlabel('Time (ms)')
        axes[i].set_ylabel('Amplitude')
        axes[i].set_title(f'{label}\nAbsorption = {abs_coef:.0%}')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(0, 100)

        # Calculate RT60 approximation (time for 60dB decay)
        energy = rir ** 2
        cumsum = np.cumsum(energy[::-1])[::-1]
        if cumsum[0] > 0:
            db_decay = 10 * np.log10(cumsum / cumsum[0] + 1e-10)
            idx_60 = np.where(db_decay < -60)[0]
            if len(idx_60) > 0:
                rt60_est = idx_60[0] / fs
                axes[i].text(0.95, 0.95, f'RT60 ≈ {rt60_est*1000:.0f} ms',
                            transform=axes[i].transAxes, ha='right', va='top',
                            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))

        print(f"  {label}: RIR length = {len(rir)} samples")

    plt.suptitle('Effect of Wall Absorption on Room Impulse Response', fontsize=14)
    plt.tight_layout()
    plt.savefig('output/plots/pyroom_step1_absorption.png', dpi=150)
    print("\nSaved: output/plots/pyroom_step1_absorption.png")

    # =========================================
    # Experiment 3: Effect of Reflection Order
    # =========================================
    print()
    print("Experiment 3: Effect of Reflection Order (max_order)")
    print("-" * 50)

    max_orders = [0, 1, 3, 10]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, max_order in enumerate(max_orders):
        room = RoomBuilder.simple_room(
            dimensions=room_dim,
            fs=fs,
            absorption=0.2,
            max_order=max_order
        )
        room.add_source(source_pos)
        room.add_microphone(mic_pos)
        room.compute_rir()
        rir = room.rir[0][0]

        t = np.arange(len(rir)) / fs * 1000
        axes[i].plot(t, rir, 'b-', linewidth=0.5)
        axes[i].set_xlabel('Time (ms)')
        axes[i].set_ylabel('Amplitude')

        if max_order == 0:
            axes[i].set_title(f'max_order = {max_order}\n(Direct sound only)')
        else:
            axes[i].set_title(f'max_order = {max_order}\n(Up to {max_order} reflections)')

        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(0, 100)

        print(f"  max_order={max_order}: RIR length = {len(rir)} samples")

    plt.suptitle('Effect of Reflection Order on RIR Complexity', fontsize=14)
    plt.tight_layout()
    plt.savefig('output/plots/pyroom_step1_reflections.png', dpi=150)
    print("\nSaved: output/plots/pyroom_step1_reflections.png")

    # =========================================
    # Experiment 4: Convolve with Audio Signal
    # =========================================
    print()
    print("Experiment 4: Convolve Signal with RIR")
    print("-" * 50)

    # Create a simple test signal (100 Hz tone)
    duration = 1.0  # seconds
    t = np.arange(int(duration * fs)) / fs
    frequency = 100
    signal = np.sin(2 * np.pi * frequency * t)

    # Get RIRs for different rooms
    room_dry = RoomBuilder.simple_room(room_dim, fs, absorption=0.9, max_order=0)
    room_dry.add_source(source_pos)
    room_dry.add_microphone(mic_pos)
    room_dry.compute_rir()

    room_wet = RoomBuilder.simple_room(room_dim, fs, absorption=0.1, max_order=5)
    room_wet.add_source(source_pos)
    room_wet.add_microphone(mic_pos)
    room_wet.compute_rir()

    # Convolve
    signal_dry = np.convolve(signal, room_dry.rir[0][0], mode='same')
    signal_wet = np.convolve(signal, room_wet.rir[0][0], mode='same')

    # Save audio files
    save_wav('output/audio/pyroom_step1_original.wav', signal, fs)
    save_wav('output/audio/pyroom_step1_dry.wav', signal_dry, fs)
    save_wav('output/audio/pyroom_step1_wet.wav', signal_wet, fs)

    print(f"Saved audio files:")
    print(f"  output/audio/pyroom_step1_original.wav (original signal)")
    print(f"  output/audio/pyroom_step1_dry.wav (nearly anechoic)")
    print(f"  output/audio/pyroom_step1_wet.wav (reverberant)")

    # Plot comparison
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    show_samples = int(0.05 * fs)  # First 50ms
    t_show = np.arange(show_samples) / fs * 1000

    axes[0].plot(t_show, signal[:show_samples], 'b-', linewidth=1)
    axes[0].set_ylabel('Original')
    axes[0].set_title('Signal Comparison: Original vs Room Response')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_show, signal_dry[:show_samples], 'g-', linewidth=1)
    axes[1].set_ylabel('Dry (90% abs)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_show, signal_wet[:show_samples], 'r-', linewidth=1)
    axes[2].set_ylabel('Wet (10% abs)')
    axes[2].set_xlabel('Time (ms)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/plots/pyroom_step1_convolution.png', dpi=150)
    print("\nSaved: output/plots/pyroom_step1_convolution.png")

    # =========================================
    # Key Observations
    # =========================================
    print()
    print("=" * 60)
    print("KEY OBSERVATIONS")
    print("=" * 60)
    print()
    print("1. ROOM IMPULSE RESPONSE (RIR):")
    print("   - The RIR captures how sound propagates in a room")
    print("   - It includes direct sound, reflections, and reverb")
    print("   - Convolving any signal with RIR simulates room acoustics")
    print()
    print("2. DIRECT SOUND:")
    print(f"   - First arrival at t = distance/c = {expected_delay_ms:.2f} ms")
    print("   - This is the PRIMARY PATH in ANC terminology")
    print()
    print("3. ABSORPTION:")
    print("   - Higher absorption = shorter RIR = less reverb")
    print("   - Car interiors have moderate absorption (carpet, seats)")
    print()
    print("4. REFLECTION ORDER:")
    print("   - max_order=0: direct sound only (anechoic)")
    print("   - Higher orders add more realistic room acoustics")
    print("   - Trade-off: accuracy vs computation time")
    print()
    print("NEXT: Use pyroomacoustics to model microphone placement and delays")
    print()

    plt.show()

    return room


if __name__ == '__main__':
    main()
