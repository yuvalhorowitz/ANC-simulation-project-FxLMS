"""
Step 3: Speaker and Wave Interaction (Superposition)

Goal: Demonstrate how a speaker can generate waves that interact with noise waves,
      including DESTRUCTIVE INTERFERENCE (cancellation).

This is THE CORE PRINCIPLE of Active Noise Cancellation:
- If speaker_wave = -noise_wave, they cancel out!
- total_pressure = noise_wave + speaker_wave = 0

This simulation demonstrates:
1. Two waves adding together (superposition)
2. Destructive interference when waves are opposite
3. Perfect cancellation in the ideal case
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulations.step1_wave_basics import AcousticSpace1D
from simulations.step2_virtual_mics import VirtualMicrophone, MicrophoneArray


class VirtualSpeaker:
    """
    A virtual loudspeaker that injects sound into the acoustic space.

    In a real ANC system, the speaker converts electrical signals to
    acoustic pressure waves. Here we simply inject pressure at the
    speaker's position.
    """

    def __init__(self, position: float, name: str = "Speaker"):
        """
        Initialize virtual speaker.

        Args:
            position: Position in meters from left end
            name: Identifier for this speaker
        """
        self.position = position
        self.name = name
        self.output_history = []

    def emit(self, space: AcousticSpace1D, amplitude: float):
        """
        Emit pressure into the acoustic space.

        Args:
            space: The acoustic space
            amplitude: Pressure amplitude to inject
        """
        space.inject_pressure(self.position, amplitude)
        self.output_history.append(amplitude)

    def emit_sine(self, space: AcousticSpace1D, frequency: float, amplitude: float):
        """
        Emit sinusoidal wave.

        Args:
            space: The acoustic space
            frequency: Frequency in Hz
            amplitude: Wave amplitude
        """
        value = amplitude * np.sin(2 * np.pi * frequency * space.time)
        self.emit(space, value)

    def reset(self):
        """Clear history."""
        self.output_history = []


def main():
    """
    Demonstrate wave superposition and destructive interference.
    """
    print("=" * 60)
    print("Step 3: Speaker and Wave Interaction")
    print("=" * 60)
    print()

    # Create acoustic space with absorbing boundaries to avoid reflections
    space = AcousticSpace1D(
        length=2.0,
        num_points=400,
        speed_of_sound=343.0,
        boundary='absorbing'
    )

    print()

    # =========================================
    # Experiment 1: Two waves adding (constructive)
    # =========================================
    print("Experiment 1: Constructive Interference (Waves ADD)")
    print("-" * 50)

    # Setup: Noise source at left, speaker at center
    noise_position = 0.1
    speaker_position = 1.0
    listener_position = 1.5

    mic = VirtualMicrophone(listener_position, "Listener")
    speaker = VirtualSpeaker(speaker_position, "ANC Speaker")

    frequency = 100  # Hz

    print(f"Noise source at: x = {noise_position} m")
    print(f"Speaker at: x = {speaker_position} m")
    print(f"Listener at: x = {listener_position} m")
    print(f"Frequency: {frequency} Hz")
    print()
    print("Playing noise wave + speaker wave IN PHASE (same direction)")

    # Source functions
    def noise_source(space, time):
        space.inject_sine_wave(noise_position, frequency, amplitude=1.0)

    def speaker_in_phase(space, time):
        """Speaker plays wave IN PHASE with noise (they add up)"""
        speaker.emit_sine(space, frequency, amplitude=1.0)

    # Run simulation
    duration = 0.03
    num_steps = int(duration / space.dt)

    for step in range(num_steps):
        noise_source(space, space.time)
        speaker_in_phase(space, space.time)
        mic.record(space)
        space.step()

    t1, signal1 = mic.get_signal()
    max_amplitude_constructive = np.max(np.abs(signal1[-len(signal1)//3:]))
    print(f"Result: Amplitude at listener = {max_amplitude_constructive:.2f} (LOUDER!)")

    # =========================================
    # Experiment 2: Two waves canceling (destructive)
    # =========================================
    print()
    print("Experiment 2: Destructive Interference (Waves CANCEL)")
    print("-" * 50)

    space.reset()
    mic.reset()
    speaker.reset()

    print("Playing noise wave + speaker wave OUT OF PHASE (opposite)")
    print("Speaker signal = NEGATIVE of noise signal")

    # Calculate the delay needed for the speaker signal
    # Noise travels: noise_position -> speaker_position -> listener_position
    # Speaker signal at speaker_position should cancel noise at listener_position
    noise_to_speaker = speaker_position - noise_position
    speaker_to_listener = listener_position - speaker_position
    noise_delay_to_speaker = noise_to_speaker / space.c

    print(f"Delay from noise source to speaker: {noise_delay_to_speaker*1000:.2f} ms")

    # For perfect cancellation at the listener, we need to account for
    # the travel time of both waves to the listener position

    # Simple approach: delay the anti-noise and invert it
    anti_noise_buffer = []

    def noise_source_2(space, time):
        value = np.sin(2 * np.pi * frequency * time)
        space.inject_pressure(noise_position, value)
        anti_noise_buffer.append(value)

    def speaker_anti_noise(space, time):
        """Speaker plays INVERTED and DELAYED noise signal"""
        # Calculate delay in samples
        delay_samples = int(noise_delay_to_speaker / space.dt)

        if len(anti_noise_buffer) > delay_samples:
            # Get the noise value from the past and INVERT it
            delayed_value = anti_noise_buffer[-delay_samples]
            anti_noise = -delayed_value  # NEGATIVE = opposite phase!
            speaker.emit(space, anti_noise)

    # Run simulation
    for step in range(num_steps):
        noise_source_2(space, space.time)
        speaker_anti_noise(space, space.time)
        mic.record(space)
        space.step()

    t2, signal2 = mic.get_signal()
    max_amplitude_destructive = np.max(np.abs(signal2[-len(signal2)//3:]))
    print(f"Result: Amplitude at listener = {max_amplitude_destructive:.2f} (QUIETER!)")

    # Calculate reduction
    if max_amplitude_destructive > 0:
        reduction_db = 20 * np.log10(max_amplitude_constructive / max_amplitude_destructive)
    else:
        reduction_db = float('inf')
    print(f"Noise reduction: {reduction_db:.1f} dB")

    # =========================================
    # Experiment 3: Ideal perfect cancellation
    # =========================================
    print()
    print("Experiment 3: Ideal Perfect Cancellation")
    print("-" * 50)

    space.reset()
    mic.reset()
    speaker.reset()
    anti_noise_buffer.clear()

    # For this demo, we'll use a simpler approach:
    # Place the speaker right at the listener position and inject
    # the exact opposite of the noise wave

    # First, record what the noise alone does at the listener
    print("Recording noise-only scenario...")

    mic_noise_only = VirtualMicrophone(listener_position, "Noise Only")

    for step in range(num_steps):
        noise_source(space, space.time)
        mic_noise_only.record(space)
        space.step()

    t_noise, signal_noise = mic_noise_only.get_signal()

    # Now reset and add perfect anti-noise
    space.reset()
    mic.reset()

    print("Recording noise + perfect anti-noise...")

    mic_with_anc = VirtualMicrophone(listener_position, "With ANC")

    # We'll cheat and use the recorded noise to generate perfect anti-noise
    noise_record_idx = 0

    for step in range(num_steps):
        noise_source(space, space.time)

        # Inject perfect anti-noise at the listener position
        # (This is "cheating" but demonstrates the principle)
        if noise_record_idx < len(signal_noise):
            # Inject opposite pressure right at listener
            perfect_anti_noise = -signal_noise[noise_record_idx] * 0.95  # 95% cancellation
            space.inject_pressure(listener_position - 0.01, perfect_anti_noise)
            noise_record_idx += 1

        mic_with_anc.record(space)
        space.step()

    t_anc, signal_anc = mic_with_anc.get_signal()

    # Plot comparison
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Noise only
    axes[0].plot(t_noise * 1000, signal_noise, 'b-', linewidth=1.5)
    axes[0].set_ylabel('Pressure')
    axes[0].set_title('Noise Only at Listener Position')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-2, 2)

    # With ANC
    axes[1].plot(t_anc * 1000, signal_anc, 'g-', linewidth=1.5)
    axes[1].set_ylabel('Pressure')
    axes[1].set_title('With Active Noise Cancellation (Noise + Anti-noise)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-2, 2)

    # Overlay
    axes[2].plot(t_noise * 1000, signal_noise, 'b-', linewidth=1.5, alpha=0.7, label='Noise')
    axes[2].plot(t_anc * 1000, signal_anc, 'g-', linewidth=1.5, alpha=0.7, label='With ANC')
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_ylabel('Pressure')
    axes[2].set_title('Comparison: Noise vs ANC')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(-2, 2)

    plt.tight_layout()
    plt.savefig('output/plots/step3_cancellation.png', dpi=150)
    print()
    print("Saved: output/plots/step3_cancellation.png")

    # Calculate final reduction
    noise_power = np.mean(signal_noise[-len(signal_noise)//3:]**2)
    anc_power = np.mean(signal_anc[-len(signal_anc)//3:]**2)

    if anc_power > 1e-10:
        reduction_db = 10 * np.log10(noise_power / anc_power)
    else:
        reduction_db = float('inf')

    print(f"Final noise reduction: {reduction_db:.1f} dB")

    # =========================================
    # The Challenge: Real systems have LATENCY
    # =========================================
    print()
    print("=" * 60)
    print("THE CHALLENGE: LATENCY")
    print("=" * 60)
    print()
    print("In the ideal case above, we 'cheated' by knowing the noise in advance.")
    print()
    print("In reality:")
    print("1. We can only measure noise at the REFERENCE microphone")
    print("2. The speaker has LATENCY (electronics + acoustic travel time)")
    print("3. The noise and anti-noise must arrive at the SAME TIME at the listener")
    print()
    print("This is why we need ADAPTIVE algorithms like FxLMS!")
    print("They learn to compensate for the system's latency.")
    print()

    # =========================================
    # Visualize the concept
    # =========================================
    fig, ax = plt.subplots(figsize=(12, 6))

    # Time axis for illustration
    t = np.linspace(0, 2*np.pi, 200)

    # Noise wave
    noise_wave = np.sin(t)

    # Anti-noise (inverted)
    anti_noise = -np.sin(t)

    # Sum (nearly zero)
    total = noise_wave + anti_noise

    ax.plot(t, noise_wave, 'b-', linewidth=2, label='Noise Wave')
    ax.plot(t, anti_noise, 'r--', linewidth=2, label='Anti-noise (inverted)')
    ax.plot(t, total, 'g-', linewidth=3, label='Sum = Near Silence!')

    ax.fill_between(t, total - 0.1, total + 0.1, alpha=0.3, color='green')

    ax.set_xlabel('Phase')
    ax.set_ylabel('Amplitude')
    ax.set_title('Principle of Destructive Interference: Noise + (-Noise) = Silence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.5, 1.5)

    plt.tight_layout()
    plt.savefig('output/plots/step3_superposition_principle.png', dpi=150)
    print("Saved: output/plots/step3_superposition_principle.png")

    # =========================================
    # Summary diagram
    # =========================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Constructive
    ax = axes[0]
    ax.plot(t, noise_wave, 'b-', linewidth=2, label='Wave 1')
    ax.plot(t, noise_wave, 'r--', linewidth=2, label='Wave 2 (in phase)')
    ax.plot(t, 2*noise_wave, 'purple', linewidth=2, label='Sum = 2x')
    ax.set_title('Constructive Interference\n(Waves in phase = LOUDER)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-2.5, 2.5)

    # Destructive
    ax = axes[1]
    ax.plot(t, noise_wave, 'b-', linewidth=2, label='Noise')
    ax.plot(t, anti_noise, 'r--', linewidth=2, label='Anti-noise (opposite)')
    ax.plot(t, total, 'g-', linewidth=3, label='Sum = ~0')
    ax.set_title('Destructive Interference\n(Waves opposite = SILENCE)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-2.5, 2.5)

    # Partial
    ax = axes[2]
    partial_cancel = 0.7 * anti_noise  # Imperfect cancellation
    partial_sum = noise_wave + partial_cancel
    ax.plot(t, noise_wave, 'b-', linewidth=2, label='Noise')
    ax.plot(t, partial_cancel, 'r--', linewidth=2, label='Anti-noise (70%)')
    ax.plot(t, partial_sum, 'orange', linewidth=2, label='Partial reduction')
    ax.set_title('Partial Cancellation\n(Real systems: 10-30 dB reduction)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-2.5, 2.5)

    plt.tight_layout()
    plt.savefig('output/plots/step3_interference_types.png', dpi=150)
    print("Saved: output/plots/step3_interference_types.png")

    print()
    print("=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print()
    print("Now that we understand wave superposition and cancellation,")
    print("the next step is to build a simple ANC system:")
    print()
    print("Step 4: Simple ANC without latency (ideal case)")
    print("Step 5: Add latency → see cancellation fail")
    print("Step 6: Implement FxLMS → cancellation works again!")
    print()

    plt.show()

    return space


if __name__ == '__main__':
    main()
