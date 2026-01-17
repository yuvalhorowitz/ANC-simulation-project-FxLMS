"""
Step 4: Simple ANC System - Ideal Case (No Latency)

Goal: Build the simplest possible Active Noise Cancellation system.

Scenario (like noise-canceling headphones):
- 1D space (tube from outside to ear)
- 1 noise source (external noise)
- 1 reference microphone (picks up noise before it reaches ear)
- 1 speaker (near the ear)
- 1 error microphone (at the ear)

In this IDEAL case:
- No latency: speaker responds INSTANTLY
- Perfect knowledge: we know exactly what the noise is
- Result: Perfect cancellation

This demonstrates the GOAL of ANC before we add real-world complications.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulations.step1_wave_basics import AcousticSpace1D
from simulations.step2_virtual_mics import VirtualMicrophone
from simulations.step3_speaker_interaction import VirtualSpeaker


class SimpleANCSystem:
    """
    Simple ANC system with reference mic, speaker, and error mic.

    This is the basic feedforward ANC structure:

    Noise Source --> [Reference Mic] --> Controller --> [Speaker] --> [Error Mic/Ear]
                          |                                              |
                          +---------- Primary Path -------------------->+

    The controller's job: Generate anti-noise that cancels the noise at the error mic.
    """

    def __init__(
        self,
        space: AcousticSpace1D,
        noise_position: float,
        reference_mic_position: float,
        speaker_position: float,
        error_mic_position: float
    ):
        """
        Initialize Simple ANC system.

        Args:
            space: The acoustic space
            noise_position: Position of noise source
            reference_mic_position: Position of reference microphone
            speaker_position: Position of cancellation speaker
            error_mic_position: Position of error microphone (listener's ear)
        """
        self.space = space
        self.noise_pos = noise_position
        self.ref_mic_pos = reference_mic_position
        self.speaker_pos = speaker_position
        self.error_mic_pos = error_mic_position

        # Create components
        self.reference_mic = VirtualMicrophone(reference_mic_position, "Reference")
        self.error_mic = VirtualMicrophone(error_mic_position, "Error")
        self.speaker = VirtualSpeaker(speaker_position, "ANC Speaker")

        # Calculate delays (for the controller to use)
        self.speed_of_sound = space.c

        # Delay from reference mic to error mic (primary path delay)
        self.primary_delay = (error_mic_position - reference_mic_position) / self.speed_of_sound

        # Delay from reference mic to speaker
        self.ref_to_speaker_delay = (speaker_position - reference_mic_position) / self.speed_of_sound

        # Delay from speaker to error mic (secondary path delay)
        self.secondary_delay = (error_mic_position - speaker_position) / self.speed_of_sound

        print("Simple ANC System initialized:")
        print(f"  Noise source: x = {noise_position} m")
        print(f"  Reference mic: x = {reference_mic_position} m")
        print(f"  Speaker: x = {speaker_position} m")
        print(f"  Error mic (ear): x = {error_mic_position} m")
        print()
        print("Path delays:")
        print(f"  Primary path (ref→ear): {self.primary_delay*1000:.2f} ms")
        print(f"  Ref→Speaker: {self.ref_to_speaker_delay*1000:.2f} ms")
        print(f"  Secondary path (speaker→ear): {self.secondary_delay*1000:.2f} ms")

        # History buffers
        self.reference_history = []
        self.error_history_no_anc = []
        self.error_history_with_anc = []

    def run_no_anc(self, noise_func, duration: float):
        """
        Run simulation WITHOUT ANC (baseline).

        Args:
            noise_func: Function that generates noise at each step
            duration: Simulation duration

        Returns:
            Error signal (what listener hears without ANC)
        """
        self.space.reset()
        self.reference_mic.reset()
        self.error_mic.reset()

        num_steps = int(duration / self.space.dt)

        for step in range(num_steps):
            # Generate noise
            noise_func(self.space, self.space.time)

            # Record at microphones
            self.reference_mic.record(self.space)
            self.error_mic.record(self.space)

            # Advance simulation
            self.space.step()

        t, error_signal = self.error_mic.get_signal()
        self.error_history_no_anc = error_signal.copy()

        return t, error_signal

    def run_ideal_anc(self, noise_func, duration: float):
        """
        Run simulation WITH IDEAL ANC (instant response, perfect knowledge).

        In this ideal case:
        - The speaker knows exactly what pressure will arrive at the error mic
        - The speaker responds with zero latency
        - Result: Perfect cancellation

        Args:
            noise_func: Function that generates noise
            duration: Simulation duration

        Returns:
            Error signal (what listener hears with ideal ANC)
        """
        self.space.reset()
        self.reference_mic.reset()
        self.error_mic.reset()
        self.speaker.reset()

        num_steps = int(duration / self.space.dt)

        # Buffer to store reference signal for delay
        ref_buffer = []

        # Calculate delay from reference to speaker (in samples)
        # The anti-noise needs to be timed so it arrives at error mic
        # at the same time as the noise
        #
        # Noise path: ref_mic -> error_mic = primary_delay
        # Anti-noise path: ref_mic -> speaker -> error_mic
        #                = ref_to_speaker_delay + secondary_delay
        #
        # For cancellation: total delays must match
        # So: controller_delay = primary_delay - secondary_delay - processing_delay

        total_delay_samples = int(self.ref_to_speaker_delay / self.space.dt)

        for step in range(num_steps):
            # Generate noise
            noise_func(self.space, self.space.time)

            # Record reference signal
            ref_value = self.space.get_pressure_at(self.ref_mic_pos)
            ref_buffer.append(ref_value)
            self.reference_mic.record(self.space)

            # IDEAL ANC: Use delayed & inverted reference as anti-noise
            # This assumes we know the exact gain and delay
            if len(ref_buffer) > total_delay_samples:
                # Get the reference value from the appropriate time in the past
                delayed_ref = ref_buffer[-(total_delay_samples + 1)]

                # The anti-noise should be the NEGATIVE of what the noise
                # will be at the error mic position
                # In ideal case, we assume perfect propagation, so just invert
                anti_noise = -delayed_ref

                self.speaker.emit(self.space, anti_noise)

            # Record error signal
            self.error_mic.record(self.space)

            # Advance simulation
            self.space.step()

        t, error_signal = self.error_mic.get_signal()
        self.error_history_with_anc = error_signal.copy()

        return t, error_signal


def main():
    """
    Demonstrate ideal ANC system.
    """
    print("=" * 60)
    print("Step 4: Simple ANC System - Ideal Case")
    print("=" * 60)
    print()

    # Create acoustic space
    space = AcousticSpace1D(
        length=0.5,           # 50 cm (like a headphone cup)
        num_points=200,
        speed_of_sound=343.0,
        boundary='absorbing'
    )

    print()

    # Create ANC system (headphone-like configuration)
    # Noise comes from outside, reference mic near outer shell,
    # speaker inside, error mic at ear position
    anc = SimpleANCSystem(
        space=space,
        noise_position=0.05,           # Noise from outside (5cm from left)
        reference_mic_position=0.10,   # Reference mic at 10cm
        speaker_position=0.35,         # Speaker at 35cm
        error_mic_position=0.45        # Ear at 45cm
    )

    print()

    # Define noise source (100 Hz sine wave)
    frequency = 100  # Hz
    print(f"Noise: {frequency} Hz sine wave")
    print()

    def noise_source(space, time):
        amplitude = 1.0
        value = amplitude * np.sin(2 * np.pi * frequency * time)
        space.inject_pressure(anc.noise_pos, value)

    # Run without ANC
    print("Running simulation WITHOUT ANC...")
    duration = 0.05  # 50 ms
    t, error_no_anc = anc.run_no_anc(noise_source, duration)
    print(f"  Max amplitude at ear: {np.max(np.abs(error_no_anc)):.3f}")

    # Run with ideal ANC
    print("Running simulation WITH IDEAL ANC...")
    t, error_with_anc = anc.run_ideal_anc(noise_source, duration)
    print(f"  Max amplitude at ear: {np.max(np.abs(error_with_anc)):.3f}")

    # Calculate noise reduction
    # Use RMS of steady-state portion
    steady_start = len(error_no_anc) // 2
    power_no_anc = np.mean(error_no_anc[steady_start:]**2)
    power_with_anc = np.mean(error_with_anc[steady_start:]**2)

    if power_with_anc > 1e-10:
        reduction_db = 10 * np.log10(power_no_anc / power_with_anc)
    else:
        reduction_db = float('inf')

    print()
    print(f"Noise Reduction: {reduction_db:.1f} dB")

    # =========================================
    # Plot results
    # =========================================
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Reference signal
    t_ref, ref_signal = anc.reference_mic.get_signal()
    axes[0].plot(t_ref * 1000, ref_signal, 'b-', linewidth=1.5)
    axes[0].set_ylabel('Reference Mic')
    axes[0].set_title('Reference Signal (Noise detected by reference microphone)')
    axes[0].grid(True, alpha=0.3)

    # Error without ANC
    axes[1].plot(t * 1000, error_no_anc, 'r-', linewidth=1.5)
    axes[1].set_ylabel('Error Mic (No ANC)')
    axes[1].set_title('What Listener Hears WITHOUT ANC')
    axes[1].grid(True, alpha=0.3)

    # Error with ANC
    axes[2].plot(t * 1000, error_with_anc, 'g-', linewidth=1.5)
    axes[2].set_ylabel('Error Mic (With ANC)')
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_title(f'What Listener Hears WITH Ideal ANC (Reduction: {reduction_db:.1f} dB)')
    axes[2].grid(True, alpha=0.3)

    # Match y-axis scales for comparison
    max_amp = max(np.max(np.abs(error_no_anc)), np.max(np.abs(error_with_anc))) * 1.2
    axes[1].set_ylim(-max_amp, max_amp)
    axes[2].set_ylim(-max_amp, max_amp)

    plt.tight_layout()
    plt.savefig('output/plots/step4_ideal_anc.png', dpi=150)
    print()
    print("Saved: output/plots/step4_ideal_anc.png")

    # =========================================
    # Overlay comparison
    # =========================================
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(t * 1000, error_no_anc, 'r-', linewidth=2, alpha=0.7, label='Without ANC')
    ax.plot(t * 1000, error_with_anc, 'g-', linewidth=2, alpha=0.7, label='With Ideal ANC')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Pressure at Ear')
    ax.set_title(f'Comparison: Noise at Ear with and without ANC ({reduction_db:.1f} dB reduction)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/plots/step4_comparison.png', dpi=150)
    print("Saved: output/plots/step4_comparison.png")

    # =========================================
    # Key points
    # =========================================
    print()
    print("=" * 60)
    print("IDEAL ANC - KEY POINTS")
    print("=" * 60)
    print()
    print("In this IDEAL case:")
    print("1. We have PERFECT knowledge of the noise")
    print("2. The system has ZERO latency (instant response)")
    print("3. We know exactly how sound propagates (perfect model)")
    print()
    print("Result: Near-perfect cancellation!")
    print()
    print("But in REALITY:")
    print("- We don't know the noise in advance")
    print("- Electronic systems have latency (processing time)")
    print("- The acoustic paths are complex and change over time")
    print()
    print("Next step: Add latency and see what happens!")
    print()

    plt.show()

    return anc


if __name__ == '__main__':
    main()
