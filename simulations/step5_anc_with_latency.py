"""
Step 5: ANC with Latency - Why FxLMS is Needed

Goal: Demonstrate why simple cancellation fails when there is latency,
      and introduce the concept of the "secondary path".

The Problem:
- Real systems have LATENCY (processing time, speaker response, etc.)
- If we simply invert the reference signal and play it, it arrives LATE
- Late anti-noise doesn't cancel - it may even make things WORSE!

This is why we need adaptive algorithms like FxLMS - they learn to
compensate for the secondary path delay and characteristics.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulations.step1_wave_basics import AcousticSpace1D
from simulations.step2_virtual_mics import VirtualMicrophone
from simulations.step3_speaker_interaction import VirtualSpeaker


class ANCSystemWithLatency:
    """
    ANC system that includes realistic latency in the secondary path.

    The "secondary path" S(z) includes:
    - A/D converter delay
    - Digital signal processing delay
    - D/A converter delay
    - Amplifier delay
    - Speaker acoustic response (not instantaneous!)
    - Acoustic propagation from speaker to error mic

    If we don't account for S(z), cancellation fails!
    """

    def __init__(
        self,
        space: AcousticSpace1D,
        noise_position: float,
        reference_mic_position: float,
        speaker_position: float,
        error_mic_position: float,
        processing_delay_ms: float = 1.0,  # Electronic processing delay
        speaker_delay_ms: float = 0.5      # Speaker response delay
    ):
        """
        Initialize ANC system with latency.

        Args:
            space: The acoustic space
            noise_position: Position of noise source
            reference_mic_position: Position of reference microphone
            speaker_position: Position of speaker
            error_mic_position: Position of error mic (ear)
            processing_delay_ms: Electronic/DSP delay in milliseconds
            speaker_delay_ms: Speaker mechanical response delay
        """
        self.space = space
        self.noise_pos = noise_position
        self.ref_mic_pos = reference_mic_position
        self.speaker_pos = speaker_position
        self.error_mic_pos = error_mic_position

        # Components
        self.reference_mic = VirtualMicrophone(reference_mic_position, "Reference")
        self.error_mic = VirtualMicrophone(error_mic_position, "Error")
        self.speaker = VirtualSpeaker(speaker_position, "ANC Speaker")

        # Delays
        self.processing_delay = processing_delay_ms / 1000  # Convert to seconds
        self.speaker_delay = speaker_delay_ms / 1000

        # Acoustic delays
        self.acoustic_delay_ref_to_ear = (error_mic_position - reference_mic_position) / space.c
        self.acoustic_delay_speaker_to_ear = (error_mic_position - speaker_position) / space.c

        # TOTAL secondary path delay (this is what causes the problem!)
        self.secondary_path_delay = (
            self.processing_delay +
            self.speaker_delay +
            self.acoustic_delay_speaker_to_ear
        )

        print("ANC System with Latency:")
        print(f"  Processing delay: {processing_delay_ms:.1f} ms")
        print(f"  Speaker delay: {speaker_delay_ms:.1f} ms")
        print(f"  Acoustic (speaker→ear): {self.acoustic_delay_speaker_to_ear*1000:.2f} ms")
        print(f"  TOTAL secondary path: {self.secondary_path_delay*1000:.2f} ms")
        print()
        print(f"  Primary path (ref→ear): {self.acoustic_delay_ref_to_ear*1000:.2f} ms")
        print()

        # Check if we have enough "headroom"
        headroom = self.acoustic_delay_ref_to_ear - self.secondary_path_delay
        if headroom > 0:
            print(f"  Headroom: {headroom*1000:.2f} ms (OK)")
        else:
            print(f"  Headroom: {headroom*1000:.2f} ms (NEGATIVE - cancellation difficult!)")

        # Delay line for simulating processing delay
        self.delay_line = []
        self.delay_samples = int(
            (self.processing_delay + self.speaker_delay) / space.dt
        )

    def run_no_anc(self, noise_func, duration: float):
        """Run without ANC (baseline)."""
        self.space.reset()
        self.reference_mic.reset()
        self.error_mic.reset()

        num_steps = int(duration / self.space.dt)

        for step in range(num_steps):
            noise_func(self.space, self.space.time)
            self.reference_mic.record(self.space)
            self.error_mic.record(self.space)
            self.space.step()

        return self.error_mic.get_signal()

    def run_naive_anc(self, noise_func, duration: float):
        """
        Run with NAIVE ANC: simply invert reference signal.

        This IGNORES the secondary path delay and will fail!
        """
        self.space.reset()
        self.reference_mic.reset()
        self.error_mic.reset()
        self.speaker.reset()
        self.delay_line = []

        num_steps = int(duration / self.space.dt)

        for step in range(num_steps):
            # Generate noise
            noise_func(self.space, self.space.time)

            # Read reference mic
            ref_value = self.space.get_pressure_at(self.ref_mic_pos)
            self.reference_mic.record(self.space)

            # NAIVE approach: Invert and send to speaker immediately
            # But there's a processing delay...
            self.delay_line.append(-ref_value)  # Inverted signal

            # Output from delay line (simulates processing delay)
            if len(self.delay_line) > self.delay_samples:
                anti_noise = self.delay_line.pop(0)
                self.speaker.emit(self.space, anti_noise)

            # Record error
            self.error_mic.record(self.space)

            self.space.step()

        return self.error_mic.get_signal()

    def run_delay_compensated_anc(self, noise_func, duration: float):
        """
        Run with delay-compensated ANC.

        We adjust the timing to account for the secondary path.
        This is closer to what FxLMS achieves, but with perfect knowledge.
        """
        self.space.reset()
        self.reference_mic.reset()
        self.error_mic.reset()
        self.speaker.reset()

        num_steps = int(duration / self.space.dt)

        # Buffer to store reference signal
        ref_buffer = []

        # Calculate the correct delay:
        # Anti-noise must arrive at error mic at the same time as noise
        # Noise: leaves ref_mic, takes primary_path_delay to reach error_mic
        # Anti-noise: processed (processing_delay), speaker responds (speaker_delay),
        #             sound travels speaker→ear (acoustic_delay)
        # So the controller should delay by:
        # primary_path_delay - secondary_path_delay
        # But this assumes we OUTPUT the anti-noise at the right time

        # Actually, let's think more carefully:
        # When ref_mic sees noise at time t:
        #   Noise arrives at error_mic at time: t + acoustic_delay_ref_to_ear
        # If we immediately start processing:
        #   Anti-noise is output from speaker at: t + processing_delay + speaker_delay
        #   Anti-noise arrives at error_mic at: t + processing_delay + speaker_delay + acoustic_speaker_to_ear
        #                                     = t + secondary_path_delay
        # For cancellation: secondary_path_delay == acoustic_delay_ref_to_ear
        # OR: we add extra delay to anti-noise = acoustic_delay_ref_to_ear - secondary_path_delay

        extra_delay_needed = self.acoustic_delay_ref_to_ear - self.secondary_path_delay
        extra_delay_samples = int(extra_delay_needed / self.space.dt)

        total_controller_delay = self.delay_samples + max(0, extra_delay_samples)

        print(f"Using delay compensation: {total_controller_delay} samples")

        for step in range(num_steps):
            noise_func(self.space, self.space.time)

            ref_value = self.space.get_pressure_at(self.ref_mic_pos)
            self.reference_mic.record(self.space)

            # Store reference
            ref_buffer.append(-ref_value)

            # Output with proper delay
            if len(ref_buffer) > total_controller_delay:
                anti_noise = ref_buffer.pop(0)
                self.speaker.emit(self.space, anti_noise)

            self.error_mic.record(self.space)
            self.space.step()

        return self.error_mic.get_signal()


def main():
    """
    Demonstrate the problem of latency in ANC.
    """
    print("=" * 60)
    print("Step 5: ANC with Latency - The Problem")
    print("=" * 60)
    print()

    # Create acoustic space
    space = AcousticSpace1D(
        length=0.5,
        num_points=200,
        speed_of_sound=343.0,
        boundary='absorbing'
    )

    print()

    # Create system WITH latency
    anc = ANCSystemWithLatency(
        space=space,
        noise_position=0.05,
        reference_mic_position=0.10,
        speaker_position=0.35,
        error_mic_position=0.45,
        processing_delay_ms=0.5,  # Half millisecond processing
        speaker_delay_ms=0.2       # Speaker response time
    )

    print()

    # Noise source
    frequency = 100  # Hz
    print(f"Noise: {frequency} Hz sine wave")

    def noise_source(space, time):
        value = np.sin(2 * np.pi * frequency * time)
        space.inject_pressure(anc.noise_pos, value)

    duration = 0.05  # 50 ms

    # =========================================
    # Test 1: No ANC (baseline)
    # =========================================
    print()
    print("Test 1: No ANC (baseline)")
    t, error_no_anc = anc.run_no_anc(noise_source, duration)
    power_no_anc = np.mean(error_no_anc[len(error_no_anc)//2:]**2)
    print(f"  Power at ear: {power_no_anc:.4f}")

    # =========================================
    # Test 2: Naive ANC (ignores latency)
    # =========================================
    print()
    print("Test 2: Naive ANC (ignores latency)")
    t, error_naive = anc.run_naive_anc(noise_source, duration)
    power_naive = np.mean(error_naive[len(error_naive)//2:]**2)

    if power_naive > 1e-10:
        reduction_naive = 10 * np.log10(power_no_anc / power_naive)
    else:
        reduction_naive = float('inf')

    print(f"  Power at ear: {power_naive:.4f}")
    print(f"  'Reduction': {reduction_naive:.1f} dB")
    if reduction_naive < 0:
        print("  WARNING: Negative reduction means we made it WORSE!")

    # =========================================
    # Test 3: Delay-compensated ANC
    # =========================================
    print()
    print("Test 3: Delay-compensated ANC (ideal)")
    t, error_compensated = anc.run_delay_compensated_anc(noise_source, duration)
    power_compensated = np.mean(error_compensated[len(error_compensated)//2:]**2)

    if power_compensated > 1e-10:
        reduction_compensated = 10 * np.log10(power_no_anc / power_compensated)
    else:
        reduction_compensated = float('inf')

    print(f"  Power at ear: {power_compensated:.4f}")
    print(f"  Reduction: {reduction_compensated:.1f} dB")

    # =========================================
    # Plot results
    # =========================================
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Reference
    t_ref, ref_signal = anc.reference_mic.get_signal()
    axes[0].plot(t_ref * 1000, ref_signal, 'b-', linewidth=1.5)
    axes[0].set_ylabel('Reference')
    axes[0].set_title('Reference Microphone Signal')
    axes[0].grid(True, alpha=0.3)

    # No ANC
    axes[1].plot(t * 1000, error_no_anc, 'r-', linewidth=1.5)
    axes[1].set_ylabel('No ANC')
    axes[1].set_title('Error Signal: No ANC (Baseline)')
    axes[1].grid(True, alpha=0.3)

    # Naive ANC
    axes[2].plot(t * 1000, error_naive, 'orange', linewidth=1.5)
    axes[2].set_ylabel('Naive ANC')
    title_naive = f'Error Signal: Naive ANC ({reduction_naive:.1f} dB)'
    if reduction_naive < 0:
        title_naive += ' - WORSE!'
    axes[2].set_title(title_naive)
    axes[2].grid(True, alpha=0.3)

    # Compensated ANC
    axes[3].plot(t * 1000, error_compensated, 'g-', linewidth=1.5)
    axes[3].set_ylabel('Compensated')
    axes[3].set_xlabel('Time (ms)')
    axes[3].set_title(f'Error Signal: Delay-Compensated ANC ({reduction_compensated:.1f} dB)')
    axes[3].grid(True, alpha=0.3)

    # Match y-scales
    max_amp = max(
        np.max(np.abs(error_no_anc)),
        np.max(np.abs(error_naive)),
        np.max(np.abs(error_compensated))
    ) * 1.2
    for ax in axes[1:]:
        ax.set_ylim(-max_amp, max_amp)

    plt.tight_layout()
    plt.savefig('output/plots/step5_latency_problem.png', dpi=150)
    print()
    print("Saved: output/plots/step5_latency_problem.png")

    # =========================================
    # Comparison overlay
    # =========================================
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(t * 1000, error_no_anc, 'r-', linewidth=2, alpha=0.7, label='No ANC')
    ax.plot(t * 1000, error_naive, 'orange', linewidth=2, alpha=0.7,
           label=f'Naive ANC ({reduction_naive:.1f} dB)')
    ax.plot(t * 1000, error_compensated, 'g-', linewidth=2, alpha=0.7,
           label=f'Compensated ANC ({reduction_compensated:.1f} dB)')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Pressure at Ear')
    ax.set_title('Effect of Latency on ANC Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/plots/step5_latency_comparison.png', dpi=150)
    print("Saved: output/plots/step5_latency_comparison.png")

    # =========================================
    # Diagram explaining the problem
    # =========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    t_demo = np.linspace(0, 4*np.pi, 200)

    # Left: Naive (wrong timing)
    ax = axes[0]
    noise = np.sin(t_demo)
    anti_naive = -np.sin(t_demo - np.pi/2)  # Delayed by 90 degrees
    total_naive = noise + anti_naive

    ax.plot(t_demo, noise, 'b-', linewidth=2, label='Noise')
    ax.plot(t_demo, anti_naive, 'r--', linewidth=2, label='Anti-noise (LATE)')
    ax.plot(t_demo, total_naive, 'purple', linewidth=2, label='Sum (NOT cancelled!)')
    ax.set_title('Naive ANC: Anti-noise arrives LATE\n→ Wrong phase → No cancellation!')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Phase')
    ax.set_ylabel('Amplitude')

    # Right: Compensated (correct timing)
    ax = axes[1]
    anti_correct = -np.sin(t_demo)  # Correct timing
    total_correct = noise + anti_correct

    ax.plot(t_demo, noise, 'b-', linewidth=2, label='Noise')
    ax.plot(t_demo, anti_correct, 'g--', linewidth=2, label='Anti-noise (on time)')
    ax.plot(t_demo, total_correct, 'lightgreen', linewidth=3, label='Sum = ~0')
    ax.fill_between(t_demo, -0.1, 0.1, alpha=0.3, color='green')
    ax.set_title('Delay-Compensated ANC: Correct timing\n→ Opposite phase → Cancellation!')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Phase')
    ax.set_ylabel('Amplitude')

    plt.tight_layout()
    plt.savefig('output/plots/step5_timing_diagram.png', dpi=150)
    print("Saved: output/plots/step5_timing_diagram.png")

    # =========================================
    # Key points
    # =========================================
    print()
    print("=" * 60)
    print("THE LATENCY PROBLEM - KEY POINTS")
    print("=" * 60)
    print()
    print("1. SECONDARY PATH:")
    print("   The path from controller output to error microphone includes:")
    print("   - Processing delay (ADC, DSP, DAC)")
    print("   - Speaker response time")
    print("   - Acoustic propagation (speaker → ear)")
    print()
    print("2. WHY NAIVE ANC FAILS:")
    print("   If we just invert the reference signal, the anti-noise")
    print("   arrives LATE at the ear. Late anti-noise has WRONG PHASE")
    print("   and doesn't cancel - it may even make noise WORSE!")
    print()
    print("3. THE SOLUTION:")
    print("   We must COMPENSATE for the secondary path delay.")
    print("   But in real systems:")
    print("   - The secondary path is complex (not just a delay)")
    print("   - It includes frequency-dependent speaker response")
    print("   - It may change over time (temperature, position, etc.)")
    print()
    print("4. WHY WE NEED FxLMS:")
    print("   The FxLMS algorithm ADAPTIVELY learns to compensate")
    print("   for the secondary path. It continuously adjusts the")
    print("   filter weights to minimize the error signal.")
    print()
    print("Next step: Implement FxLMS and see it work!")
    print()

    plt.show()

    return anc


if __name__ == '__main__':
    main()
