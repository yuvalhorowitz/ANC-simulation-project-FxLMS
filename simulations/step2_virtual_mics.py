"""
Step 2: Virtual Microphones

Goal: Understand what microphones "measure" at different positions in the acoustic space.

This simulation demonstrates:
1. How microphones record pressure variations over time
2. Time delay between microphones based on wave travel time
3. Phase relationships between measurement points

Key concept for ANC:
- A "reference microphone" picks up noise BEFORE it reaches the listener
- This gives the system TIME to generate anti-noise
- The delay between microphones = distance / speed_of_sound
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulations.step1_wave_basics import AcousticSpace1D


class VirtualMicrophone:
    """
    A virtual microphone that records pressure at a fixed position.

    In a real ANC system, microphones convert acoustic pressure to
    electrical signals. Here we simply sample the pressure field
    at the microphone's position.
    """

    def __init__(self, position: float, name: str = "Mic"):
        """
        Initialize virtual microphone.

        Args:
            position: Position in meters from left end
            name: Identifier for this microphone
        """
        self.position = position
        self.name = name
        self.recording = []
        self.time_stamps = []

    def record(self, space: AcousticSpace1D):
        """
        Record current pressure from the acoustic space.

        Args:
            space: The acoustic space to measure
        """
        pressure = space.get_pressure_at(self.position)
        self.recording.append(pressure)
        self.time_stamps.append(space.time)

    def get_signal(self) -> tuple:
        """
        Get the recorded signal.

        Returns:
            Tuple of (time_array, pressure_array)
        """
        return np.array(self.time_stamps), np.array(self.recording)

    def reset(self):
        """Clear recorded data."""
        self.recording = []
        self.time_stamps = []


class MicrophoneArray:
    """
    An array of virtual microphones for multi-point measurement.
    """

    def __init__(self, positions: list, names: list = None):
        """
        Create array of microphones.

        Args:
            positions: List of positions in meters
            names: Optional list of names for each mic
        """
        if names is None:
            names = [f"Mic {i+1}" for i in range(len(positions))]

        self.microphones = [
            VirtualMicrophone(pos, name)
            for pos, name in zip(positions, names)
        ]

    def record_all(self, space: AcousticSpace1D):
        """Record from all microphones."""
        for mic in self.microphones:
            mic.record(space)

    def reset_all(self):
        """Reset all microphones."""
        for mic in self.microphones:
            mic.reset()


def calculate_delay(distance: float, speed_of_sound: float = 343.0) -> float:
    """
    Calculate time delay for sound to travel a distance.

    Args:
        distance: Distance in meters
        speed_of_sound: Speed of sound in m/s

    Returns:
        Time delay in seconds
    """
    return distance / speed_of_sound


def main():
    """
    Demonstrate virtual microphones and time delay measurement.
    """
    print("=" * 60)
    print("Step 2: Virtual Microphones")
    print("=" * 60)
    print()

    # Create acoustic space
    space = AcousticSpace1D(
        length=2.0,
        num_points=400,
        speed_of_sound=343.0,
        boundary='absorbing'  # Use absorbing to avoid reflections confusing things
    )

    print()

    # =========================================
    # Experiment 1: Measure delay between microphones
    # =========================================
    print("Experiment 1: Measuring Time Delay Between Microphones")
    print("-" * 50)

    # Place three microphones at different positions
    mic_positions = [0.3, 0.8, 1.5]  # meters
    mic_names = ["Reference Mic", "Middle Mic", "Error Mic (listener)"]

    mic_array = MicrophoneArray(mic_positions, mic_names)

    print("Microphone positions:")
    for mic in mic_array.microphones:
        print(f"  {mic.name}: x = {mic.position} m")

    # Calculate expected delays relative to first mic
    print()
    print("Expected delays (relative to Reference Mic):")
    for i, mic in enumerate(mic_array.microphones[1:], 1):
        distance = mic.position - mic_positions[0]
        delay = calculate_delay(distance, space.c)
        print(f"  {mic.name}: {delay*1000:.3f} ms ({distance:.2f} m)")

    # Source: Single pulse at x=0
    print()
    print("Generating pulse at x = 0.1 m...")

    # Inject pulse
    pulse_position = 0.1
    pulse_width = 0.02
    x_centered = space.x - pulse_position
    space.p_current = np.exp(-x_centered**2 / (2 * pulse_width**2))
    space.p_previous = space.p_current.copy()

    # Run simulation and record
    duration = 0.01  # 10 ms
    num_steps = int(duration / space.dt)

    for step in range(num_steps):
        mic_array.record_all(space)
        space.step()

    # Plot microphone recordings
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    colors = ['blue', 'green', 'red']

    for ax, mic, color in zip(axes, mic_array.microphones, colors):
        t, signal = mic.get_signal()
        ax.plot(t * 1000, signal, color=color, linewidth=1.5)
        ax.set_ylabel(f'{mic.name}\n(x={mic.position}m)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, duration * 1000)

        # Mark the peak
        peak_idx = np.argmax(signal)
        peak_time = t[peak_idx] * 1000
        ax.axvline(x=peak_time, color=color, linestyle='--', alpha=0.5)
        ax.text(peak_time + 0.1, np.max(signal) * 0.8,
                f'Peak: {peak_time:.2f} ms', fontsize=9)

    axes[-1].set_xlabel('Time (ms)')
    axes[0].set_title('Pulse Detection at Different Microphone Positions')

    plt.tight_layout()
    plt.savefig('output/plots/step2_mic_delay.png', dpi=150)
    print("Saved: output/plots/step2_mic_delay.png")

    # =========================================
    # Experiment 2: Continuous wave - observe phase
    # =========================================
    print()
    print("Experiment 2: Continuous Wave - Phase Relationships")
    print("-" * 50)

    space.reset()
    mic_array.reset_all()

    # Use a frequency where wavelength is comparable to mic spacing
    frequency = 200  # Hz
    wavelength = space.c / frequency
    print(f"Frequency: {frequency} Hz")
    print(f"Wavelength: {wavelength:.2f} m")

    # Calculate expected phase differences
    print()
    print("Expected phase differences (relative to Reference Mic):")
    for mic in mic_array.microphones[1:]:
        distance = mic.position - mic_positions[0]
        phase_delay = (distance / wavelength) * 360  # degrees
        phase_delay = phase_delay % 360
        print(f"  {mic.name}: {phase_delay:.1f} degrees")

    # Source function
    def sine_source(space, time):
        space.inject_sine_wave(position=0.01, frequency=frequency, amplitude=1.0)

    # Run simulation
    duration = 0.02  # 20 ms = 4 periods at 200 Hz
    num_steps = int(duration / space.dt)

    for step in range(num_steps):
        sine_source(space, space.time)
        mic_array.record_all(space)
        space.step()

    # Plot overlaid signals
    fig, ax = plt.subplots(figsize=(12, 6))

    for mic, color in zip(mic_array.microphones, colors):
        t, signal = mic.get_signal()
        ax.plot(t * 1000, signal, color=color, linewidth=1.5,
               label=f'{mic.name} (x={mic.position}m)')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Pressure')
    ax.set_title(f'Continuous {frequency} Hz Wave at Different Positions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(10, 20)  # Show steady-state portion

    plt.tight_layout()
    plt.savefig('output/plots/step2_phase_relationship.png', dpi=150)
    print("Saved: output/plots/step2_phase_relationship.png")

    # =========================================
    # Key observations for ANC
    # =========================================
    print()
    print("=" * 60)
    print("KEY OBSERVATIONS FOR ANC")
    print("=" * 60)
    print()
    print("1. TIME DELAY:")
    print("   - Sound takes TIME to travel between points")
    print(f"   - Delay = distance / speed = d / {space.c} m/s")
    print()
    print("2. REFERENCE MICROPHONE CONCEPT:")
    print("   - Place a 'reference mic' CLOSER to the noise source")
    print("   - It detects the noise BEFORE it reaches the listener")
    print("   - This gives time to compute and play anti-noise")
    print()
    print("3. PHASE RELATIONSHIP:")
    print("   - At different positions, the wave has different phase")
    print("   - For cancellation, anti-noise must have OPPOSITE phase")
    print("   - Phase = 180 degrees apart = destructive interference")
    print()
    print("4. WHY THIS MATTERS:")
    print("   In our ANC system:")
    print("   - Reference Mic: at x=0.3m (detects noise early)")
    print("   - Speaker: between reference mic and listener")
    print("   - Error Mic: at x=1.5m (at listener's ear)")
    print()

    # =========================================
    # Visualize the ANC concept
    # =========================================
    fig, ax = plt.subplots(figsize=(12, 4))

    # Draw the tube
    ax.fill_between([0, space.L], -0.3, 0.3, alpha=0.1, color='gray')
    ax.axhline(y=0.3, color='gray', linewidth=2)
    ax.axhline(y=-0.3, color='gray', linewidth=2)

    # Mark positions
    ax.plot(0.1, 0, 'ko', markersize=15, label='Noise Source')
    ax.annotate('Noise\nSource', (0.1, -0.5), ha='center', fontsize=10)

    ax.plot(0.3, 0, 'b^', markersize=15, label='Reference Mic')
    ax.annotate('Reference\nMic', (0.3, -0.5), ha='center', fontsize=10, color='blue')

    ax.plot(1.0, 0, 'gs', markersize=15, label='Speaker')
    ax.annotate('ANC\nSpeaker', (1.0, -0.5), ha='center', fontsize=10, color='green')

    ax.plot(1.5, 0, 'r^', markersize=15, label='Error Mic (ear)')
    ax.annotate('Error Mic\n(Listener)', (1.5, -0.5), ha='center', fontsize=10, color='red')

    # Draw wave direction
    ax.annotate('', xy=(0.8, 0.15), xytext=(0.2, 0.15),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(0.5, 0.25, 'Noise Wave', ha='center', fontsize=10)

    ax.set_xlim(-0.1, space.L + 0.1)
    ax.set_ylim(-0.8, 0.6)
    ax.set_xlabel('Position (m)')
    ax.set_title('Feedforward ANC Setup: Reference Mic Detects Noise Before Listener')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('output/plots/step2_anc_concept.png', dpi=150)
    print()
    print("Saved: output/plots/step2_anc_concept.png")

    plt.show()

    return mic_array


if __name__ == '__main__':
    main()
