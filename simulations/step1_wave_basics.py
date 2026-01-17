"""
Step 1: Basic Wave Propagation Simulation

Goal: Understand how acoustic waves propagate in a simple 1D space (like a tube).

This simulation demonstrates:
1. Wave propagation at the speed of sound
2. Reflections at boundaries
3. The superposition principle

Physics:
- Sound is a pressure wave traveling through air
- Speed of sound: c ≈ 343 m/s at 20°C
- 1D wave equation: ∂²p/∂t² = c² * ∂²p/∂x²

We use a simple finite difference method to solve the wave equation numerically.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.audio import save_wav, pressure_to_audio


class AcousticSpace1D:
    """
    Simple 1D acoustic space simulation.

    Models a tube or narrow space where sound waves propagate in one dimension.
    Uses the finite difference method to solve the wave equation.
    """

    def __init__(
        self,
        length: float = 2.0,           # Length of space in meters
        num_points: int = 200,          # Spatial resolution
        speed_of_sound: float = 343.0,  # m/s at 20°C
        boundary: str = 'reflecting'    # 'reflecting' or 'absorbing'
    ):
        """
        Initialize the 1D acoustic space.

        Args:
            length: Physical length of the space in meters
            num_points: Number of spatial grid points
            speed_of_sound: Speed of sound in m/s
            boundary: Boundary condition type
        """
        self.L = length
        self.N = num_points
        self.c = speed_of_sound
        self.boundary = boundary

        # Spatial grid
        self.dx = length / (num_points - 1)
        self.x = np.linspace(0, length, num_points)

        # Time step (CFL condition for stability: dt <= dx/c)
        self.dt = 0.5 * self.dx / speed_of_sound  # Safety factor of 0.5

        # Pressure field: current, previous, next
        self.p_current = np.zeros(num_points)
        self.p_previous = np.zeros(num_points)
        self.p_next = np.zeros(num_points)

        # Courant number (should be <= 1 for stability)
        self.courant = (speed_of_sound * self.dt / self.dx) ** 2

        # Time tracking
        self.time = 0.0
        self.step_count = 0

        # History for analysis
        self.pressure_history = []

        print(f"1D Acoustic Space initialized:")
        print(f"  Length: {length} m")
        print(f"  Grid points: {num_points}")
        print(f"  dx: {self.dx*1000:.2f} mm")
        print(f"  dt: {self.dt*1e6:.2f} µs")
        print(f"  Courant number: {np.sqrt(self.courant):.3f}")

    def inject_pressure(self, position: float, amplitude: float):
        """
        Inject pressure at a specific position.

        Args:
            position: Position in meters from left end
            amplitude: Pressure amplitude (arbitrary units)
        """
        # Find nearest grid point
        idx = int(position / self.dx)
        idx = max(0, min(idx, self.N - 1))
        self.p_current[idx] += amplitude

    def inject_sine_wave(
        self,
        position: float,
        frequency: float,
        amplitude: float = 1.0
    ):
        """
        Inject a sinusoidal pressure wave at a position.

        Args:
            position: Position in meters
            frequency: Frequency in Hz
            amplitude: Wave amplitude
        """
        idx = int(position / self.dx)
        idx = max(0, min(idx, self.N - 1))

        # Add sine wave contribution at current time
        self.p_current[idx] += amplitude * np.sin(2 * np.pi * frequency * self.time)

    def step(self):
        """
        Advance simulation by one time step.

        Uses the finite difference method:
        p(x, t+dt) = 2*p(x,t) - p(x,t-dt) + c²*dt²/dx² * [p(x+dx,t) - 2*p(x,t) + p(x-dx,t)]
        """
        # Interior points (finite difference)
        for i in range(1, self.N - 1):
            self.p_next[i] = (
                2 * self.p_current[i]
                - self.p_previous[i]
                + self.courant * (
                    self.p_current[i+1]
                    - 2 * self.p_current[i]
                    + self.p_current[i-1]
                )
            )

        # Boundary conditions
        if self.boundary == 'reflecting':
            # Hard wall: pressure doubles at boundary (fixed end)
            # Implemented as: derivative = 0 at boundary
            self.p_next[0] = self.p_next[1]
            self.p_next[-1] = self.p_next[-2]
        else:  # absorbing
            # Simple absorbing boundary (not perfect, but reduces reflections)
            self.p_next[0] = self.p_current[1]
            self.p_next[-1] = self.p_current[-2]

        # Shift time levels
        self.p_previous = self.p_current.copy()
        self.p_current = self.p_next.copy()

        # Update time
        self.time += self.dt
        self.step_count += 1

    def run(self, duration: float, source_func=None, store_every: int = 10,
            record_position: float = None):
        """
        Run simulation for a duration.

        Args:
            duration: Simulation duration in seconds
            source_func: Optional function(space, time) to inject waves
            store_every: Store pressure field every N steps
            record_position: Optional position (meters) to record audio

        Returns:
            pressure_history array, and optionally (history, recording) if record_position set
        """
        num_steps = int(duration / self.dt)

        self.pressure_history = []

        # Audio recording at a point
        recording = [] if record_position is not None else None
        record_idx = int(record_position / self.dx) if record_position else 0

        for step in range(num_steps):
            # Advance simulation FIRST
            self.step()

            # Apply source AFTER the step (so it doesn't get overwritten)
            if source_func is not None:
                source_func(self, self.time)

            # Store history
            if step % store_every == 0:
                self.pressure_history.append(self.p_current.copy())

            # Record audio at listener position
            if recording is not None:
                recording.append(self.p_current[record_idx])

        history = np.array(self.pressure_history)

        if recording is not None:
            return history, np.array(recording)
        return history

    def get_pressure_at(self, position: float) -> float:
        """
        Get current pressure at a specific position.

        Args:
            position: Position in meters

        Returns:
            Pressure value at that position
        """
        idx = int(position / self.dx)
        idx = max(0, min(idx, self.N - 1))
        return self.p_current[idx]

    def reset(self):
        """Reset simulation to initial state."""
        self.p_current = np.zeros(self.N)
        self.p_previous = np.zeros(self.N)
        self.p_next = np.zeros(self.N)
        self.time = 0.0
        self.step_count = 0
        self.pressure_history = []


def create_animation(space: AcousticSpace1D, pressure_history: np.ndarray, interval: int = 50):
    """
    Create an animation of wave propagation.

    Args:
        space: The acoustic space object
        pressure_history: Array of pressure fields over time
        interval: Milliseconds between frames

    Returns:
        Matplotlib animation object
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    # Find amplitude range
    max_amp = np.max(np.abs(pressure_history)) * 1.2

    line, = ax.plot(space.x, pressure_history[0], 'b-', linewidth=2)
    ax.set_xlim(0, space.L)
    ax.set_ylim(-max_amp, max_amp)
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Pressure (arbitrary units)')
    ax.set_title('1D Acoustic Wave Propagation')
    ax.grid(True, alpha=0.3)

    # Add boundary markers
    ax.axvline(x=0, color='gray', linestyle='--', label='Boundary')
    ax.axvline(x=space.L, color='gray', linestyle='--')

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def update(frame):
        line.set_ydata(pressure_history[frame])
        time_text.set_text(f'Time: {frame * space.dt * 10:.4f} s')
        return line, time_text

    ani = FuncAnimation(fig, update, frames=len(pressure_history),
                       interval=interval, blit=True)
    return ani, fig


def main():
    """
    Main demonstration of 1D wave propagation.
    """
    print("=" * 60)
    print("Step 1: Basic Wave Propagation Simulation")
    print("=" * 60)
    print()

    # Create a 2-meter tube
    space = AcousticSpace1D(
        length=2.0,
        num_points=400,
        speed_of_sound=343.0,
        boundary='reflecting'
    )

    print()

    # =========================================
    # Experiment 1: Single pulse propagation
    # =========================================
    print("Experiment 1: Single Pulse Propagation")
    print("-" * 40)

    # Inject a Gaussian pulse at the left side
    pulse_position = 0.2  # meters from left
    pulse_width = 0.05    # meters

    x_centered = space.x - pulse_position
    space.p_current = np.exp(-x_centered**2 / (2 * pulse_width**2))
    space.p_previous = space.p_current.copy()

    print(f"Injected Gaussian pulse at x = {pulse_position} m")
    print(f"Pulse width: {pulse_width*100} cm")

    # Calculate expected travel time to the right boundary and back
    travel_distance = 2 * (space.L - pulse_position)
    expected_time = travel_distance / space.c
    print(f"Expected round-trip time: {expected_time*1000:.2f} ms")

    # Run simulation
    duration = 0.015  # 15 ms - enough for a round trip
    history = space.run(duration, store_every=5)

    print(f"Simulation complete: {len(history)} frames")

    # Plot snapshots
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    frames_to_show = [0, len(history)//5, 2*len(history)//5,
                      3*len(history)//5, 4*len(history)//5, len(history)-1]

    for ax, frame_idx in zip(axes, frames_to_show):
        ax.plot(space.x, history[frame_idx], 'b-', linewidth=1.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=space.L, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlim(0, space.L)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel('Position (m)')
        ax.set_ylabel('Pressure')
        time_ms = frame_idx * space.dt * 5 * 1000
        ax.set_title(f't = {time_ms:.2f} ms')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Wave Propagation: Pulse Traveling and Reflecting', fontsize=14)
    plt.tight_layout()
    plt.savefig('output/plots/step1_pulse_propagation.png', dpi=150)
    print("Saved: output/plots/step1_pulse_propagation.png")

    # =========================================
    # Experiment 2: Continuous sine wave
    # =========================================
    print()
    print("Experiment 2: Continuous Sine Wave (100 Hz)")
    print("-" * 40)

    space.reset()

    # Source function: inject sine wave at left boundary
    frequency = 100  # Hz
    wavelength = space.c / frequency
    print(f"Frequency: {frequency} Hz")
    print(f"Wavelength: {wavelength:.2f} m")

    def sine_source(space, time):
        amplitude = 1.0
        space.inject_sine_wave(position=0.01, frequency=frequency, amplitude=amplitude)

    # Run for several periods
    duration = 0.03  # 30 ms = 3 periods at 100 Hz
    history = space.run(duration, source_func=sine_source, store_every=5)

    # Plot snapshots showing standing wave pattern developing
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    frames_to_show = [0, len(history)//5, 2*len(history)//5,
                      3*len(history)//5, 4*len(history)//5, len(history)-1]

    for ax, frame_idx in zip(axes, frames_to_show):
        ax.plot(space.x, history[frame_idx], 'b-', linewidth=1.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=space.L, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlim(0, space.L)
        ax.set_ylim(-3, 3)
        ax.set_xlabel('Position (m)')
        ax.set_ylabel('Pressure')
        time_ms = frame_idx * space.dt * 5 * 1000
        ax.set_title(f't = {time_ms:.2f} ms')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Continuous {frequency} Hz Wave with Reflections', fontsize=14)
    plt.tight_layout()
    plt.savefig('output/plots/step1_sine_wave.png', dpi=150)
    print("Saved: output/plots/step1_sine_wave.png")

    # =========================================
    # Experiment 3: Record audio at listener position
    # =========================================
    print()
    print("Experiment 3: Recording Audio at Listener Position")
    print("-" * 40)

    space.reset()

    # Listener position (1 meter from source)
    listener_position = 1.0  # meters
    print(f"Source at: 0.01 m")
    print(f"Listener at: {listener_position} m")

    # Run longer for audible audio (1 second)
    duration_audio = 1.0
    print(f"Recording duration: {duration_audio} seconds")

    history_audio, recording = space.run(
        duration_audio,
        source_func=sine_source,
        store_every=100,
        record_position=listener_position
    )

    # Convert to audio sample rate
    print(f"Simulation sample rate: {1/space.dt:.0f} Hz")
    print(f"Recording length: {len(recording)} samples")

    # Resample to 16 kHz for audio
    audio_signal = pressure_to_audio(recording, space.dt, target_sample_rate=16000)
    print(f"Audio length after resampling: {len(audio_signal)} samples")

    # Save audio file
    os.makedirs('output/audio', exist_ok=True)
    audio_path = save_wav(
        'output/audio/step1_100hz_wave.wav',
        audio_signal,
        sample_rate=16000
    )
    print(f"Saved audio: {audio_path}")

    # Also create a multi-frequency example
    print()
    print("Creating multi-frequency audio (50 + 100 + 150 Hz)...")
    space.reset()

    def multi_source(space, time):
        space.p_current[2] += 0.5 * np.sin(2 * np.pi * 50 * time)
        space.p_current[2] += 0.3 * np.sin(2 * np.pi * 100 * time)
        space.p_current[2] += 0.2 * np.sin(2 * np.pi * 150 * time)

    _, recording_multi = space.run(
        duration_audio,
        source_func=multi_source,
        store_every=100,
        record_position=listener_position
    )

    audio_multi = pressure_to_audio(recording_multi, space.dt, target_sample_rate=16000)
    audio_path_multi = save_wav(
        'output/audio/step1_multi_freq_wave.wav',
        audio_multi,
        sample_rate=16000
    )
    print(f"Saved audio: {audio_path_multi}")

    # =========================================
    # Key observations
    # =========================================
    print()
    print("=" * 60)
    print("KEY OBSERVATIONS")
    print("=" * 60)
    print()
    print("1. WAVE PROPAGATION:")
    print(f"   - Speed of sound: {space.c} m/s")
    print(f"   - A wave at {frequency} Hz has wavelength {wavelength:.2f} m")
    print()
    print("2. REFLECTIONS:")
    print("   - Waves reflect at hard boundaries (walls)")
    print("   - Reflected wave + incoming wave = interference pattern")
    print()
    print("3. SUPERPOSITION PRINCIPLE:")
    print("   - Multiple waves add together (linearly)")
    print("   - This is the KEY to noise cancellation!")
    print("   - If we add wave2 = -wave1, they cancel: wave1 + wave2 = 0")
    print()
    print("4. AUDIO OUTPUT:")
    print("   - Listen to output/audio/step1_100hz_wave.wav")
    print("   - Listen to output/audio/step1_multi_freq_wave.wav")
    print("   - These are recordings at the 'listener' position in the tube")
    print()

    plt.show()

    return space, history


if __name__ == '__main__':
    main()
