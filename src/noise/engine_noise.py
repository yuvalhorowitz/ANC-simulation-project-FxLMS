"""
Engine Noise Generator

Generates realistic engine noise with RPM-dependent harmonics.

Engine noise characteristics:
- Fundamental frequency related to RPM: f = (RPM / 60) * (cylinders / 2) for 4-stroke
- Multiple harmonics (2x, 3x, 4x, etc.)
- For 20-300 Hz range: covers most RPM ranges
"""

import numpy as np
from typing import List


class EngineNoiseGenerator:
    """
    Generates engine noise with harmonics based on RPM.
    """

    def __init__(self, sample_rate: float = 16000, num_cylinders: int = 4):
        """
        Initialize engine noise generator.

        Args:
            sample_rate: Sampling rate in Hz
            num_cylinders: Number of engine cylinders
        """
        self.fs = sample_rate
        self.cylinders = num_cylinders

    def rpm_to_fundamental(self, rpm: float) -> float:
        """
        Convert RPM to fundamental firing frequency.

        For 4-stroke engine: f = (RPM / 60) * (cylinders / 2)

        Examples:
        - 4-cylinder at 2000 RPM: (2000/60) * 2 = 66.7 Hz
        - 4-cylinder at 3000 RPM: (3000/60) * 2 = 100 Hz
        """
        return (rpm / 60) * (self.cylinders / 2)

    def generate(
        self,
        duration: float,
        rpm: float = 2000,
        num_harmonics: int = 6,
        harmonic_weights: List[float] = None,
        amplitude: float = 1.0
    ) -> np.ndarray:
        """
        Generate engine noise signal.

        Args:
            duration: Signal duration in seconds
            rpm: Engine RPM
            num_harmonics: Number of harmonic components
            harmonic_weights: Relative amplitude of each harmonic
            amplitude: Overall amplitude

        Returns:
            Engine noise signal
        """
        n_samples = int(duration * self.fs)
        t = np.arange(n_samples) / self.fs

        f0 = self.rpm_to_fundamental(rpm)

        # Default harmonic weights (typical engine spectrum)
        if harmonic_weights is None:
            harmonic_weights = [1.0, 0.6, 0.35, 0.2, 0.12, 0.08]

        harmonic_weights = harmonic_weights[:num_harmonics]

        # Generate harmonics
        signal = np.zeros(n_samples)
        for k, weight in enumerate(harmonic_weights, 1):
            freq = f0 * k
            if freq <= 300:  # Stay within target range
                phase = np.random.rand() * 2 * np.pi  # Random initial phase
                signal += weight * np.sin(2 * np.pi * freq * t + phase)

        # Normalize and scale
        if np.max(np.abs(signal)) > 0:
            signal = amplitude * signal / np.max(np.abs(signal))

        return signal
