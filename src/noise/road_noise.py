"""
Road Noise Generator

Generates broadband road/tire noise.

Road noise characteristics:
- Broadband (not tonal like engine)
- Tire cavity resonance around 200-250 Hz
- Increases with speed
- Extended frequency range: 20-1000 Hz for better ML training
"""

import numpy as np
from scipy import signal as scipy_signal


class RoadNoiseGenerator:
    """
    Generates road/tire noise (broadband).
    """

    def __init__(self, sample_rate: float = 16000):
        """
        Initialize road noise generator.

        Args:
            sample_rate: Sampling rate in Hz
        """
        self.fs = sample_rate

    def generate(
        self,
        duration: float,
        speed_kmh: float = 80,
        amplitude: float = 1.0
    ) -> np.ndarray:
        """
        Generate road noise signal.

        Args:
            duration: Duration in seconds
            speed_kmh: Vehicle speed in km/h
            amplitude: Overall amplitude

        Returns:
            Road noise signal
        """
        n_samples = int(duration * self.fs)
        nyq = self.fs / 2

        # Base broadband noise
        noise = np.random.randn(n_samples)

        # Bandpass filter to 20-1000 Hz range (extended for ML training)
        low = 20 / nyq
        high = min(1000 / nyq, 0.99)

        if high > low:
            b, a = scipy_signal.butter(4, [low, high], btype='band')
            noise = scipy_signal.filtfilt(b, a, noise)

        # Speed-dependent amplitude
        speed_factor = (speed_kmh / 100) ** 1.5

        # Normalize and scale
        if np.max(np.abs(noise)) > 0:
            noise = amplitude * speed_factor * noise / np.max(np.abs(noise))

        return noise
