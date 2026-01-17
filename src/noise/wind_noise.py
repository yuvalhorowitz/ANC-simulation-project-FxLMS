"""
Wind Noise Generator

Generates low-frequency aerodynamic noise.

Wind noise characteristics:
- Low-frequency dominated
- Turbulent, random
- Increases with speed (approximately speed^1.5 for amplitude)
"""

import numpy as np
from scipy import signal as scipy_signal


class WindNoiseGenerator:
    """
    Generates aerodynamic wind noise.
    """

    def __init__(self, sample_rate: float = 16000):
        """
        Initialize wind noise generator.

        Args:
            sample_rate: Sampling rate in Hz
        """
        self.fs = sample_rate

    def generate(
        self,
        duration: float,
        speed_kmh: float = 100,
        amplitude: float = 1.0
    ) -> np.ndarray:
        """
        Generate wind noise signal.

        Args:
            duration: Duration in seconds
            speed_kmh: Vehicle speed
            amplitude: Overall amplitude

        Returns:
            Wind noise signal
        """
        n_samples = int(duration * self.fs)
        nyq = self.fs / 2

        # Base noise
        noise = np.random.randn(n_samples)

        # Low-pass filter (wind noise is low-frequency dominated)
        cutoff = min(150 / nyq, 0.99)
        if cutoff > 0:
            b, a = scipy_signal.butter(3, cutoff, btype='low')
            noise = scipy_signal.filtfilt(b, a, noise)

        # Speed-dependent amplitude
        speed_factor = (speed_kmh / 100) ** 1.5

        # Normalize and scale
        if np.max(np.abs(noise)) > 0:
            noise = amplitude * speed_factor * noise / np.max(np.abs(noise))

        return noise
