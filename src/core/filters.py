"""
Filter Utilities

FIR and IIR filter implementations for signal processing.
"""

import numpy as np
from typing import Tuple


class FIRFilter:
    """
    General-purpose FIR filter with sample-by-sample processing.
    """

    def __init__(self, coefficients: np.ndarray):
        """
        Initialize FIR filter.

        Args:
            coefficients: Filter tap weights (impulse response)
        """
        self.coefficients = np.array(coefficients)
        self.order = len(coefficients)
        self.buffer = np.zeros(self.order)

    def filter_sample(self, x: float) -> float:
        """
        Filter a single sample.

        Args:
            x: Input sample

        Returns:
            Filtered output sample
        """
        # Shift buffer and insert new sample
        self.buffer = np.roll(self.buffer, 1)
        self.buffer[0] = x

        # Compute output
        return np.dot(self.coefficients, self.buffer)

    def filter_signal(self, x: np.ndarray) -> np.ndarray:
        """
        Filter entire signal array.

        Args:
            x: Input signal

        Returns:
            Filtered signal
        """
        return np.convolve(x, self.coefficients, mode='same')

    def reset(self) -> None:
        """Clear filter state."""
        self.buffer = np.zeros(self.order)

    def get_frequency_response(
        self,
        fs: float,
        num_points: int = 512
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute frequency response.

        Args:
            fs: Sample rate
            num_points: Number of frequency points

        Returns:
            Tuple of (frequencies, magnitude in dB)
        """
        from scipy import signal
        w, h = signal.freqz(self.coefficients, worN=num_points)
        frequencies = w * fs / (2 * np.pi)
        magnitude_db = 20 * np.log10(np.abs(h) + 1e-10)
        return frequencies, magnitude_db


class BandpassPreFilter:
    """
    Sample-by-sample IIR bandpass filter using second-order sections.
    SOS form is used for numerical stability at low normalized frequencies.
    """

    def __init__(self, low_freq: float, high_freq: float, sample_rate: float, order: int = 4):
        from scipy.signal import butter
        nyq = sample_rate / 2.0
        low = max(low_freq / nyq, 0.001)
        high = min(high_freq / nyq, 0.99)
        self.sos = butter(order, [low, high], btype='band', output='sos')
        self.state = np.zeros((self.sos.shape[0], 2))

    def filter_sample(self, x: float) -> float:
        for i in range(self.sos.shape[0]):
            b0, b1, b2, _, a1, a2 = self.sos[i]
            w = x - a1 * self.state[i, 0] - a2 * self.state[i, 1]
            y = b0 * w + b1 * self.state[i, 0] + b2 * self.state[i, 1]
            self.state[i, 1] = self.state[i, 0]
            self.state[i, 0] = w
            x = y
        return y

    def reset(self):
        self.state = np.zeros((self.sos.shape[0], 2))
