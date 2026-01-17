"""
Acoustic Path Generator

Extracts Room Impulse Responses (RIRs) from pyroomacoustics simulations
for use in ANC systems.
"""

import numpy as np
import pyroomacoustics as pra
from typing import Tuple, Optional
from scipy import signal as scipy_signal


class AcousticPathGenerator:
    """
    Extracts and manages acoustic path impulse responses from pyroomacoustics.

    This class provides the interface between pyroomacoustics room simulations
    and the FxLMS algorithm, extracting RIRs that can be used as:
    - Primary path P(z): noise source to error microphone
    - Secondary path S(z): speaker to error microphone
    - Reference path: noise source to reference microphone
    """

    def __init__(self, room: pra.ShoeBox):
        """
        Initialize path generator with a pyroomacoustics room.

        Args:
            room: pyroomacoustics ShoeBox room with sources and microphones
        """
        self.room = room
        self.fs = room.fs
        self._rir_computed = False

    def compute_rir(self) -> None:
        """
        Compute Room Impulse Responses for all source-microphone pairs.

        Must be called before extracting paths.
        """
        self.room.compute_rir()
        self._rir_computed = True

    def get_rir(self, mic_idx: int, source_idx: int) -> np.ndarray:
        """
        Get raw RIR between a source and microphone.

        Args:
            mic_idx: Index of the microphone
            source_idx: Index of the source

        Returns:
            RIR as numpy array
        """
        if not self._rir_computed:
            self.compute_rir()
        return np.array(self.room.rir[mic_idx][source_idx])

    def get_primary_path(
        self,
        noise_source_idx: int = 0,
        error_mic_idx: int = 1
    ) -> np.ndarray:
        """
        Get primary path RIR: noise source to error microphone.

        This is P(z) in ANC terminology - the acoustic path that
        the noise travels through to reach the listener.

        Args:
            noise_source_idx: Index of noise source (default 0)
            error_mic_idx: Index of error microphone (default 1)

        Returns:
            Primary path impulse response
        """
        return self.get_rir(error_mic_idx, noise_source_idx)

    def get_secondary_path(
        self,
        speaker_idx: int = 1,
        error_mic_idx: int = 1
    ) -> np.ndarray:
        """
        Get secondary path RIR: control speaker to error microphone.

        This is S(z) in ANC terminology - the acoustic path from
        the control speaker to the listener's ear.

        Args:
            speaker_idx: Index of control speaker (default 1)
            error_mic_idx: Index of error microphone (default 1)

        Returns:
            Secondary path impulse response
        """
        return self.get_rir(error_mic_idx, speaker_idx)

    def get_reference_path(
        self,
        noise_source_idx: int = 0,
        ref_mic_idx: int = 0
    ) -> np.ndarray:
        """
        Get reference path RIR: noise source to reference microphone.

        This is the path used to measure the reference signal x(n)
        in feedforward ANC.

        Args:
            noise_source_idx: Index of noise source (default 0)
            ref_mic_idx: Index of reference microphone (default 0)

        Returns:
            Reference path impulse response
        """
        return self.get_rir(ref_mic_idx, noise_source_idx)

    def get_secondary_path_estimate(
        self,
        speaker_idx: int = 1,
        error_mic_idx: int = 1,
        modeling_error: float = 0.05
    ) -> np.ndarray:
        """
        Get secondary path estimate with optional modeling error.

        In real systems, we only have an estimate of the secondary path.
        This method adds realistic modeling error to the true path.

        Args:
            speaker_idx: Index of control speaker
            error_mic_idx: Index of error microphone
            modeling_error: Relative error to add (0.05 = 5% error)

        Returns:
            Estimated secondary path impulse response
        """
        true_path = self.get_secondary_path(speaker_idx, error_mic_idx)

        if modeling_error > 0:
            # Add Gaussian noise proportional to signal energy
            noise = modeling_error * np.std(true_path) * np.random.randn(len(true_path))
            estimated_path = true_path + noise

            # Add small gain error
            gain_error = 1 + modeling_error * np.random.randn()
            estimated_path *= gain_error
        else:
            estimated_path = true_path.copy()

        return estimated_path

    def get_all_anc_paths(
        self,
        noise_source_idx: int = 0,
        speaker_idx: int = 1,
        ref_mic_idx: int = 0,
        error_mic_idx: int = 1,
        modeling_error: float = 0.05
    ) -> dict:
        """
        Get all acoustic paths needed for ANC simulation.

        Args:
            noise_source_idx: Index of noise source
            speaker_idx: Index of control speaker
            ref_mic_idx: Index of reference microphone
            error_mic_idx: Index of error microphone
            modeling_error: Error to add to secondary path estimate

        Returns:
            Dict with 'primary', 'secondary', 'secondary_estimate', 'reference' paths
        """
        if not self._rir_computed:
            self.compute_rir()

        return {
            'primary': self.get_primary_path(noise_source_idx, error_mic_idx),
            'secondary': self.get_secondary_path(speaker_idx, error_mic_idx),
            'secondary_estimate': self.get_secondary_path_estimate(
                speaker_idx, error_mic_idx, modeling_error
            ),
            'reference': self.get_reference_path(noise_source_idx, ref_mic_idx)
        }

    def get_path_delay(self, path: np.ndarray, threshold: float = 0.1) -> int:
        """
        Estimate the delay (in samples) of an impulse response.

        Finds the first sample where the absolute value exceeds
        threshold * max(|path|).

        Args:
            path: Impulse response array
            threshold: Detection threshold relative to peak

        Returns:
            Delay in samples
        """
        peak = np.max(np.abs(path))
        indices = np.where(np.abs(path) > threshold * peak)[0]
        return indices[0] if len(indices) > 0 else 0

    def truncate_path(
        self,
        path: np.ndarray,
        max_length: int = None,
        energy_threshold: float = 0.99
    ) -> np.ndarray:
        """
        Truncate path to reduce computational cost.

        Keeps the portion of the RIR containing most of the energy.

        Args:
            path: Full impulse response
            max_length: Maximum length (if None, uses energy threshold)
            energy_threshold: Fraction of energy to retain (0-1)

        Returns:
            Truncated impulse response
        """
        if max_length is not None:
            return path[:max_length]

        # Find length containing threshold% of energy
        cumulative_energy = np.cumsum(path ** 2)
        total_energy = cumulative_energy[-1]

        if total_energy == 0:
            return path

        idx = np.searchsorted(cumulative_energy, energy_threshold * total_energy)
        return path[:idx + 1]


class FIRPath:
    """
    Wrapper for using RIRs as FIR filters with sample-by-sample processing.

    Compatible with the FxLMS algorithm's expected interface.
    """

    def __init__(self, impulse_response: np.ndarray):
        """
        Initialize FIR path filter.

        Args:
            impulse_response: Filter coefficients (RIR)
        """
        self.coefficients = np.array(impulse_response)
        self.order = len(impulse_response)
        self.buffer = np.zeros(self.order)

    def filter_sample(self, x: float) -> float:
        """
        Filter a single sample through the path.

        Args:
            x: Input sample

        Returns:
            Filtered output sample
        """
        # Shift buffer and insert new sample
        self.buffer = np.roll(self.buffer, 1)
        self.buffer[0] = x

        # Convolve
        return np.dot(self.coefficients, self.buffer)

    def filter_signal(self, x: np.ndarray) -> np.ndarray:
        """
        Filter entire signal array.

        Args:
            x: Input signal

        Returns:
            Filtered signal
        """
        return np.convolve(x, self.coefficients, mode='full')[:len(x)]

    def reset(self) -> None:
        """Clear filter state."""
        self.buffer = np.zeros(self.order)

    def get_impulse_response(self) -> np.ndarray:
        """Get the impulse response (filter coefficients)."""
        return self.coefficients.copy()
