"""
ANC Controller

Orchestrates the complete ANC simulation.
"""

import numpy as np
from typing import Dict

from .fxlms import FxLMS, FxNLMS
from .filters import FIRFilter


class ANCController:
    """
    Complete ANC system controller for feedforward architecture.
    """

    def __init__(
        self,
        primary_path: np.ndarray,
        secondary_path: np.ndarray,
        secondary_path_estimate: np.ndarray,
        filter_length: int,
        step_size: float,
        algorithm: str = 'fxlms'
    ):
        """
        Initialize ANC controller.

        Args:
            primary_path: Impulse response of primary acoustic path
            secondary_path: Impulse response of secondary path
            secondary_path_estimate: Estimated secondary path for FxLMS
            filter_length: Adaptive filter length
            step_size: Learning rate
            algorithm: 'fxlms' or 'fxnlms'
        """
        self.primary_path = FIRFilter(primary_path)
        self.secondary_path = FIRFilter(secondary_path)

        # Initialize adaptive algorithm
        if algorithm.lower() == 'fxnlms':
            self.adaptive_filter = FxNLMS(
                filter_length, step_size, secondary_path_estimate
            )
        else:
            self.adaptive_filter = FxLMS(
                filter_length, step_size, secondary_path_estimate
            )

        # Buffer for secondary path filtering
        self.y_buffer = np.zeros(len(secondary_path))

        # Results storage
        self.results = {
            'reference': [],
            'desired': [],
            'antinoise': [],
            'error': [],
            'mse': []
        }

    def process_sample(self, x: float) -> Dict[str, float]:
        """
        Process one sample through complete ANC system.

        Args:
            x: Reference signal sample

        Returns:
            Dict with signal values
        """
        # Primary path
        d = self.primary_path.filter_sample(x)

        # Generate anti-noise
        y = self.adaptive_filter.generate_antinoise(x)

        # Secondary path
        y_prime = self.secondary_path.filter_sample(y)

        # Error signal
        e = d + y_prime

        # Update adaptive filter
        self.adaptive_filter.filter_reference(x)
        self.adaptive_filter.update_weights(e)

        # Store results
        self.results['reference'].append(x)
        self.results['desired'].append(d)
        self.results['antinoise'].append(y)
        self.results['error'].append(e)
        self.results['mse'].append(e**2)

        return {'reference': x, 'desired': d, 'antinoise': y, 'error': e}

    def run_simulation(
        self,
        reference_signal: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Run complete simulation.

        Args:
            reference_signal: Input noise signal
            verbose: Print progress

        Returns:
            Dict of output signals
        """
        n_samples = len(reference_signal)

        for i, x in enumerate(reference_signal):
            self.process_sample(x)

            if verbose and (i + 1) % (n_samples // 10) == 0:
                progress = (i + 1) / n_samples * 100
                current_mse = np.mean(self.results['mse'][-1000:])
                print(f"Progress: {progress:.0f}% | MSE: {current_mse:.6f}")

        return {k: np.array(v) for k, v in self.results.items()}

    def get_noise_reduction_db(self, window: int = 1000) -> float:
        """Calculate noise reduction in dB."""
        if len(self.results['desired']) < window:
            return 0.0

        d_power = np.mean(np.array(self.results['desired'][-window:])**2)
        e_power = np.mean(np.array(self.results['error'][-window:])**2)

        if e_power < 1e-10:
            return 60.0

        return 10 * np.log10(d_power / e_power)

    def reset(self) -> None:
        """Reset controller state."""
        self.primary_path.reset()
        self.secondary_path.reset()
        self.adaptive_filter.reset()
        self.results = {k: [] for k in self.results}
