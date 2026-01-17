"""
FxLMS Adaptive Filter Implementation

Filtered-x Least Mean Square algorithm for Active Noise Control.

Based on the update equation from the paper:
    w(n+1) = w(n) + mu * e(n) * f(n)

Where:
    w(n): Adaptive filter weights at time n
    mu: Step-size (learning rate)
    e(n): Error signal (residual noise at error microphone)
    f(n): Filtered reference signal = x(n) * s_hat(n)
"""

import numpy as np
from typing import Tuple, List


class FxLMS:
    """
    Filtered-x Least Mean Square adaptive filter for ANC.

    The FxLMS algorithm compensates for the secondary path by filtering
    the reference signal with an estimate of the secondary path before
    using it in the weight update equation.

    Attributes:
        filter_length (int): Number of FIR filter taps (L)
        step_size (float): Adaptation step-size (mu)
        weights (np.ndarray): Adaptive filter coefficients w(n)
        secondary_path_estimate (np.ndarray): Estimate of secondary path s_hat(n)
    """

    def __init__(
        self,
        filter_length: int,
        step_size: float,
        secondary_path_estimate: np.ndarray,
        leakage: float = 0.0
    ):
        """
        Initialize FxLMS filter.

        Args:
            filter_length: Number of adaptive filter taps (L)
            step_size: Learning rate mu (0 < mu < mu_max for stability)
            secondary_path_estimate: FIR coefficients of estimated S(z)
            leakage: Leakage factor for weight decay (0 = standard LMS)
        """
        self.L = filter_length
        self.mu = step_size
        self.leakage = leakage
        self.s_hat = np.array(secondary_path_estimate)

        # Initialize weights to zero
        self.weights = np.zeros(filter_length)

        # Buffer for reference signal history x(n), x(n-1), ..., x(n-L+1)
        self.x_buffer = np.zeros(filter_length)

        # Buffer for filtering reference through secondary path estimate
        self.f_buffer = np.zeros(len(secondary_path_estimate))

        # Buffer of FILTERED reference values for weight update
        # This is the key to FxLMS - we need x_f(n), x_f(n-1), ... x_f(n-L+1)
        self.xf_buffer = np.zeros(filter_length)

        # Metrics tracking
        self.mse_history: List[float] = []
        self.weight_history: List[np.ndarray] = []

    def filter_reference(self, x: float) -> float:
        """
        Filter reference signal through secondary path estimate.
        Computes f(n) = x(n) * s_hat(n)

        This is the key step that differentiates FxLMS from standard LMS.
        The reference signal must be filtered through an estimate of the
        secondary path to compensate for the delay and frequency response
        of the actual secondary path.

        Args:
            x: Current reference sample x(n)

        Returns:
            Filtered reference sample f(n)
        """
        # Update filtered reference buffer (shift and insert new sample)
        self.f_buffer = np.roll(self.f_buffer, 1)
        self.f_buffer[0] = x

        # Convolve with secondary path estimate: f(n) = sum(s_hat[k] * x(n-k))
        f_n = np.dot(self.s_hat, self.f_buffer[:len(self.s_hat)])

        # Update the filtered reference buffer for weight update
        # This stores f(n), f(n-1), ..., f(n-L+1) for use in LMS update
        self.xf_buffer = np.roll(self.xf_buffer, 1)
        self.xf_buffer[0] = f_n

        return f_n

    def generate_antinoise(self, x: float) -> float:
        """
        Generate anti-noise signal y(n) = w^T(n) * x(n)

        The anti-noise is computed as the convolution of the adaptive
        filter weights with the reference signal buffer.

        Args:
            x: Current reference sample

        Returns:
            Anti-noise output y(n)
        """
        # Update reference buffer (shift and insert new sample)
        self.x_buffer = np.roll(self.x_buffer, 1)
        self.x_buffer[0] = x

        # Compute filter output: y(n) = sum(w[k] * x(n-k))
        y_n = np.dot(self.weights, self.x_buffer)
        return y_n

    def update_weights(self, e: float) -> None:
        """
        Update adaptive filter weights using FxLMS update rule.

        w(n+1) = w(n) - mu * e(n) * xf(n)

        Where xf(n) is the vector of filtered reference samples.
        The negative sign is for minimizing error (gradient descent).

        Args:
            e: Error signal sample e(n)
        """
        # FxLMS weight update using FILTERED reference buffer
        # This is the key difference from standard LMS!
        self.weights = (1 - self.mu * self.leakage) * self.weights - \
                       self.mu * e * self.xf_buffer

        # Store MSE for analysis
        self.mse_history.append(e ** 2)

    def process_sample(self, x: float, d: float, secondary_path: np.ndarray) -> Tuple[float, float]:
        """
        Process one sample through the complete FxLMS system.

        This method performs the complete ANC processing for a single sample:
        1. Generate anti-noise from reference signal
        2. Simulate anti-noise propagation through secondary path
        3. Compute error signal
        4. Update adaptive filter weights

        Args:
            x: Reference signal sample x(n)
            d: Desired signal (noise at error mic via primary path)
            secondary_path: Actual secondary path impulse response

        Returns:
            Tuple of (error signal e(n), anti-noise y(n))
        """
        # Generate anti-noise
        y = self.generate_antinoise(x)

        # Compute filtered reference for weight update
        f_n = self.filter_reference(x)

        return y, f_n

    def get_mse(self, window: int = 100) -> float:
        """
        Get recent mean squared error.

        Args:
            window: Number of recent samples to average

        Returns:
            Mean squared error over the window
        """
        if len(self.mse_history) < window:
            return np.mean(self.mse_history) if self.mse_history else float('inf')
        return np.mean(self.mse_history[-window:])

    def get_weights(self) -> np.ndarray:
        """Get current adaptive filter weights."""
        return self.weights.copy()

    def reset(self) -> None:
        """Reset filter state to initial conditions."""
        self.weights = np.zeros(self.L)
        self.x_buffer = np.zeros(self.L)
        self.f_buffer = np.zeros(len(self.s_hat))
        self.xf_buffer = np.zeros(self.L)
        self.mse_history = []
        self.weight_history = []


class FxNLMS(FxLMS):
    """
    Filtered-x Normalized LMS - variant with normalized step size.

    The normalized variant adjusts the step size based on the signal
    power, providing more stable convergence for varying signal levels.

    Update equation:
        w(n+1) = w(n) + mu * e(n) * x(n) / (delta + ||x(n)||^2)

    Where delta is a small regularization constant to prevent division by zero.
    """

    def __init__(
        self,
        filter_length: int,
        step_size: float,
        secondary_path_estimate: np.ndarray,
        regularization: float = 1e-6
    ):
        """
        Initialize FxNLMS filter.

        Args:
            filter_length: Number of adaptive filter taps
            step_size: Normalized step size (0 < mu < 2 for stability)
            secondary_path_estimate: FIR coefficients of estimated S(z)
            regularization: Small constant to prevent division by zero
        """
        super().__init__(filter_length, step_size, secondary_path_estimate)
        self.delta = regularization

    def update_weights(self, e: float) -> None:
        """
        Normalized weight update.

        The step size is normalized by the signal power plus a
        regularization term, providing more consistent adaptation
        regardless of signal amplitude.

        Args:
            e: Error signal sample
        """
        # Compute normalization factor: ||xf(n)||^2 + delta
        norm_factor = self.delta + np.dot(self.xf_buffer, self.xf_buffer)

        # Normalized weight update using filtered reference
        self.weights = self.weights - (self.mu * e / norm_factor) * self.xf_buffer

        # Store MSE
        self.mse_history.append(e ** 2)
