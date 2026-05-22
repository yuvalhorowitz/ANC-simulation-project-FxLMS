"""
MIMO FxNLMS Adaptive Filter — Stage 1 (1 ref × M speakers × 1 error mic)

True MIMO ANC: M independent adaptive filters, each driving its own speaker
with its own learned anti-noise signal. Each speaker has its own secondary
path estimate, so each filter learns the optimal weights for its own path.

Differs from pseudo-MIMO (the scalar FxNLMS broadcast across speakers) in that:
- Each speaker emits a unique signal (not a broadcast copy)
- Each filter is updated using its own filtered reference (filtered through
  its own secondary path estimate)
- The single scalar error from the error mic feeds back into all M filter
  updates simultaneously

This is a stepping stone toward Stage 2 (multi-error-mic) which is where the
spatial robustness benefits become provable. See docs/mimo_plan.md for the
full design.
"""

import numpy as np
from typing import List


class MIMOFxNLMS:
    """
    MIMO Filtered-x Normalized LMS — 1 reference, M speakers, 1 error mic.

    Maintains M independent FIR filter weight vectors, one per speaker. Each
    speaker has its own secondary path estimate s_hat_m, so each filter
    learns weights tuned to its own acoustic path.
    """

    def __init__(
        self,
        filter_length: int,
        step_size: float,
        secondary_path_estimates: List[np.ndarray],
        regularization: float = 1e-4,
        leakage: float = 0.0,
    ):
        """
        Args:
            filter_length: Number of FIR taps per speaker (L)
            step_size: Normalized step size (mu)
            secondary_path_estimates: List of M FIR coefficient arrays, one
                per speaker (s_hat_m for each m).
            regularization: Small constant in NLMS denominator (delta)
            leakage: Optional weight decay factor (gamma). 0 = no leakage.
        """
        self.M = len(secondary_path_estimates)
        self.L = filter_length
        self.mu = step_size
        self.delta = regularization
        self.leakage = leakage

        # Per-speaker secondary path estimates. Different speakers may have
        # different path lengths in principle; we store them as-is.
        self.s_hat = [np.array(s) for s in secondary_path_estimates]
        self.s_hat_lens = [len(s) for s in self.s_hat]

        # Per-speaker weight vectors, shape (M, L)
        self.W = np.zeros((self.M, self.L))

        # Shared reference buffer (all filters see the same x)
        self.x_buffer = np.zeros(self.L)

        # Per-speaker filtered-reference buffers, shape (M, L)
        # xf_m[k] = filtered reference at lag k for speaker m
        self.xf_buffers = np.zeros((self.M, self.L))

        # Per-speaker history buffer used to compute the filtered reference
        # by convolution with s_hat_m. We need a buffer at least as long as
        # the longest secondary path estimate.
        max_s_len = max(self.s_hat_lens)
        self.s_history = np.zeros(max_s_len)

        self.mse_history: List[float] = []

    def generate_antinoise(self, x: float) -> np.ndarray:
        """
        Generate one anti-noise sample per speaker.

        Args:
            x: Current reference sample x(n)

        Returns:
            np.ndarray of shape (M,) — anti-noise sample for each speaker.
            Each speaker m emits y_m(n) = w_m^T · x_buffer.
        """
        # Update shared reference buffer
        self.x_buffer = np.roll(self.x_buffer, 1)
        self.x_buffer[0] = x

        # Compute y_m = w_m · x_buffer for each speaker
        # W shape (M, L) · x_buffer shape (L,) → result shape (M,)
        y_per_speaker = self.W @ self.x_buffer
        return y_per_speaker

    def filter_reference(self, x: float) -> None:
        """
        Update per-speaker filtered-reference buffers xf_m.

        For each speaker m, computes:
            xf_m(n) = sum_k s_hat_m[k] * x(n - k)
        and shifts xf_m_buffer to maintain history of xf_m values.

        Args:
            x: Current reference sample x(n)
        """
        # Update shared history buffer for the convolution
        self.s_history = np.roll(self.s_history, 1)
        self.s_history[0] = x

        # For each speaker, compute filtered-reference value and update its buffer
        for m in range(self.M):
            s_len = self.s_hat_lens[m]
            xf_m_n = np.dot(self.s_hat[m], self.s_history[:s_len])

            self.xf_buffers[m] = np.roll(self.xf_buffers[m], 1)
            self.xf_buffers[m, 0] = xf_m_n

    def update_weights(self, e: float) -> None:
        """
        Update all M weight vectors using the single scalar error.

        For each speaker m:
            norm_m = delta + xf_m_buffer · xf_m_buffer
            w_m(n+1) = (1 - mu*gamma) * w_m(n) - (mu*e/norm_m) * xf_m_buffer

        Args:
            e: Single scalar error sample at the error mic
        """
        for m in range(self.M):
            xf_m = self.xf_buffers[m]
            norm_m = self.delta + np.dot(xf_m, xf_m)
            self.W[m] = (1 - self.mu * self.leakage) * self.W[m] \
                        - (self.mu * e / norm_m) * xf_m

        self.mse_history.append(e ** 2)

    def reset(self) -> None:
        """Reset all internal state to initial conditions."""
        self.W = np.zeros((self.M, self.L))
        self.x_buffer = np.zeros(self.L)
        self.xf_buffers = np.zeros((self.M, self.L))
        self.s_history = np.zeros(len(self.s_history))
        self.mse_history = []

    @property
    def weights(self) -> np.ndarray:
        """Return concatenated weights across all M speakers (for inspection)."""
        return self.W.flatten()
