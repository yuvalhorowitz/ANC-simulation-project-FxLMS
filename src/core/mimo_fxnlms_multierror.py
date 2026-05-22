"""
MIMO FxNLMS — Multi-Error-Mic Variant (Stage 2: 1 ref × M speakers × K error mics)

Extends Stage 1 by adding K error microphones distributed in space. Cost function
becomes J = sum_k e_k(n)^2. Each speaker m maintains K filtered-reference buffers
(one per error mic), and is updated using the sum over K of (e_k * xf_{m,k}).

The benefit: the optimization spreads cancellation across the K-point region
defined by the error mics, creating a wider zone of quiet rather than a sharp
peak at a single point.

This class is independent of the Stage 1 MIMOFxNLMS class; both remain available.
"""

import numpy as np
from typing import List


class MIMOFxNLMSMultiError:
    """
    1 reference, M speakers, K error mics.

    Filter weights: W of shape (M, L)
    Per-speaker secondary path estimates: s_hat[m][k] for each (speaker m, error mic k)
    Filtered-reference buffers: xf_buffers[m][k] of length L
    """

    def __init__(
        self,
        filter_length: int,
        step_size: float,
        secondary_path_estimates: List[List[np.ndarray]],
        regularization: float = 1e-4,
        leakage: float = 0.0,
    ):
        """
        Args:
            filter_length: Number of FIR taps per speaker (L)
            step_size: Normalized step size (mu)
            secondary_path_estimates: List of M lists, each containing K FIR
                coefficient arrays. secondary_path_estimates[m][k] is s_hat
                from speaker m to error mic k.
            regularization: Small constant in NLMS denominator (delta)
            leakage: Weight decay factor (gamma); 0 = no leakage
        """
        self.M = len(secondary_path_estimates)
        if self.M == 0:
            raise ValueError("Need at least one speaker")
        self.K = len(secondary_path_estimates[0])
        if self.K == 0:
            raise ValueError("Need at least one error mic")

        self.L = filter_length
        self.mu = step_size
        self.delta = regularization
        self.leakage = leakage

        # Per-(speaker, error_mic) secondary path estimate
        # s_hat[m][k] is np.array of length L_path (may differ per pair)
        self.s_hat = [[np.array(s) for s in row] for row in secondary_path_estimates]

        # Lengths for indexing during convolution
        self.s_hat_lens = [[len(s) for s in row] for row in self.s_hat]
        max_path_len = max(max(row) for row in self.s_hat_lens)

        # Filter weights: M independent vectors of length L
        self.W = np.zeros((self.M, self.L))

        # Shared reference buffer (all filters see same x)
        self.x_buffer = np.zeros(self.L)

        # Per-(speaker, error_mic) filtered-reference buffer of length L
        # xf_buffers[m][k] holds xf_{m,k}(n), xf_{m,k}(n-1), ..., xf_{m,k}(n-L+1)
        self.xf_buffers = np.zeros((self.M, self.K, self.L))

        # Shared raw-reference history for computing each xf_{m,k}
        self.s_history = np.zeros(max_path_len)

        self.mse_history: List[float] = []

    def generate_antinoise(self, x: float) -> np.ndarray:
        """
        Generate one anti-noise sample per speaker. Returns array (M,).

        Each speaker m emits y_m(n) = w_m^T · x_buffer.
        """
        self.x_buffer = np.roll(self.x_buffer, 1)
        self.x_buffer[0] = x
        return self.W @ self.x_buffer

    def filter_reference(self, x: float) -> None:
        """
        Update all M·K filtered-reference buffers.

        For each (speaker m, error mic k):
            xf_{m,k}(n) = sum over taps of s_hat[m][k] * x(n - tap)
        """
        self.s_history = np.roll(self.s_history, 1)
        self.s_history[0] = x

        for m in range(self.M):
            for k in range(self.K):
                L_path = self.s_hat_lens[m][k]
                xf_mk_n = np.dot(self.s_hat[m][k], self.s_history[:L_path])
                self.xf_buffers[m, k] = np.roll(self.xf_buffers[m, k], 1)
                self.xf_buffers[m, k, 0] = xf_mk_n

    def update_weights(self, errors: np.ndarray) -> None:
        """
        Update each filter w_m using all K error signals.

        w_m(n+1) = (1 - mu*gamma) * w_m(n)
                   - (mu / norm_m) * sum_k (e_k * xf_{m,k}_buffer)

        norm_m = delta + sum_k (xf_{m,k}_buffer · xf_{m,k}_buffer)

        Args:
            errors: array of shape (K,) — one error per error mic
        """
        for m in range(self.M):
            # Weighted sum of filtered-reference buffers across error mics
            grad = np.zeros(self.L)
            norm_m = self.delta
            for k in range(self.K):
                xf_mk = self.xf_buffers[m, k]
                grad += errors[k] * xf_mk
                norm_m += np.dot(xf_mk, xf_mk)

            self.W[m] = (1 - self.mu * self.leakage) * self.W[m] \
                        - (self.mu / norm_m) * grad

        self.mse_history.append(float(np.sum(errors ** 2)))

    def reset(self) -> None:
        self.W = np.zeros((self.M, self.L))
        self.x_buffer = np.zeros(self.L)
        self.xf_buffers = np.zeros((self.M, self.K, self.L))
        self.s_history = np.zeros(len(self.s_history))
        self.mse_history = []

    @property
    def weights(self) -> np.ndarray:
        return self.W.flatten()
