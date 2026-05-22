"""
Full MIMO FxNLMS — Stage 3 (N reference mics × M speakers × K error mics)

True MIMO with all three dimensions multi-channel:
  - N independent reference signals (NOT averaged)
  - M independent speaker outputs
  - K error mics defining the cancellation region

Filter weights become a 3D matrix W of shape (M, N, L). Each speaker m
maintains N separate filters, one per reference signal. The contribution
from speaker m is:
    y_m(n) = sum_n  w_{m,n}^T · x_n_buffer

This is the canonical "MIMO FxLMS" structure described in vehicle ANC
literature.

Note on secondary paths: each speaker m has K secondary path estimates
(one per error mic), shape s_hat[m][k]. They do NOT depend on which
reference mic is used — sound only cares about its own source-mic pair.

Filtered references xf_{m,n,k} are 4D-indexed: filtered ref-n through the
secondary path from speaker m to error mic k.
"""

import numpy as np
from typing import List, Sequence


class MIMOFxNLMSFull:
    """
    Full MIMO: N reference inputs, M speaker outputs, K error mics.
    """

    def __init__(
        self,
        filter_length: int,
        step_size: float,
        num_reference_mics: int,
        secondary_path_estimates: List[List[np.ndarray]],
        regularization: float = 1e-4,
        leakage: float = 0.0,
    ):
        """
        Args:
            filter_length: L taps per (speaker, ref_mic) filter
            step_size: mu
            num_reference_mics: N
            secondary_path_estimates: list of M lists, each containing K
                FIR coefficient arrays. s_hat[m][k] = speaker m → error mic k.
                Same structure as Stage 2; secondary paths are independent
                of the reference mic.
            regularization: delta in NLMS denominator
            leakage: gamma weight decay (0 = none)
        """
        self.M = len(secondary_path_estimates)
        if self.M == 0:
            raise ValueError("Need at least one speaker")
        self.K = len(secondary_path_estimates[0])
        if self.K == 0:
            raise ValueError("Need at least one error mic")
        self.N = num_reference_mics
        if self.N == 0:
            raise ValueError("Need at least one reference mic")

        self.L = filter_length
        self.mu = step_size
        self.delta = regularization
        self.leakage = leakage

        self.s_hat = [[np.array(s) for s in row] for row in secondary_path_estimates]
        self.s_hat_lens = [[len(s) for s in row] for row in self.s_hat]
        max_path = max(max(row) for row in self.s_hat_lens)

        # Filter weights: W[m, n] is a vector of length L for the (speaker m, ref n) tap path
        # Shape (M, N, L)
        self.W = np.zeros((self.M, self.N, self.L))

        # Per-reference-mic raw buffer: x_buffers[n] of length L
        self.x_buffers = np.zeros((self.N, self.L))

        # Per-(speaker, ref_mic, error_mic) filtered-reference buffer of length L
        # xf_buffers[m, n, k] of length L
        self.xf_buffers = np.zeros((self.M, self.N, self.K, self.L))

        # Per-reference history for computing filtered references
        self.s_history = np.zeros((self.N, max_path))

        self.mse_history: List[float] = []

    def generate_antinoise(self, x: Sequence[float]) -> np.ndarray:
        """
        Generate one anti-noise sample per speaker given N reference samples.

        Args:
            x: array-like of N reference samples for time n

        Returns:
            anti-noise array of shape (M,)
        """
        x = np.asarray(x, dtype=float)
        if x.shape != (self.N,):
            raise ValueError(f"x must have shape ({self.N},), got {x.shape}")

        # Update per-reference buffers
        for n in range(self.N):
            self.x_buffers[n] = np.roll(self.x_buffers[n], 1)
            self.x_buffers[n, 0] = x[n]

        # Compute y_m = sum_n W[m, n] · x_buffers[n]
        # Vectorized: sum over n of W[:, n] · x_buffers[n]
        y_per_speaker = np.zeros(self.M)
        for m in range(self.M):
            for n in range(self.N):
                y_per_speaker[m] += np.dot(self.W[m, n], self.x_buffers[n])
        return y_per_speaker

    def filter_reference(self, x: Sequence[float]) -> None:
        """
        Update all M·N·K filtered-reference buffers.

        For each (speaker m, ref n, error mic k):
            xf_{m,n,k}(t) = sum_tap s_hat[m][k][tap] * x_n(t - tap)
        """
        x = np.asarray(x, dtype=float)
        # Update per-reference history
        for n in range(self.N):
            self.s_history[n] = np.roll(self.s_history[n], 1)
            self.s_history[n, 0] = x[n]

        for m in range(self.M):
            for k in range(self.K):
                L_path = self.s_hat_lens[m][k]
                s_h = self.s_hat[m][k]
                for n in range(self.N):
                    xf_val = np.dot(s_h, self.s_history[n, :L_path])
                    self.xf_buffers[m, n, k] = np.roll(self.xf_buffers[m, n, k], 1)
                    self.xf_buffers[m, n, k, 0] = xf_val

    def update_weights(self, errors: np.ndarray) -> None:
        """
        Update each filter w_{m,n} using all K error signals.

        For each (m, n):
            grad_{m,n} = sum_k (e_k * xf_{m,n,k}_buffer)
            norm_{m,n} = delta + sum_k (xf_{m,n,k}_buffer · xf_{m,n,k}_buffer)
            w_{m,n}(t+1) = (1 - mu*gamma) w_{m,n}(t) - (mu / norm_{m,n}) * grad_{m,n}

        Args:
            errors: array of shape (K,)
        """
        for m in range(self.M):
            for n in range(self.N):
                grad = np.zeros(self.L)
                norm_mn = self.delta
                for k in range(self.K):
                    xf = self.xf_buffers[m, n, k]
                    grad += errors[k] * xf
                    norm_mn += np.dot(xf, xf)
                self.W[m, n] = (1 - self.mu * self.leakage) * self.W[m, n] \
                               - (self.mu / norm_mn) * grad

        self.mse_history.append(float(np.sum(errors ** 2)))

    def reset(self) -> None:
        self.W = np.zeros((self.M, self.N, self.L))
        self.x_buffers = np.zeros((self.N, self.L))
        self.xf_buffers = np.zeros((self.M, self.N, self.K, self.L))
        self.s_history = np.zeros_like(self.s_history)
        self.mse_history = []

    @property
    def weights(self) -> np.ndarray:
        return self.W.flatten()
