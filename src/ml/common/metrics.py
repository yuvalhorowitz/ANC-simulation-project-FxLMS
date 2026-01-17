"""
Metrics for evaluating ANC performance.

These metrics are used to compare ML-enhanced ANC against the baseline FxNLMS.
"""

import numpy as np
from typing import Optional, List


def noise_reduction_db(
    desired: np.ndarray,
    error: np.ndarray,
    steady_state_start: Optional[int] = None
) -> float:
    """
    Calculate noise reduction in dB.

    The noise reduction is computed as the ratio of desired signal power
    (noise at error mic before cancellation) to error signal power
    (residual noise after cancellation).

    Args:
        desired: Noise signal at error mic (what we want to cancel)
        error: Residual signal after ANC
        steady_state_start: Sample index where system has converged.
                           If None, uses second half of signals.

    Returns:
        Noise reduction in dB (higher = better cancellation)

    Example:
        >>> d = np.sin(2 * np.pi * 100 * np.arange(16000) / 16000)
        >>> e = 0.1 * d  # 90% reduction
        >>> nr = noise_reduction_db(d, e)
        >>> print(f"Noise reduction: {nr:.1f} dB")  # ~20 dB
    """
    if steady_state_start is None:
        steady_state_start = len(desired) // 2

    # Ensure we have valid range
    steady_state_start = max(0, min(steady_state_start, len(desired) - 100))

    # Compute power in steady state
    d_power = np.mean(desired[steady_state_start:] ** 2)
    e_power = np.mean(error[steady_state_start:] ** 2)

    # Avoid log of zero
    if e_power < 1e-10:
        return 60.0  # Maximum reasonable NR

    return 10 * np.log10(d_power / (e_power + 1e-10))


def convergence_time(
    mse_history: List[float],
    threshold_ratio: float = 0.1,
    window: int = 100
) -> int:
    """
    Find when MSE drops below threshold, indicating convergence.

    Convergence is defined as the point where MSE drops below a fraction
    of the initial MSE and stays there.

    Args:
        mse_history: List of MSE values per sample or per window
        threshold_ratio: Converged when MSE < initial_MSE * threshold_ratio
        window: Number of initial samples to average for initial MSE

    Returns:
        Number of samples/steps to convergence (lower = faster convergence)
        Returns len(mse_history) if never converged.

    Example:
        >>> mse = [1.0, 0.5, 0.2, 0.1, 0.08, 0.05, 0.05]
        >>> t = convergence_time(mse, threshold_ratio=0.1)
        >>> print(f"Converged at step {t}")  # ~3 or 4
    """
    if len(mse_history) < window:
        return len(mse_history)

    # Calculate initial MSE as average of first 'window' samples
    initial_mse = np.mean(mse_history[:window])
    threshold = initial_mse * threshold_ratio

    # Find first index where MSE drops below threshold
    for i, mse in enumerate(mse_history):
        if mse < threshold:
            return i

    return len(mse_history)  # Never converged


def stability_score(
    mse_history: List[float],
    divergence_threshold: float = 10.0,
    window: int = 100
) -> float:
    """
    Check if the filter remained stable during adaptation.

    Stability is measured by checking for MSE explosions that indicate
    filter divergence.

    Args:
        mse_history: List of MSE values
        divergence_threshold: Consider diverged if MSE > initial * threshold
        window: Number of initial samples for baseline MSE

    Returns:
        1.0 if stable throughout, 0.0 if diverged at any point

    Example:
        >>> stable_mse = [1.0, 0.5, 0.2, 0.1]  # decreasing
        >>> unstable_mse = [1.0, 0.5, 5.0, 50.0]  # exploding
        >>> print(stability_score(stable_mse))  # 1.0
        >>> print(stability_score(unstable_mse))  # 0.0
    """
    if len(mse_history) < window:
        return 1.0  # Not enough data to judge

    # Calculate initial MSE
    initial_mse = np.mean(mse_history[:window])
    max_allowed = initial_mse * divergence_threshold

    # Check for MSE explosion
    for mse in mse_history[window:]:
        if mse > max_allowed:
            return 0.0

    return 1.0


def compute_all_metrics(
    desired: np.ndarray,
    error: np.ndarray,
    mse_history: Optional[List[float]] = None
) -> dict:
    """
    Compute all metrics at once.

    Args:
        desired: Noise signal at error mic
        error: Residual signal after ANC
        mse_history: Optional MSE history for convergence/stability metrics

    Returns:
        Dictionary with all computed metrics
    """
    metrics = {
        'noise_reduction_db': noise_reduction_db(desired, error),
    }

    if mse_history is not None:
        metrics['convergence_time'] = convergence_time(mse_history)
        metrics['stability_score'] = stability_score(mse_history)
        metrics['final_mse'] = mse_history[-1] if mse_history else float('nan')

    return metrics
