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


def convergence_time_90pct(
    mse_history: List[float],
    sample_rate: int = 16000,
    threshold_percent: float = 0.90,
    desired: Optional[np.ndarray] = None,
    error: Optional[np.ndarray] = None
) -> float:
    """
    Measure time to reach threshold_percent of final noise reduction.

    This is the industry-standard way to measure convergence:
    time until system achieves 90% of its final steady-state performance.

    Uses the correct noise reduction metric: NR(dB) = 10 * log10(desired_power / error_power)

    Args:
        mse_history: Array of MSE values over time (used for length if desired/error provided)
        sample_rate: Audio sample rate (Hz)
        threshold_percent: Fraction of final reduction (default 0.90)
        desired: Optional desired signal (noise at error mic before ANC)
        error: Optional error signal (residual after ANC)

    Returns:
        Convergence time in seconds

    Example:
        >>> t = convergence_time_90pct(mse, sample_rate=16000, desired=d, error=e)
        >>> print(f"Converged in {t:.2f} seconds")
    """
    # If desired and error are provided, use correct noise reduction calculation
    if desired is not None and error is not None:
        desired = np.array(desired)
        error = np.array(error)

        if len(desired) < 100:
            return len(desired) / sample_rate

        # Calculate noise reduction over time using sliding window
        window = 500  # Samples for averaging
        if len(desired) < window * 2:
            window = max(50, len(desired) // 10)

        step = window // 2  # 50% overlap

        reduction_db = []
        times = []

        for i in range(window, len(desired), step):
            d_power = np.mean(desired[i - window:i] ** 2)
            e_power = np.mean(error[i - window:i] ** 2)

            if e_power > 1e-10 and d_power > 1e-10:
                db = 10 * np.log10(d_power / e_power)
            else:
                db = 0

            # Clamp to reasonable range
            db = max(-5, min(30, db))
            reduction_db.append(db)
            times.append(i / sample_rate)

        if len(reduction_db) == 0:
            return len(desired) / sample_rate

        reduction_db = np.array(reduction_db)
        times = np.array(times)

        # Get final reduction (average of last 10% for stability)
        final_samples = max(1, len(reduction_db) // 10)
        final_reduction = np.mean(reduction_db[-final_samples:])

        # Handle case where no improvement
        if final_reduction <= 0:
            return len(desired) / sample_rate

        # Threshold is 90% of final reduction
        threshold = threshold_percent * final_reduction

        # Find first time exceeding threshold
        for t, db in zip(times, reduction_db):
            if db >= threshold:
                return t

        return len(desired) / sample_rate  # Never converged

    # Fallback to MSE-based calculation if desired/error not provided
    mse_array = np.array(mse_history)

    if len(mse_array) < 10:
        return len(mse_array) / sample_rate

    # Convert MSE to noise reduction (dB)
    # Use first samples as reference (before convergence)
    initial_mse = np.mean(mse_array[:min(100, len(mse_array) // 10)])

    # Avoid log of zero/negative
    mse_clipped = np.maximum(mse_array, 1e-10)
    reduction_db = 10 * np.log10(initial_mse / mse_clipped)

    # Get final reduction (average of last 10% of samples for stability)
    final_samples = max(10, len(reduction_db) // 10)
    final_reduction = np.mean(reduction_db[-final_samples:])

    # Handle case where no improvement
    if final_reduction <= 0:
        return len(mse_array) / sample_rate

    # Threshold is 90% of final reduction
    threshold = threshold_percent * final_reduction

    # Find first sample exceeding threshold
    for i, r in enumerate(reduction_db):
        if r >= threshold:
            return i / sample_rate

    return len(mse_array) / sample_rate  # Never converged


def stability_score(
    mse_history: List[float],
    divergence_threshold: float = 5.0,
    window: int = 100
) -> float:
    """
    Check if the filter remained stable during adaptation.

    Stability is measured by checking for MSE explosions that indicate
    filter divergence. Uses absolute threshold to avoid issues with
    small initial MSE values.

    Args:
        mse_history: List of MSE values
        divergence_threshold: Absolute MSE threshold for divergence (default: 5.0)
        window: Number of samples to skip at start (filter initialization)

    Returns:
        1.0 if stable throughout, 0.0 if diverged at any point

    Example:
        >>> stable_mse = [1.0, 0.5, 0.2, 0.1]  # decreasing
        >>> unstable_mse = [1.0, 0.5, 10.0, 50.0]  # exploding
        >>> print(stability_score(stable_mse))  # 1.0
        >>> print(stability_score(unstable_mse))  # 0.0
    """
    if len(mse_history) < window:
        return 1.0  # Not enough data to judge

    # Check for absolute MSE explosion (indicates divergence)
    # Skip first 'window' samples to allow filter initialization
    for mse in mse_history[window:]:
        if mse > divergence_threshold:
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
