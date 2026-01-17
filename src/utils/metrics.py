"""
Metrics and Performance Evaluation for ANC

Functions to calculate noise reduction and other performance metrics.
"""

import numpy as np
from typing import Dict, Tuple


def calculate_noise_reduction_db(
    original: np.ndarray,
    cancelled: np.ndarray,
    window: int = None
) -> float:
    """
    Calculate noise reduction in decibels.

    Args:
        original: Original noise signal (d(n))
        cancelled: Signal after ANC (e(n))
        window: Optional window size for calculation (uses last N samples)

    Returns:
        Noise reduction in dB
    """
    if window is not None:
        original = original[-window:]
        cancelled = cancelled[-window:]

    original_power = np.mean(original ** 2)
    cancelled_power = np.mean(cancelled ** 2)

    if cancelled_power < 1e-10:
        return 60.0  # Cap at 60 dB

    return 10 * np.log10(original_power / cancelled_power)


def calculate_frequency_reduction(
    original: np.ndarray,
    cancelled: np.ndarray,
    sample_rate: int,
    freq_range: Tuple[float, float] = (20, 300)
) -> Dict[str, float]:
    """
    Calculate noise reduction in specific frequency bands.

    Args:
        original: Original noise signal
        cancelled: Signal after ANC
        sample_rate: Sample rate in Hz
        freq_range: Frequency range of interest (min, max) in Hz

    Returns:
        Dict with frequency band reductions
    """
    from scipy import signal

    # Compute PSDs
    f, psd_orig = signal.welch(original, sample_rate, nperseg=1024)
    _, psd_cancel = signal.welch(cancelled, sample_rate, nperseg=1024)

    # Find indices within frequency range
    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])

    # Overall reduction in band
    orig_power = np.sum(psd_orig[freq_mask])
    cancel_power = np.sum(psd_cancel[freq_mask])

    if cancel_power < 1e-10:
        overall_reduction = 60.0
    else:
        overall_reduction = 10 * np.log10(orig_power / cancel_power)

    # Reduction at specific frequencies
    results = {
        'overall_db': overall_reduction,
        'freq_range': freq_range
    }

    # Calculate reduction at key frequencies
    key_freqs = [50, 100, 150, 200, 250]
    for freq in key_freqs:
        if freq >= freq_range[0] and freq <= freq_range[1]:
            idx = np.argmin(np.abs(f - freq))
            if psd_cancel[idx] > 1e-10:
                reduction = 10 * np.log10(psd_orig[idx] / psd_cancel[idx])
            else:
                reduction = 60.0
            results[f'{freq}Hz_db'] = reduction

    return results


def calculate_convergence_time(
    mse_history: np.ndarray,
    sample_rate: int,
    threshold_db: float = -10.0
) -> float:
    """
    Calculate time to reach convergence.

    Args:
        mse_history: Mean squared error over time
        sample_rate: Sample rate in Hz
        threshold_db: Threshold in dB relative to initial MSE

    Returns:
        Convergence time in seconds (or -1 if not converged)
    """
    if len(mse_history) < 100:
        return -1.0

    # Smooth MSE
    window = min(100, len(mse_history) // 10)
    mse_smooth = np.convolve(mse_history, np.ones(window)/window, mode='valid')

    # Initial MSE (average of first 100 samples)
    initial_mse = np.mean(mse_history[:100])

    # Threshold MSE
    threshold_linear = initial_mse * (10 ** (threshold_db / 10))

    # Find first crossing
    converged_indices = np.where(mse_smooth < threshold_linear)[0]

    if len(converged_indices) == 0:
        return -1.0

    convergence_sample = converged_indices[0] + window // 2
    return convergence_sample / sample_rate


def generate_metrics_report(
    results: Dict[str, np.ndarray],
    sample_rate: int
) -> str:
    """
    Generate a text report of ANC performance metrics.

    Args:
        results: Dict containing 'desired', 'error', 'mse' arrays
        sample_rate: Sample rate in Hz

    Returns:
        Formatted string report
    """
    report_lines = [
        "=" * 50,
        "ANC PERFORMANCE METRICS REPORT",
        "=" * 50,
        ""
    ]

    # Overall noise reduction
    nr_db = calculate_noise_reduction_db(
        results['desired'],
        results['error'],
        window=sample_rate  # Last 1 second
    )
    report_lines.append(f"Overall Noise Reduction: {nr_db:.1f} dB")

    # Frequency-specific reduction
    freq_results = calculate_frequency_reduction(
        results['desired'],
        results['error'],
        sample_rate
    )
    report_lines.append(f"Reduction in 20-300 Hz band: {freq_results['overall_db']:.1f} dB")

    # Convergence time
    if 'mse' in results:
        conv_time = calculate_convergence_time(
            results['mse'],
            sample_rate,
            threshold_db=-10.0
        )
        if conv_time > 0:
            report_lines.append(f"Convergence time (-10 dB): {conv_time:.3f} s")
        else:
            report_lines.append("Convergence time: Did not converge")

    report_lines.append("")
    report_lines.append("=" * 50)

    return "\n".join(report_lines)
