"""
Visualization Plots for ANC Playground

Creates matplotlib figures for displaying simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from typing import Dict, Any


def plot_before_after(results: Dict[str, Any]) -> plt.Figure:
    """
    Plot time-domain before/after comparison.

    Args:
        results: Simulation results dictionary

    Returns:
        Matplotlib figure
    """
    fs = results['fs']
    desired = results['desired']
    error = results['error']

    # Show last 100ms
    show_samples = int(0.1 * fs)
    t_ms = np.arange(show_samples) / fs * 1000

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(t_ms, desired[-show_samples:], 'r-', linewidth=1.2, alpha=0.8, label='Noise (without ANC)')
    ax.plot(t_ms, error[-show_samples:], 'g-', linewidth=1.2, alpha=0.8, label='With ANC')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f"Before vs After ANC ({results['noise_reduction_db']:.1f} dB reduction)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_spectrum(results: Dict[str, Any]) -> plt.Figure:
    """
    Plot frequency spectrum comparison.

    Args:
        results: Simulation results dictionary

    Returns:
        Matplotlib figure
    """
    fs = results['fs']
    desired = results['desired']
    error = results['error']

    # Use steady-state portion
    steady_start = len(desired) // 2

    # Compute PSD
    f_d, psd_d = scipy_signal.welch(desired[steady_start:], fs, nperseg=2048)
    f_e, psd_e = scipy_signal.welch(error[steady_start:], fs, nperseg=2048)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.semilogy(f_d, psd_d, 'r-', linewidth=1.5, alpha=0.8, label='Noise')
    ax.semilogy(f_e, psd_e, 'g-', linewidth=1.5, alpha=0.8, label='With ANC')

    # Mark target frequency range
    ax.axvline(x=20, color='gray', linestyle='--', alpha=0.5, label='Target range')
    ax.axvline(x=300, color='gray', linestyle='--', alpha=0.5)
    ax.axvspan(20, 300, alpha=0.1, color='blue')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('Frequency Spectrum (20-300 Hz target)')
    ax.set_xlim(0, 350)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_convergence(results: Dict[str, Any]) -> plt.Figure:
    """
    Plot MSE convergence over time.

    Args:
        results: Simulation results dictionary

    Returns:
        Matplotlib figure
    """
    fs = results['fs']
    mse = results['mse']

    # Smooth MSE
    window = 200
    if len(mse) > window:
        mse_smooth = np.convolve(mse, np.ones(window)/window, mode='valid')
        t = np.arange(len(mse_smooth)) / fs
    else:
        mse_smooth = mse
        t = np.arange(len(mse)) / fs

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.semilogy(t, mse_smooth, 'b-', linewidth=1.5)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('Algorithm Convergence')
    ax.grid(True, alpha=0.3)

    # Add annotation for final MSE
    final_mse = mse_smooth[-1] if len(mse_smooth) > 0 else 0
    ax.annotate(f'Final MSE: {final_mse:.2e}',
                xy=(t[-1], final_mse),
                xytext=(t[-1]*0.7, final_mse*10),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10)

    plt.tight_layout()
    return fig


def plot_filter_coefficients(results: Dict[str, Any]) -> plt.Figure:
    """
    Plot learned adaptive filter weights.

    Args:
        results: Simulation results dictionary

    Returns:
        Matplotlib figure
    """
    weights = results['weights']

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(weights, 'b-', linewidth=1)
    ax.fill_between(range(len(weights)), weights, alpha=0.3)

    ax.set_xlabel('Tap Index')
    ax.set_ylabel('Weight Value')
    ax.set_title(f'Learned Filter Coefficients ({len(weights)} taps)')
    ax.grid(True, alpha=0.3)

    # Add zero line
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    return fig


def plot_noise_reduction_over_time(results: Dict[str, Any]) -> plt.Figure:
    """
    Plot noise reduction in dB over time windows.

    Args:
        results: Simulation results dictionary

    Returns:
        Matplotlib figure
    """
    fs = results['fs']
    desired = results['desired']
    error = results['error']

    # Calculate in 0.5s windows
    window_size = int(0.5 * fs)
    n_windows = len(desired) // window_size

    nr_over_time = []
    time_points = []

    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        d_pow = np.mean(desired[start:end] ** 2)
        e_pow = np.mean(error[start:end] ** 2)

        if e_pow > 1e-10:
            nr = 10 * np.log10(d_pow / e_pow)
        else:
            nr = 30
        nr_over_time.append(nr)
        time_points.append((start + end) / 2 / fs)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(time_points, nr_over_time, 'g-o', linewidth=2, markersize=6)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='0 dB (no reduction)')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Noise Reduction (dB)')
    ax.set_title('Performance Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_all_plots(results: Dict[str, Any]) -> Dict[str, plt.Figure]:
    """
    Create all visualization plots.

    Args:
        results: Simulation results dictionary

    Returns:
        Dictionary of plot names to figures
    """
    return {
        'before_after': plot_before_after(results),
        'spectrum': plot_spectrum(results),
        'convergence': plot_convergence(results),
        'filter_coefficients': plot_filter_coefficients(results),
        'nr_over_time': plot_noise_reduction_over_time(results),
    }
