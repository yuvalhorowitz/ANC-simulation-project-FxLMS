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

    # Add annotation for final MSE (note: this is smoothed)
    final_mse = mse_smooth[-1] if len(mse_smooth) > 0 else 0
    ax.annotate(f'Smoothed MSE: {final_mse:.2e}',
                xy=(t[-1], final_mse),
                xytext=(t[-1]*0.7, final_mse*10),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10)

    plt.tight_layout()
    return fig


def plot_convergence_db(results: Dict[str, Any]) -> plt.Figure:
    """
    Plot noise reduction (dB) over time with 90% convergence threshold.

    Shows how the noise reduction builds up over time and marks
    the point where 90% of final reduction is achieved.

    Uses the same calculation as the official noise reduction metric:
    NR(dB) = 10 * log10(desired_power / error_power)

    Args:
        results: Simulation results dictionary

    Returns:
        Matplotlib figure
    """
    fs = results['fs']
    desired = np.array(results['desired'])  # Noise at error mic (without ANC effect)
    error = np.array(results['error'])      # Residual after ANC

    # Compute noise reduction over time using sliding window
    window = 500  # Samples for averaging (31.25 ms at 16kHz)
    if len(desired) < window * 2:
        window = max(50, len(desired) // 10)

    step = window // 2  # 50% overlap

    # Calculate running noise reduction in dB
    reduction_db = []
    times = []

    for i in range(window, len(desired), step):
        # Power of desired signal (what we want to cancel) in this window
        d_power = np.mean(desired[i - window:i] ** 2)
        # Power of error signal (residual after ANC) in this window
        e_power = np.mean(error[i - window:i] ** 2)

        if e_power > 1e-10 and d_power > 1e-10:
            db = 10 * np.log10(d_power / e_power)
        else:
            db = 0

        # Clamp to reasonable range
        db = max(-5, min(30, db))
        reduction_db.append(db)
        times.append(i / fs)

    reduction_db = np.array(reduction_db)
    times = np.array(times)

    if len(reduction_db) == 0:
        # Fallback if not enough data
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        return fig

    # Calculate final reduction (average of last 10% for stability)
    final_samples = max(1, len(reduction_db) // 10)
    final_reduction = np.mean(reduction_db[-final_samples:])

    # 90% threshold (only if final reduction is positive)
    if final_reduction > 0:
        threshold_90 = 0.90 * final_reduction
    else:
        threshold_90 = 0

    # Find convergence point
    conv_time = None
    conv_db = None
    if threshold_90 > 0:
        for i, (t, db) in enumerate(zip(times, reduction_db)):
            if db >= threshold_90:
                conv_time = t
                conv_db = db
                break

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot noise reduction over time
    ax.plot(times, reduction_db, 'b-', linewidth=1.5, label='Noise Reduction')

    # Plot final reduction line
    ax.axhline(y=final_reduction, color='g', linestyle='--', linewidth=1.5,
               label=f'Final: {final_reduction:.1f} dB')

    # Plot 90% threshold line (only if meaningful)
    if threshold_90 > 0.5:
        ax.axhline(y=threshold_90, color='orange', linestyle=':', linewidth=1.5,
                   label=f'90% threshold: {threshold_90:.1f} dB')

    # Mark convergence point
    if conv_time is not None:
        ax.axvline(x=conv_time, color='r', linestyle='--', alpha=0.7)
        ax.scatter([conv_time], [conv_db], color='r', s=100, zorder=5,
                   label=f'Converged: {conv_time:.2f} s')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Noise Reduction (dB)')
    ax.set_title('Noise Reduction Over Time (90% Convergence)')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Set y-axis to show 0 and a bit of negative for context
    y_min = min(-1, np.min(reduction_db) - 1)
    y_max = max(5, np.max(reduction_db) * 1.1)
    ax.set_ylim(y_min, y_max)

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


# Reference mic color scheme
REF_MIC_COLORS = {
    'firewall': '#e67e22',   # Orange - engine noise
    'floor': '#9b59b6',      # Purple - road noise
    'a_pillar': '#1abc9c',   # Teal - wind noise
    'dashboard': '#3498db',  # Blue - general
}


def plot_ref_mic_signals_time(results: Dict[str, Any]) -> plt.Figure:
    """
    Plot individual reference mic signals in time domain.

    Args:
        results: Simulation results dictionary

    Returns:
        Matplotlib figure
    """
    fs = results['fs']
    ref_mic_signals = results.get('ref_mic_signals', {})
    ref_mic_names = results.get('ref_mic_names', [])

    if not ref_mic_signals:
        # Fallback to single reference signal
        ref_mic_signals = {'reference': results.get('reference', np.zeros(100))}
        ref_mic_names = ['reference']

    # Show last 100ms
    show_samples = int(0.1 * fs)
    t_ms = np.arange(show_samples) / fs * 1000

    fig, ax = plt.subplots(figsize=(10, 5))

    for name in ref_mic_names:
        signal = ref_mic_signals[name]
        color = REF_MIC_COLORS.get(name, '#3498db')
        label = name.replace('_', ' ').title()
        ax.plot(t_ms, signal[-show_samples:], color=color, linewidth=1.2, alpha=0.8, label=label)

    # Also plot the averaged signal if in multi-ref mode
    if len(ref_mic_names) > 1 and 'reference' in results:
        ax.plot(t_ms, results['reference'][-show_samples:], 'k--', linewidth=2, alpha=0.7, label='Averaged')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Reference Mic Signals (Time Domain - Last 100ms)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_ref_mic_signals_freq(results: Dict[str, Any]) -> plt.Figure:
    """
    Plot individual reference mic signals in frequency domain.

    Args:
        results: Simulation results dictionary

    Returns:
        Matplotlib figure
    """
    fs = results['fs']
    ref_mic_signals = results.get('ref_mic_signals', {})
    ref_mic_names = results.get('ref_mic_names', [])

    if not ref_mic_signals:
        # Fallback to single reference signal
        ref_mic_signals = {'reference': results.get('reference', np.zeros(100))}
        ref_mic_names = ['reference']

    fig, ax = plt.subplots(figsize=(10, 5))

    # Use steady-state portion
    for name in ref_mic_names:
        signal = ref_mic_signals[name]
        steady_start = len(signal) // 2
        f, psd = scipy_signal.welch(signal[steady_start:], fs, nperseg=2048)
        color = REF_MIC_COLORS.get(name, '#3498db')
        label = name.replace('_', ' ').title()
        ax.semilogy(f, psd, color=color, linewidth=1.5, alpha=0.8, label=label)

    # Mark target frequency range
    ax.axvline(x=20, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=300, color='gray', linestyle='--', alpha=0.5)
    ax.axvspan(20, 300, alpha=0.1, color='blue', label='Target range')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('Reference Mic Signals (Frequency Domain)')
    ax.set_xlim(0, 500)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_error_mic_time(results: Dict[str, Any]) -> plt.Figure:
    """
    Plot error mic signal before and after ANC in time domain.

    Args:
        results: Simulation results dictionary

    Returns:
        Matplotlib figure
    """
    fs = results['fs']
    desired = results['desired']  # Before ANC (noise at error mic)
    error = results['error']      # After ANC

    # Show last 100ms
    show_samples = int(0.1 * fs)
    t_ms = np.arange(show_samples) / fs * 1000

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Before ANC
    axes[0].plot(t_ms, desired[-show_samples:], 'r-', linewidth=1.2, alpha=0.8)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Error Mic: Before ANC (Noise at Driver Ear)')
    axes[0].grid(True, alpha=0.3)

    # After ANC
    axes[1].plot(t_ms, error[-show_samples:], 'g-', linewidth=1.2, alpha=0.8)
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title(f'Error Mic: After ANC ({results["noise_reduction_db"]:.1f} dB reduction)')
    axes[1].grid(True, alpha=0.3)

    # Match y-axis scales for comparison
    max_amp = max(np.max(np.abs(desired[-show_samples:])), np.max(np.abs(error[-show_samples:])))
    for ax in axes:
        ax.set_ylim(-max_amp * 1.1, max_amp * 1.1)

    plt.tight_layout()
    return fig


def plot_error_mic_freq(results: Dict[str, Any]) -> plt.Figure:
    """
    Plot error mic signal before and after ANC in frequency domain.

    Args:
        results: Simulation results dictionary

    Returns:
        Matplotlib figure
    """
    fs = results['fs']
    desired = results['desired']  # Before ANC
    error = results['error']      # After ANC

    # Use steady-state portion
    steady_start = len(desired) // 2

    # Compute PSD
    f_d, psd_d = scipy_signal.welch(desired[steady_start:], fs, nperseg=2048)
    f_e, psd_e = scipy_signal.welch(error[steady_start:], fs, nperseg=2048)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.semilogy(f_d, psd_d, 'r-', linewidth=2, alpha=0.8, label='Before ANC')
    ax.semilogy(f_e, psd_e, 'g-', linewidth=2, alpha=0.8, label='After ANC')

    # Mark target frequency range
    ax.axvline(x=20, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=300, color='gray', linestyle='--', alpha=0.5)
    ax.axvspan(20, 300, alpha=0.1, color='blue', label='Target range')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('Error Mic Signal (Frequency Domain)')
    ax.set_xlim(0, 500)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_noise_source_time(results: Dict[str, Any]) -> plt.Figure:
    """
    Plot raw noise source signal in time domain.

    Args:
        results: Simulation results dictionary

    Returns:
        Matplotlib figure
    """
    fs = results['fs']
    noise_source = results.get('noise_source', np.zeros(100))

    # Show last 100ms
    show_samples = int(0.1 * fs)
    t_ms = np.arange(show_samples) / fs * 1000

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(t_ms, noise_source[-show_samples:], 'b-', linewidth=1.2, alpha=0.8)

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Noise Source Signal (Time Domain - Last 100ms)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_noise_source_freq(results: Dict[str, Any]) -> plt.Figure:
    """
    Plot raw noise source signal in frequency domain.

    Args:
        results: Simulation results dictionary

    Returns:
        Matplotlib figure
    """
    fs = results['fs']
    noise_source = results.get('noise_source', np.zeros(100))

    # Use steady-state portion
    steady_start = len(noise_source) // 2

    # Compute PSD
    f, psd = scipy_signal.welch(noise_source[steady_start:], fs, nperseg=2048)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.semilogy(f, psd, 'b-', linewidth=2, alpha=0.8)

    # Mark target frequency range
    ax.axvline(x=20, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=300, color='gray', linestyle='--', alpha=0.5)
    ax.axvspan(20, 300, alpha=0.1, color='blue', label='Target range (20-300 Hz)')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('Noise Source Signal (Frequency Domain)')
    ax.set_xlim(0, 500)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_mic_signal_plots(results: Dict[str, Any]) -> Dict[str, plt.Figure]:
    """
    Create microphone signal visualization plots.

    Args:
        results: Simulation results dictionary

    Returns:
        Dictionary of plot names to figures
    """
    plots = {
        'error_mic_time': plot_error_mic_time(results),
        'error_mic_freq': plot_error_mic_freq(results),
    }

    # Only add ref mic plots if we have multiple ref mics
    if 'ref_mic_signals' in results and len(results.get('ref_mic_names', [])) > 1:
        plots['ref_mic_time'] = plot_ref_mic_signals_time(results)
        plots['ref_mic_freq'] = plot_ref_mic_signals_freq(results)

    return plots
