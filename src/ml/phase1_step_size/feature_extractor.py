"""
Feature Extractor for Step Size Prediction

Extracts signal features from the reference signal for use in
adaptive step size selection.

Version 3: Extended to 16 features for better scenario separation,
including raw amplitude for realistic amplitude training.
"""

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
from typing import Optional


# Number of features extracted
N_FEATURES = 16


def extract_features(x: np.ndarray, fs: int = 16000, n_features: int = 12) -> np.ndarray:
    """
    Extract features from reference signal for step size prediction.

    Extracts 16 features that characterize the signal:

    Time-domain features (1-4):
    1. Variance - Signal power variability
    2. RMS amplitude - Average signal level (normalized)
    3. Zero-crossing rate - High-frequency content indicator
    4. Crest factor - Peak-to-RMS ratio

    Spectral features (5-12):
    5. Spectral centroid - "Center of mass" of spectrum
    6. Spectral bandwidth - Spread of frequencies
    7. Spectral rolloff - Frequency below which 85% of energy exists
    8. Dominant frequency - Strongest frequency component
    9. Low-frequency energy ratio - Energy in 0-1000 Hz band
    10. Harmonic ratio - Tonal content indicator
    11. Spectral entropy - Uniformity of spectrum (normalized 0-1)
    12. Signal stationarity - How constant the signal is over time

    New features for improved step size prediction (13-16):
    13. Raw amplitude - Unnormalized RMS (KEY for scenario separation)
    14. Power variation - Short-term power dynamics (50ms windows)
    15. Spectral flux - Rate of spectral change over time
    16. Low-to-high energy ratio - Energy below vs above 500 Hz

    Args:
        x: Reference signal (1D numpy array)
        fs: Sample rate in Hz (default 16000)
        n_features: Number of features to return (12 or 16, default 12 for compatibility)

    Returns:
        Feature vector as float32 array (12 or 16 dimensions)

    Example:
        >>> signal = np.random.randn(16000)  # 1 second
        >>> features = extract_features(signal, fs=16000)
        >>> print(features.shape)  # (16,)
    """
    features = []

    # Ensure we have enough samples
    if len(x) < 256:
        x = np.pad(x, (0, 256 - len(x)), mode='constant')

    # ============== Time-domain features ==============

    # 1. Variance
    variance = np.var(x)
    features.append(variance)

    # 2. RMS amplitude
    rms = np.sqrt(np.mean(x ** 2))
    features.append(rms)

    # 3. Zero-crossing rate
    # Count sign changes normalized by signal length
    signs = np.sign(x)
    sign_changes = np.abs(np.diff(signs))
    zcr = np.sum(sign_changes) / (2 * len(x))
    features.append(zcr)

    # 4. Crest factor (moved from position 8 for logical grouping)
    # Peak amplitude / RMS (indicates "peakiness")
    peak = np.max(np.abs(x))
    crest_factor = peak / (rms + 1e-10)
    features.append(crest_factor)

    # ============== Spectral features ==============

    # Compute FFT
    fft_result = rfft(x)
    fft_magnitude = np.abs(fft_result)
    freqs = rfftfreq(len(x), 1.0 / fs)

    # Avoid division by zero
    total_energy = np.sum(fft_magnitude) + 1e-10

    # 5. Spectral centroid
    # Weighted average of frequencies
    centroid = np.sum(freqs * fft_magnitude) / total_energy
    features.append(centroid)

    # 6. Spectral bandwidth
    # Standard deviation of frequencies around centroid
    bandwidth = np.sqrt(
        np.sum(((freqs - centroid) ** 2) * fft_magnitude) / total_energy
    )
    features.append(bandwidth)

    # 7. Spectral rolloff
    # Frequency below which 85% of energy exists
    cumsum = np.cumsum(fft_magnitude)
    rolloff_threshold = 0.85 * cumsum[-1]
    rolloff_idx = np.searchsorted(cumsum, rolloff_threshold)
    rolloff_idx = min(rolloff_idx, len(freqs) - 1)
    rolloff = freqs[rolloff_idx]
    features.append(rolloff)

    # 8. Dominant frequency
    # Frequency with maximum magnitude
    dominant_idx = np.argmax(fft_magnitude)
    dominant_freq = freqs[dominant_idx]
    features.append(dominant_freq)

    # 9. Low-frequency energy ratio
    # Fraction of energy in 0-1000 Hz band (relevant for ANC)
    low_freq_mask = freqs <= 1000
    fft_power = fft_magnitude ** 2
    total_fft_energy = np.sum(fft_power) + 1e-10
    low_freq_energy = np.sum(fft_power[low_freq_mask])
    low_freq_ratio = low_freq_energy / total_fft_energy
    features.append(low_freq_ratio)

    # 10. Harmonic ratio
    # Detects tonal content by finding peaks in spectrum
    # Higher ratio = more tonal (like engine noise)
    peak_threshold = np.max(fft_power) * 0.1 if np.max(fft_power) > 0 else 0
    peaks, _ = find_peaks(fft_power, height=peak_threshold)
    harmonic_energy = np.sum(fft_power[peaks]) if len(peaks) > 0 else 0
    harmonic_ratio = harmonic_energy / (total_fft_energy + 1e-10)
    features.append(harmonic_ratio)

    # 11. Spectral entropy
    # Measures uniformity of spectrum (0 = concentrated, 1 = flat/uniform)
    fft_prob = fft_magnitude / (np.sum(fft_magnitude) + 1e-10)
    # Avoid log(0) by adding small epsilon
    fft_prob = np.clip(fft_prob, 1e-10, 1.0)
    spectral_entropy = -np.sum(fft_prob * np.log(fft_prob))
    # Normalize by maximum entropy (log of number of bins)
    max_entropy = np.log(len(fft_magnitude))
    normalized_entropy = spectral_entropy / (max_entropy + 1e-10)
    features.append(normalized_entropy)

    # 12. Signal stationarity
    # Measures how constant the signal is over time
    # Uses variance of RMS in 100ms windows
    window_size_100ms = int(0.1 * fs)  # 100ms windows
    if len(x) >= window_size_100ms * 2:
        n_windows = len(x) // window_size_100ms
        rms_values = [
            np.sqrt(np.mean(x[i * window_size_100ms:(i + 1) * window_size_100ms] ** 2))
            for i in range(n_windows)
        ]
        mean_rms = np.mean(rms_values)
        std_rms = np.std(rms_values)
        # High stationarity = low variation in RMS
        stationarity = 1.0 - (std_rms / (mean_rms + 1e-10))
        stationarity = max(0.0, min(1.0, stationarity))  # Clamp to [0, 1]
    else:
        stationarity = 1.0  # Assume stationary for short signals
    features.append(stationarity)

    # ============== New features for step size prediction (v3) ==============

    # 13. Raw amplitude (NOT normalized - KEY for scenario separation)
    # This is the actual signal level, crucial when training with
    # realistic amplitudes (idle=0.2, city=0.5, highway=0.8, accel=1.0)
    raw_amplitude = rms  # Same as RMS but semantically different
    features.append(raw_amplitude)

    # 14. Power variation (short-term dynamics)
    # Std of RMS in 50ms windows, normalized by mean
    # High variation = non-stationary noise (city, highway)
    # Low variation = stationary noise (idle)
    window_50ms = int(0.05 * fs)  # 50ms windows
    if len(x) >= window_50ms * 4:  # Need at least 4 windows
        n_win = len(x) // window_50ms
        rms_50ms = [
            np.sqrt(np.mean(x[i * window_50ms:(i + 1) * window_50ms] ** 2))
            for i in range(n_win)
        ]
        mean_rms_50 = np.mean(rms_50ms)
        std_rms_50 = np.std(rms_50ms)
        power_variation = std_rms_50 / (mean_rms_50 + 1e-10)
    else:
        power_variation = 0.0
    features.append(power_variation)

    # 15. Spectral flux
    # Rate of spectral change over time
    # High flux = rapidly changing spectrum (city transients)
    # Low flux = stable spectrum (idle engine harmonics)
    n_frames = min(10, len(x) // 1600)  # Up to 10 frames of 100ms each
    if n_frames >= 2:
        frame_len = len(x) // n_frames
        spectra = []
        for i in range(n_frames):
            frame = x[i * frame_len:(i + 1) * frame_len]
            spec = np.abs(rfft(frame))
            # Normalize each spectrum
            spec = spec / (np.sum(spec) + 1e-10)
            spectra.append(spec)
        # Compute flux as mean L1 distance between consecutive spectra
        flux_values = []
        for i in range(len(spectra) - 1):
            # Ensure same length (may differ slightly due to frame_len)
            min_len = min(len(spectra[i]), len(spectra[i + 1]))
            flux = np.sum(np.abs(spectra[i + 1][:min_len] - spectra[i][:min_len]))
            flux_values.append(flux)
        spectral_flux = np.mean(flux_values) if flux_values else 0.0
    else:
        spectral_flux = 0.0
    features.append(spectral_flux)

    # 16. Low-to-high energy ratio
    # Energy below 500 Hz vs above 500 Hz
    # High ratio = low-frequency dominated (engine idle)
    # Low ratio = more high-frequency content (wind, road noise)
    low_mask = freqs <= 500
    high_mask = freqs > 500
    low_energy = np.sum(fft_power[low_mask])
    high_energy = np.sum(fft_power[high_mask])
    lh_ratio = low_energy / (high_energy + 1e-10)
    # Clip to reasonable range
    lh_ratio = min(lh_ratio, 100.0)
    features.append(lh_ratio)

    features = np.array(features, dtype=np.float32)

    # Return only requested number of features (12 for backward compatibility)
    if n_features == 12:
        return features[:12]
    return features


def extract_features_windowed(
    x: np.ndarray,
    fs: int = 16000,
    window_size: int = 4096,
    hop_size: Optional[int] = None
) -> np.ndarray:
    """
    Extract features from signal using sliding windows.

    Useful for analyzing longer signals or detecting changes over time.

    Args:
        x: Reference signal (1D numpy array)
        fs: Sample rate in Hz
        window_size: Size of analysis window in samples
        hop_size: Hop between windows (default: window_size // 2)

    Returns:
        2D array of features, shape (n_windows, 16)
    """
    if hop_size is None:
        hop_size = window_size // 2

    n_samples = len(x)
    features_list = []

    for start in range(0, n_samples - window_size + 1, hop_size):
        window = x[start:start + window_size]
        features = extract_features(window, fs)
        features_list.append(features)

    return np.array(features_list)


def normalize_features(
    features: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> tuple:
    """
    Normalize features to zero mean and unit variance.

    Args:
        features: Feature array, shape (16,) or (n_samples, 16)
        mean: Pre-computed mean (for inference)
        std: Pre-computed std (for inference)

    Returns:
        Tuple of (normalized_features, mean, std)
    """
    if mean is None:
        mean = np.mean(features, axis=0) if features.ndim > 1 else features
    if std is None:
        std = np.std(features, axis=0) if features.ndim > 1 else np.ones_like(features)

    # Avoid division by zero
    std = np.where(std < 1e-10, 1.0, std)

    normalized = (features - mean) / std
    return normalized, mean, std


# Feature names for reference (16 features total)
FEATURE_NAMES = [
    # Time-domain (1-4)
    'variance',
    'rms',
    'zero_crossing_rate',
    'crest_factor',
    # Spectral (5-12)
    'spectral_centroid',
    'spectral_bandwidth',
    'spectral_rolloff',
    'dominant_frequency',
    'low_freq_ratio',
    'harmonic_ratio',
    'spectral_entropy',
    'stationarity',
    # New features for step size prediction (13-16)
    'raw_amplitude',
    'power_variation',
    'spectral_flux',
    'lh_energy_ratio',
]
