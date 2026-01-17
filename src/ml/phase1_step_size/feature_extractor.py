"""
Feature Extractor for Step Size Prediction

Extracts signal features from the reference signal for use in
adaptive step size selection.
"""

import numpy as np
from scipy.fft import rfft, rfftfreq
from typing import Optional


def extract_features(x: np.ndarray, fs: int = 16000) -> np.ndarray:
    """
    Extract features from reference signal for step size prediction.

    Extracts 8 features that characterize the signal:
    1. Variance - Signal power variability
    2. RMS amplitude - Average signal level
    3. Zero-crossing rate - High-frequency content indicator
    4. Spectral centroid - "Center of mass" of spectrum
    5. Spectral bandwidth - Spread of frequencies
    6. Spectral rolloff - Frequency below which 85% of energy exists
    7. Dominant frequency - Strongest frequency component
    8. Crest factor - Peak-to-RMS ratio

    Args:
        x: Reference signal (1D numpy array)
        fs: Sample rate in Hz (default 16000)

    Returns:
        8-dimensional feature vector as float32 array

    Example:
        >>> signal = np.random.randn(16000)  # 1 second
        >>> features = extract_features(signal, fs=16000)
        >>> print(features.shape)  # (8,)
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

    # ============== Spectral features ==============

    # Compute FFT
    fft_result = rfft(x)
    fft_magnitude = np.abs(fft_result)
    freqs = rfftfreq(len(x), 1.0 / fs)

    # Avoid division by zero
    total_energy = np.sum(fft_magnitude) + 1e-10

    # 4. Spectral centroid
    # Weighted average of frequencies
    centroid = np.sum(freqs * fft_magnitude) / total_energy
    features.append(centroid)

    # 5. Spectral bandwidth
    # Standard deviation of frequencies around centroid
    bandwidth = np.sqrt(
        np.sum(((freqs - centroid) ** 2) * fft_magnitude) / total_energy
    )
    features.append(bandwidth)

    # 6. Spectral rolloff
    # Frequency below which 85% of energy exists
    cumsum = np.cumsum(fft_magnitude)
    rolloff_threshold = 0.85 * cumsum[-1]
    rolloff_idx = np.searchsorted(cumsum, rolloff_threshold)
    rolloff_idx = min(rolloff_idx, len(freqs) - 1)
    rolloff = freqs[rolloff_idx]
    features.append(rolloff)

    # 7. Dominant frequency
    # Frequency with maximum magnitude
    dominant_idx = np.argmax(fft_magnitude)
    dominant_freq = freqs[dominant_idx]
    features.append(dominant_freq)

    # 8. Crest factor
    # Peak amplitude / RMS (indicates "peakiness")
    peak = np.max(np.abs(x))
    crest_factor = peak / (rms + 1e-10)
    features.append(crest_factor)

    return np.array(features, dtype=np.float32)


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
        2D array of features, shape (n_windows, 8)
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
        features: Feature array, shape (8,) or (n_samples, 8)
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


# Feature names for reference
FEATURE_NAMES = [
    'variance',
    'rms',
    'zero_crossing_rate',
    'spectral_centroid',
    'spectral_bandwidth',
    'spectral_rolloff',
    'dominant_frequency',
    'crest_factor',
]
