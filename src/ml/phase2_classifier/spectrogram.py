"""
Mel Spectrogram Extractor

Extracts mel spectrograms from audio signals for noise classification.
"""

import numpy as np
from typing import Optional, Tuple


def compute_mel_spectrogram(
    signal: np.ndarray,
    fs: int = 16000,
    n_fft: int = 512,
    hop_length: int = 256,
    n_mels: int = 64,
    fmin: float = 20.0,
    fmax: Optional[float] = None
) -> np.ndarray:
    """
    Compute mel spectrogram from audio signal.

    Uses a simple implementation without external dependencies like librosa.
    For production, consider using torchaudio.transforms.MelSpectrogram.

    Args:
        signal: Audio signal, 1D array
        fs: Sample rate in Hz
        n_fft: FFT window size
        hop_length: Hop between frames
        n_mels: Number of mel frequency bins
        fmin: Minimum frequency for mel filterbank
        fmax: Maximum frequency (defaults to fs/2)

    Returns:
        Mel spectrogram of shape (n_mels, n_frames)
    """
    if fmax is None:
        fmax = fs / 2

    # Compute STFT
    stft = _compute_stft(signal, n_fft, hop_length)

    # Compute power spectrogram
    power_spec = np.abs(stft) ** 2

    # Create mel filterbank
    mel_filterbank = _create_mel_filterbank(n_fft, fs, n_mels, fmin, fmax)

    # Apply mel filterbank
    mel_spec = np.dot(mel_filterbank, power_spec)

    # Convert to log scale (dB)
    mel_spec_db = 10 * np.log10(mel_spec + 1e-10)

    return mel_spec_db


def _compute_stft(
    signal: np.ndarray,
    n_fft: int,
    hop_length: int
) -> np.ndarray:
    """
    Compute Short-Time Fourier Transform.

    Args:
        signal: Input signal
        n_fft: FFT window size
        hop_length: Hop between frames

    Returns:
        STFT magnitude, shape (n_fft//2 + 1, n_frames)
    """
    # Pad signal
    pad_length = n_fft // 2
    signal_padded = np.pad(signal, (pad_length, pad_length), mode='reflect')

    # Create Hann window
    window = np.hanning(n_fft)

    # Compute number of frames
    n_frames = 1 + (len(signal_padded) - n_fft) // hop_length

    # Initialize output
    stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)

    # Compute STFT frame by frame
    for i in range(n_frames):
        start = i * hop_length
        frame = signal_padded[start:start + n_fft] * window
        stft[:, i] = np.fft.rfft(frame)

    return stft


def _hz_to_mel(hz: float) -> float:
    """Convert frequency from Hz to mel scale."""
    return 2595 * np.log10(1 + hz / 700)


def _mel_to_hz(mel: float) -> float:
    """Convert frequency from mel scale to Hz."""
    return 700 * (10 ** (mel / 2595) - 1)


def _create_mel_filterbank(
    n_fft: int,
    fs: int,
    n_mels: int,
    fmin: float,
    fmax: float
) -> np.ndarray:
    """
    Create mel filterbank matrix.

    Args:
        n_fft: FFT size
        fs: Sample rate
        n_mels: Number of mel bands
        fmin: Minimum frequency
        fmax: Maximum frequency

    Returns:
        Filterbank matrix of shape (n_mels, n_fft//2 + 1)
    """
    # Convert frequency range to mel scale
    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)

    # Create mel points evenly spaced in mel scale
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])

    # Convert to FFT bin indices
    bin_points = np.floor((n_fft + 1) * hz_points / fs).astype(int)

    # Create filterbank
    n_bins = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, n_bins))

    for i in range(n_mels):
        # Left slope
        for j in range(bin_points[i], bin_points[i + 1]):
            if bin_points[i + 1] != bin_points[i]:
                filterbank[i, j] = (j - bin_points[i]) / (bin_points[i + 1] - bin_points[i])

        # Right slope
        for j in range(bin_points[i + 1], bin_points[i + 2]):
            if bin_points[i + 2] != bin_points[i + 1]:
                filterbank[i, j] = (bin_points[i + 2] - j) / (bin_points[i + 2] - bin_points[i + 1])

    return filterbank


def normalize_spectrogram(
    mel_spec: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize mel spectrogram to zero mean and unit variance.

    Args:
        mel_spec: Mel spectrogram of shape (n_mels, n_frames)
        mean: Pre-computed mean (optional)
        std: Pre-computed std (optional)

    Returns:
        Tuple of (normalized_spec, mean, std)
    """
    if mean is None:
        mean = np.mean(mel_spec)
    if std is None:
        std = np.std(mel_spec)

    normalized = (mel_spec - mean) / (std + 1e-8)

    return normalized, mean, std


def extract_mel_spectrogram(
    signal: np.ndarray,
    fs: int = 16000,
    target_shape: Tuple[int, int] = (64, 32),
    normalize: bool = True
) -> np.ndarray:
    """
    Extract mel spectrogram with fixed output shape for CNN input.

    This is the main function to use for classification.

    Args:
        signal: Audio signal (should be ~1 second at 16kHz)
        fs: Sample rate
        target_shape: Output shape (n_mels, n_frames)
        normalize: Whether to normalize to zero mean/unit variance

    Returns:
        Mel spectrogram of shape target_shape
    """
    n_mels, target_frames = target_shape

    # Compute hop length to achieve target number of frames
    # n_frames ≈ len(signal) / hop_length
    hop_length = max(1, len(signal) // target_frames)
    n_fft = hop_length * 2

    # Compute mel spectrogram
    mel_spec = compute_mel_spectrogram(
        signal,
        fs=fs,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    # Resize to target shape if needed
    if mel_spec.shape[1] != target_frames:
        mel_spec = _resize_spectrogram(mel_spec, target_frames)

    # Normalize
    if normalize:
        mel_spec, _, _ = normalize_spectrogram(mel_spec)

    return mel_spec


def _resize_spectrogram(spec: np.ndarray, target_frames: int) -> np.ndarray:
    """
    Resize spectrogram to target number of frames using linear interpolation.

    Args:
        spec: Input spectrogram (n_mels, n_frames)
        target_frames: Target number of frames

    Returns:
        Resized spectrogram (n_mels, target_frames)
    """
    n_mels, n_frames = spec.shape

    if n_frames == target_frames:
        return spec

    # Create interpolation indices
    old_indices = np.linspace(0, n_frames - 1, n_frames)
    new_indices = np.linspace(0, n_frames - 1, target_frames)

    # Interpolate each mel band
    resized = np.zeros((n_mels, target_frames))
    for i in range(n_mels):
        resized[i] = np.interp(new_indices, old_indices, spec[i])

    return resized


def spectrogram_to_tensor(mel_spec: np.ndarray) -> np.ndarray:
    """
    Prepare mel spectrogram for CNN input.

    Adds batch and channel dimensions.

    Args:
        mel_spec: Mel spectrogram of shape (n_mels, n_frames)

    Returns:
        Tensor-ready array of shape (1, 1, n_mels, n_frames)
    """
    return mel_spec[np.newaxis, np.newaxis, :, :].astype(np.float32)
