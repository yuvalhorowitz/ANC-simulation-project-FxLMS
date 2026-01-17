"""
Audio File Utilities

Functions to convert simulation pressure signals to audio WAV files.
"""

import numpy as np
from scipy.io import wavfile
from typing import Optional, Tuple
import os


def normalize_audio(signal: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """
    Normalize audio signal to a target dB level.

    Args:
        signal: Input signal array
        target_db: Target peak level in dB (default -3 dB to avoid clipping)

    Returns:
        Normalized signal as float array in range [-1, 1]
    """
    # Avoid division by zero
    max_val = np.max(np.abs(signal))
    if max_val < 1e-10:
        return signal

    # Normalize to [-1, 1]
    normalized = signal / max_val

    # Apply target dB level
    target_linear = 10 ** (target_db / 20)
    return normalized * target_linear


def save_wav(
    filename: str,
    signal: np.ndarray,
    sample_rate: int = 16000,
    normalize: bool = True,
    bit_depth: int = 16
) -> str:
    """
    Save a signal array as a WAV audio file.

    Args:
        filename: Output filename (with or without .wav extension)
        signal: Audio signal array
        sample_rate: Sample rate in Hz
        normalize: Whether to normalize the signal
        bit_depth: Bit depth (16 or 32)

    Returns:
        Full path to saved file
    """
    # Ensure .wav extension
    if not filename.endswith('.wav'):
        filename += '.wav'

    # Create directory if needed
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)

    # Normalize if requested
    if normalize:
        audio = normalize_audio(signal)
    else:
        audio = signal

    # Convert to appropriate bit depth
    if bit_depth == 16:
        # Scale to 16-bit integer range
        audio_int = (audio * 32767).astype(np.int16)
    else:
        # 32-bit float
        audio_int = audio.astype(np.float32)

    # Save file
    wavfile.write(filename, sample_rate, audio_int)

    return filename


def save_comparison_wav(
    filename_prefix: str,
    original: np.ndarray,
    cancelled: np.ndarray,
    sample_rate: int = 16000,
    output_dir: str = "output/audio"
) -> Tuple[str, str, str]:
    """
    Save original, cancelled, and combined comparison audio files.

    Creates three files:
    - {prefix}_original.wav: The original noise
    - {prefix}_cancelled.wav: After ANC
    - {prefix}_comparison.wav: Original then cancelled (with silence gap)

    Args:
        filename_prefix: Base name for files
        original: Original noise signal
        cancelled: Signal after ANC
        sample_rate: Sample rate in Hz
        output_dir: Output directory

    Returns:
        Tuple of (original_path, cancelled_path, comparison_path)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find common normalization factor (so volumes are comparable)
    max_original = np.max(np.abs(original))
    max_cancelled = np.max(np.abs(cancelled))
    max_val = max(max_original, max_cancelled, 1e-10)

    # Normalize both to same scale
    original_norm = original / max_val * 0.7  # -3dB headroom
    cancelled_norm = cancelled / max_val * 0.7

    # Save individual files
    original_path = os.path.join(output_dir, f"{filename_prefix}_original.wav")
    cancelled_path = os.path.join(output_dir, f"{filename_prefix}_cancelled.wav")

    wavfile.write(original_path, sample_rate, (original_norm * 32767).astype(np.int16))
    wavfile.write(cancelled_path, sample_rate, (cancelled_norm * 32767).astype(np.int16))

    # Create comparison: original -> silence -> cancelled
    silence_duration = 0.5  # seconds
    silence = np.zeros(int(silence_duration * sample_rate))

    comparison = np.concatenate([original_norm, silence, cancelled_norm])
    comparison_path = os.path.join(output_dir, f"{filename_prefix}_comparison.wav")
    wavfile.write(comparison_path, sample_rate, (comparison * 32767).astype(np.int16))

    return original_path, cancelled_path, comparison_path


def pressure_to_audio(
    pressure: np.ndarray,
    simulation_dt: float,
    target_sample_rate: int = 16000
) -> np.ndarray:
    """
    Convert pressure simulation data to audio-rate signal.

    The wave simulation may run at a different time step than audio.
    This function resamples to the target audio sample rate.

    Args:
        pressure: Pressure values from simulation
        simulation_dt: Time step of simulation in seconds
        target_sample_rate: Desired audio sample rate

    Returns:
        Resampled audio signal
    """
    from scipy import signal as scipy_signal

    # Calculate simulation sample rate
    sim_sample_rate = 1.0 / simulation_dt

    # If rates are similar, just return normalized
    if abs(sim_sample_rate - target_sample_rate) < 100:
        return normalize_audio(pressure)

    # Resample to target rate
    num_samples = int(len(pressure) * target_sample_rate / sim_sample_rate)
    resampled = scipy_signal.resample(pressure, num_samples)

    return normalize_audio(resampled)


def generate_test_tone(
    frequency: float = 440.0,
    duration: float = 1.0,
    sample_rate: int = 16000,
    amplitude: float = 0.5
) -> np.ndarray:
    """
    Generate a simple test tone.

    Args:
        frequency: Tone frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        amplitude: Peak amplitude (0-1)

    Returns:
        Tone signal array
    """
    t = np.arange(int(duration * sample_rate)) / sample_rate
    return amplitude * np.sin(2 * np.pi * frequency * t)
