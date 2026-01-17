"""
Audio Player Component for ANC Playground

Handles audio playback in Streamlit using st.audio().
"""

import numpy as np
import io
from scipy.io import wavfile
import streamlit as st
from typing import Dict, Any


def normalize_audio(signal: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """
    Normalize audio signal to prevent clipping.

    Args:
        signal: Input signal
        target_db: Target peak level in dB

    Returns:
        Normalized signal
    """
    max_val = np.max(np.abs(signal))
    if max_val < 1e-10:
        return signal

    normalized = signal / max_val
    target_linear = 10 ** (target_db / 20)
    return normalized * target_linear


def signal_to_wav_bytes(signal: np.ndarray, sample_rate: int = 16000) -> bytes:
    """
    Convert numpy signal to WAV file bytes for Streamlit audio player.

    Args:
        signal: Audio signal array
        sample_rate: Sample rate in Hz

    Returns:
        WAV file as bytes
    """
    # Normalize to prevent clipping
    audio = normalize_audio(signal)

    # Convert to 16-bit integer
    audio_int = (audio * 32767).astype(np.int16)

    # Write to bytes buffer
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, audio_int)
    buffer.seek(0)

    return buffer.read()


def create_comparison_audio(original: np.ndarray, cancelled: np.ndarray,
                           sample_rate: int = 16000, gap_seconds: float = 0.5) -> bytes:
    """
    Create comparison audio: original -> silence -> cancelled.

    Args:
        original: Original noise signal
        cancelled: Cancelled signal
        sample_rate: Sample rate in Hz
        gap_seconds: Silence gap between segments

    Returns:
        WAV file as bytes
    """
    # Normalize both to same scale
    max_val = max(np.max(np.abs(original)), np.max(np.abs(cancelled)), 1e-10)
    original_norm = original / max_val * 0.7
    cancelled_norm = cancelled / max_val * 0.7

    # Create silence gap
    silence = np.zeros(int(gap_seconds * sample_rate))

    # Concatenate
    comparison = np.concatenate([original_norm, silence, cancelled_norm])

    return signal_to_wav_bytes(comparison, sample_rate)


def render_audio_player(results: Dict[str, Any]):
    """
    Render audio players in Streamlit.

    Args:
        results: Simulation results dictionary
    """
    st.subheader("Audio Playback")

    fs = results['fs']
    desired = results['desired']
    error = results['error']

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Original Noise**")
        original_bytes = signal_to_wav_bytes(desired, fs)
        st.audio(original_bytes, format='audio/wav')

    with col2:
        st.markdown("**With ANC**")
        cancelled_bytes = signal_to_wav_bytes(error, fs)
        st.audio(cancelled_bytes, format='audio/wav')

    with col3:
        st.markdown("**Comparison**")
        st.caption("Original → Silence → With ANC")
        comparison_bytes = create_comparison_audio(desired, error, fs)
        st.audio(comparison_bytes, format='audio/wav')
