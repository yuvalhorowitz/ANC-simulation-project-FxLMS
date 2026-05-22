"""
Audio Player Component for ANC Playground

Handles audio playback in Streamlit using st.audio().
"""

import numpy as np
import io
from scipy.io import wavfile
import streamlit as st
from typing import Dict, Any


def normalize_audio(signal: np.ndarray, target_db: float = -3.0,
                    reference_peak: float = None) -> np.ndarray:
    """
    Normalize audio signal to prevent clipping.

    If reference_peak is given, the signal is scaled relative to that peak —
    so two signals normalized with the SAME reference_peak preserve their
    relative loudness. This is essential for fairly comparing original vs
    cancelled audio: independently normalizing each would defeat the
    comparison by boosting the quieter signal back to the same peak.

    Args:
        signal: Input signal
        target_db: Target peak level in dB (only used when reference_peak is None)
        reference_peak: Common peak value used to scale this signal. When set,
            the signal is divided by reference_peak (so a quieter signal stays
            quieter). The resulting signal's max amplitude is approximately
            target_linear (= 10^(target_db/20)).

    Returns:
        Normalized signal
    """
    if reference_peak is not None and reference_peak > 1e-10:
        # Scale relative to a common reference. Preserves relative loudness.
        target_linear = 10 ** (target_db / 20)
        return signal * (target_linear / reference_peak)

    # Independent normalization (each signal peaks at target_db) — DESTROYS
    # relative loudness. Only safe when comparison isn't intended.
    max_val = np.max(np.abs(signal))
    if max_val < 1e-10:
        return signal
    normalized = signal / max_val
    target_linear = 10 ** (target_db / 20)
    return normalized * target_linear


def signal_to_wav_bytes(signal: np.ndarray, sample_rate: int = 16000,
                        reference_peak: float = None) -> bytes:
    """
    Convert numpy signal to WAV file bytes for Streamlit audio player.

    Args:
        signal: Audio signal array
        sample_rate: Sample rate in Hz
        reference_peak: Optional common peak for relative-loudness normalization.

    Returns:
        WAV file as bytes
    """
    # Normalize to prevent clipping (uses common reference if given)
    audio = normalize_audio(signal, reference_peak=reference_peak)

    # Clip to [-1, 1] in case the cancelled signal exceeds ref_peak
    audio = np.clip(audio, -1.0, 1.0)

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
    Render audio players in Streamlit with download buttons.

    Important: BOTH the original and cancelled signals are normalized using the
    SAME reference peak (the larger of the two peaks). This preserves their
    relative loudness so the cancellation is actually audible. Independently
    normalizing each signal (the prior bug) would defeat the comparison —
    cancellation reduces peaks more than RMS, so a peak-normalized cancelled
    signal can sound LOUDER than the original despite real attenuation.

    Args:
        results: Simulation results dictionary
    """
    st.subheader("Audio Playback")

    fs = results['fs']
    desired = results['desired']
    error = results['error']

    # Common reference peak: the louder of the two signals.
    # Both audio renderings will use this so relative loudness is preserved.
    ref_peak = max(np.max(np.abs(desired)), np.max(np.abs(error)), 1e-10)

    # Sanity: report RMS of each so the user can see the cancellation in numbers
    rms_desired = float(np.sqrt(np.mean(desired ** 2)))
    rms_error = float(np.sqrt(np.mean(error ** 2)))
    rms_db_diff = 20 * np.log10(rms_desired / max(rms_error, 1e-12))

    # Pre-render WAV bytes for both playback and download
    original_bytes = signal_to_wav_bytes(desired, fs, reference_peak=ref_peak)
    cancelled_bytes = signal_to_wav_bytes(error, fs, reference_peak=ref_peak)
    comparison_bytes = create_comparison_audio(desired, error, fs)

    # Filename suffix derived from MIMO mode + scenario for clearer downloads
    mimo_mode = results.get('mimo_mode', 'Off')
    suffix_parts = []
    if mimo_mode and mimo_mode != 'Off':
        # "Stage 1 SIMO (1×M×1)" → "stage1"
        for tag, label in [('Stage 1', 'stage1'), ('Stage 2', 'stage2'),
                           ('Stage 3', 'stage3')]:
            if tag in mimo_mode:
                suffix_parts.append(label)
                break
    else:
        speaker_mode = results.get('speaker_mode', '')
        if '4-Speaker' in speaker_mode:
            suffix_parts.append('pseudo-simo')
        else:
            suffix_parts.append('siso')
    nr_db = results.get('noise_reduction_db', 0)
    suffix_parts.append(f"{nr_db:+.1f}dB".replace('+', 'p').replace('-', 'm'))
    suffix = '_'.join(suffix_parts) if suffix_parts else 'output'

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Original Noise**")
        st.caption(f"RMS: {20*np.log10(max(rms_desired, 1e-12)):.1f} dBFS")
        st.audio(original_bytes, format='audio/wav')
        st.download_button(
            label="⬇️ Download",
            data=original_bytes,
            file_name=f"original_{suffix}.wav",
            mime="audio/wav",
            key=f"download_original_{suffix}",
            use_container_width=True,
        )

    with col2:
        st.markdown("**With ANC**")
        st.caption(f"RMS: {20*np.log10(max(rms_error, 1e-12)):.1f} dBFS  "
                   f"(↓ {rms_db_diff:+.1f} dB vs original)")
        st.audio(cancelled_bytes, format='audio/wav')
        st.download_button(
            label="⬇️ Download",
            data=cancelled_bytes,
            file_name=f"cancelled_{suffix}.wav",
            mime="audio/wav",
            key=f"download_cancelled_{suffix}",
            use_container_width=True,
        )

    with col3:
        st.markdown("**Comparison**")
        st.caption("Original → Silence → With ANC")
        st.audio(comparison_bytes, format='audio/wav')
        st.download_button(
            label="⬇️ Download",
            data=comparison_bytes,
            file_name=f"comparison_{suffix}.wav",
            mime="audio/wav",
            key=f"download_comparison_{suffix}",
            use_container_width=True,
        )
