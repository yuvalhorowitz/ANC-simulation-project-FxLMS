"""
RIR (Room Impulse Response) Optimization

Provides energy-based RIR truncation to optimize computation while maintaining accuracy.

Key Insight:
    Current implementation uses hardcoded 512 samples for all scenarios.
    Energy-based truncation keeps only the significant part of the RIR,
    reducing computation without sacrificing acoustic accuracy.

Solution:
    Truncate RIR based on cumulative energy threshold (e.g., 99% of total energy).
"""

import numpy as np
from typing import Tuple, Optional


def optimal_rir_length(
    rir: np.ndarray,
    energy_threshold: float = 0.99,
    min_length: int = 256,
    max_length: int = 2048
) -> int:
    """
    Calculate optimal RIR length based on energy content.

    Keeps samples until energy_threshold (e.g., 99%) of total energy is captured.
    This allows shorter RIRs for well-damped rooms and longer for reverberant ones.

    Args:
        rir: Room impulse response array
        energy_threshold: Fraction of total energy to capture (default 0.99)
        min_length: Minimum RIR length (default 256)
        max_length: Maximum RIR length (default 2048)

    Returns:
        Optimal RIR length in samples
    """
    if len(rir) == 0:
        return min_length

    # Calculate cumulative energy
    energy = rir ** 2
    cumulative_energy = np.cumsum(energy)
    total_energy = cumulative_energy[-1]

    if total_energy == 0:
        return min_length

    # Find index where threshold is reached
    threshold_energy = energy_threshold * total_energy
    idx = np.searchsorted(cumulative_energy, threshold_energy)

    # Add small margin (10% extra samples)
    idx = int(idx * 1.1)

    # Apply bounds
    return max(min_length, min(idx + 1, max_length, len(rir)))


def truncate_rir(
    rir: np.ndarray,
    target_length: Optional[int] = None,
    energy_threshold: float = 0.99,
    min_length: int = 256,
    max_length: int = 2048,
    fade_out: bool = True,
    fade_samples: int = 32
) -> np.ndarray:
    """
    Truncate RIR to optimal length with optional fade-out.

    Args:
        rir: Room impulse response array
        target_length: Specific length to truncate to (None = auto-calculate)
        energy_threshold: For auto-calculation
        min_length: Minimum length
        max_length: Maximum length
        fade_out: Apply fade-out to avoid clicks (default True)
        fade_samples: Number of samples for fade-out window

    Returns:
        Truncated (and optionally faded) RIR
    """
    if target_length is None:
        target_length = optimal_rir_length(
            rir, energy_threshold, min_length, max_length
        )

    # Truncate
    truncated = rir[:target_length].copy()

    # Apply fade-out to avoid artifacts
    if fade_out and len(truncated) > fade_samples:
        fade_window = np.linspace(1.0, 0.0, fade_samples)
        truncated[-fade_samples:] *= fade_window

    return truncated


def analyze_rir_energy(
    rir: np.ndarray,
    fs: int = 16000
) -> dict:
    """
    Analyze RIR energy distribution.

    Args:
        rir: Room impulse response
        fs: Sample rate

    Returns:
        Dictionary with energy analysis metrics
    """
    if len(rir) == 0:
        return {'error': 'Empty RIR'}

    energy = rir ** 2
    cumulative = np.cumsum(energy)
    total_energy = cumulative[-1]

    if total_energy == 0:
        return {'error': 'Zero energy RIR'}

    # Find various thresholds
    thresholds = [0.90, 0.95, 0.99, 0.999]
    threshold_samples = {}
    threshold_times = {}

    for thresh in thresholds:
        idx = np.searchsorted(cumulative, thresh * total_energy)
        threshold_samples[f'{int(thresh*100)}%'] = idx
        threshold_times[f'{int(thresh*100)}%'] = idx / fs * 1000  # ms

    # Peak location
    peak_idx = np.argmax(np.abs(rir))
    peak_time = peak_idx / fs * 1000

    # Estimate RT60 (rough approximation)
    # Find where energy drops to -60 dB (1e-6 of peak)
    peak_energy = np.max(energy)
    threshold_60db = peak_energy * 1e-6
    below_threshold = np.where(energy[peak_idx:] < threshold_60db)[0]
    if len(below_threshold) > 0:
        rt60_samples = below_threshold[0]
        rt60_estimate = rt60_samples / fs
    else:
        rt60_estimate = len(rir) / fs

    return {
        'total_length_samples': len(rir),
        'total_length_ms': len(rir) / fs * 1000,
        'peak_index': peak_idx,
        'peak_time_ms': peak_time,
        'energy_thresholds_samples': threshold_samples,
        'energy_thresholds_ms': threshold_times,
        'estimated_rt60_s': rt60_estimate,
        'recommended_length': optimal_rir_length(rir),
    }


def get_optimal_truncation_params(scenario: str) -> dict:
    """
    Get recommended truncation parameters for common scenarios.

    Args:
        scenario: One of 'compact_car', 'sedan', 'suv', 'damped_room',
                  'typical_room', 'reverberant_room'

    Returns:
        Dictionary with recommended parameters
    """
    params = {
        # Car interiors (short RT60)
        'compact_car': {
            'energy_threshold': 0.99,
            'min_length': 256,
            'max_length': 512,
            'notes': 'Small cabin, fast decay'
        },
        'sedan': {
            'energy_threshold': 0.99,
            'min_length': 256,
            'max_length': 768,
            'notes': 'Standard cabin size'
        },
        'suv': {
            'energy_threshold': 0.99,
            'min_length': 256,
            'max_length': 1024,
            'notes': 'Larger cabin'
        },

        # Room scenarios
        'damped_room': {
            'energy_threshold': 0.99,
            'min_length': 256,
            'max_length': 512,
            'notes': 'High absorption'
        },
        'typical_room': {
            'energy_threshold': 0.99,
            'min_length': 512,
            'max_length': 1024,
            'notes': 'Moderate reverb'
        },
        'reverberant_room': {
            'energy_threshold': 0.995,
            'min_length': 512,
            'max_length': 2048,
            'notes': 'Long reverb tail'
        },
    }

    if scenario not in params:
        return params['sedan']  # Default

    return params[scenario]


def compare_rir_truncations(
    rir: np.ndarray,
    lengths: list = None,
    fs: int = 16000
) -> list:
    """
    Compare different truncation lengths.

    Args:
        rir: Original RIR
        lengths: List of lengths to compare
        fs: Sample rate

    Returns:
        List of dicts with comparison metrics
    """
    if lengths is None:
        lengths = [128, 256, 512, 768, 1024, 1536, 2048]

    # Filter to valid lengths
    lengths = [l for l in lengths if l <= len(rir)]

    total_energy = np.sum(rir ** 2)
    results = []

    for length in lengths:
        truncated = truncate_rir(rir, target_length=length, fade_out=False)
        trunc_energy = np.sum(truncated ** 2)

        energy_captured = trunc_energy / total_energy if total_energy > 0 else 0

        results.append({
            'length': length,
            'length_ms': length / fs * 1000,
            'energy_captured': energy_captured,
            'energy_lost_db': 10 * np.log10(1 - energy_captured + 1e-10),
            'is_recommended': length == optimal_rir_length(rir)
        })

    return results


if __name__ == '__main__':
    print("RIR Optimization Module")
    print("=" * 50)

    # Generate synthetic RIR for testing
    np.random.seed(42)
    fs = 16000
    rt60 = 0.15  # 150ms reverb (typical car)

    # Simple exponential decay model
    t = np.arange(int(rt60 * 3 * fs)) / fs
    decay = np.exp(-6.9 * t / rt60)  # -60dB at RT60
    noise = np.random.randn(len(t))
    synthetic_rir = noise * decay

    print(f"\nSynthetic RIR (RT60={rt60}s):")
    analysis = analyze_rir_energy(synthetic_rir, fs)
    for key, value in analysis.items():
        print(f"  {key}: {value}")

    print("\nTruncation comparison:")
    comparison = compare_rir_truncations(synthetic_rir)
    print(f"{'Length':>8} {'Time (ms)':>10} {'Energy':>10} {'Loss (dB)':>10}")
    print("-" * 42)
    for item in comparison:
        marker = " *" if item['is_recommended'] else ""
        print(f"{item['length']:>8} {item['length_ms']:>10.1f} "
              f"{item['energy_captured']:>10.4f} {item['energy_lost_db']:>10.2f}{marker}")
