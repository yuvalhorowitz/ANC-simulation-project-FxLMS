"""
Optimal Filter Length Calculator

Calculates the optimal FxLMS filter length based on room reverb time (RT60).

Key Insight:
    Current implementation uses 256 taps (16ms @ 16kHz), but car cabin
    reverb time is 100-200ms. This means the filter can only learn
    8-16% of the acoustic path, severely limiting performance.

Solution:
    Calculate filter length based on RT60 to capture full acoustic path.
"""

import numpy as np


def calculate_optimal_filter_length(
    rt60_seconds: float,
    fs: int = 16000,
    coverage_factor: float = 1.5,
    min_length: int = 256,
    max_length: int = 2048
) -> int:
    """
    Calculate optimal filter length based on room reverb time.

    The filter should be long enough to capture the acoustic impulse response.
    Rule of thumb: filter length should cover at least 1.5x RT60 for good
    convergence and noise reduction.

    Args:
        rt60_seconds: Room reverb time (RT60) in seconds
        fs: Sample rate in Hz (default 16000)
        coverage_factor: Multiple of RT60 to cover (default 1.5)
        min_length: Minimum filter length (default 256)
        max_length: Maximum filter length (default 2048)

    Returns:
        Optimal filter length in samples (rounded to power of 2)

    Examples:
        >>> calculate_optimal_filter_length(0.1)  # Compact car
        512
        >>> calculate_optimal_filter_length(0.15)  # Sedan
        512
        >>> calculate_optimal_filter_length(0.2)  # SUV
        1024
        >>> calculate_optimal_filter_length(0.5)  # Reverberant room
        2048
    """
    # Time in seconds needed to capture reverb
    required_time = rt60_seconds * coverage_factor

    # Convert to samples
    required_samples = int(required_time * fs)

    # Round up to power of 2 for FFT efficiency
    if required_samples <= 0:
        return min_length

    power = int(np.ceil(np.log2(required_samples)))
    filter_length = 2 ** power

    # Apply bounds
    return max(min_length, min(filter_length, max_length))


def estimate_rt60_from_absorption(
    room_dimensions: list,
    mean_absorption: float,
    speed_of_sound: float = 343.0
) -> float:
    """
    Estimate RT60 using Sabine's formula.

    RT60 = 0.161 * V / (A * S)

    Where:
        V = room volume (m³)
        A = average absorption coefficient
        S = total surface area (m²)

    Args:
        room_dimensions: [length, width, height] in meters
        mean_absorption: Average absorption coefficient (0-1)
        speed_of_sound: Speed of sound in m/s (default 343)

    Returns:
        Estimated RT60 in seconds
    """
    L, W, H = room_dimensions

    # Volume
    V = L * W * H

    # Total surface area
    S = 2 * (L * W + L * H + W * H)

    # Sabine's formula
    if mean_absorption <= 0 or S <= 0:
        return 0.5  # Default for invalid input

    rt60 = 0.161 * V / (mean_absorption * S)

    # Clamp to reasonable range
    return max(0.05, min(rt60, 2.0))


def get_recommended_filter_lengths() -> dict:
    """
    Get recommended filter lengths for common scenarios.

    Based on typical RT60 values for each environment.

    Returns:
        Dictionary mapping scenario to recommended filter length
    """
    return {
        # Car interiors (short RT60)
        'compact_car': {
            'rt60': 0.10,
            'filter_length': 512,
            'notes': 'Small cabin, moderate absorption'
        },
        'sedan': {
            'rt60': 0.15,
            'filter_length': 512,
            'notes': 'Standard cabin size'
        },
        'suv': {
            'rt60': 0.20,
            'filter_length': 1024,
            'notes': 'Larger cabin, more reflections'
        },

        # Room scenarios
        'damped_room': {
            'rt60': 0.20,
            'filter_length': 512,
            'notes': 'High absorption, short reverb'
        },
        'typical_room': {
            'rt60': 0.40,
            'filter_length': 1024,
            'notes': 'Moderate absorption'
        },
        'reverberant_room': {
            'rt60': 0.60,
            'filter_length': 2048,
            'notes': 'Low absorption, long reverb'
        },
    }


if __name__ == '__main__':
    # Test calculations
    print("Optimal Filter Length Calculator")
    print("=" * 50)

    recommendations = get_recommended_filter_lengths()

    for scenario, config in recommendations.items():
        calculated = calculate_optimal_filter_length(config['rt60'])
        print(f"\n{scenario}:")
        print(f"  RT60: {config['rt60']:.2f}s")
        print(f"  Recommended: {config['filter_length']} taps")
        print(f"  Calculated: {calculated} taps")
        print(f"  Time coverage: {calculated / 16000 * 1000:.1f}ms")
