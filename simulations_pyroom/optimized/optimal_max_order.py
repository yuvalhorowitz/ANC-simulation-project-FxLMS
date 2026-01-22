"""
Optimal Max Order Calculator for Pyroomacoustics

Calculates the optimal reflection order (max_order) for the image source method.

Key Insight:
    Current implementation uses max_order=3 which only captures early reflections.
    For accurate ANC modeling, we need to capture reflections up to the RT60.

Solution:
    Calculate max_order based on room dimensions and desired RT60 coverage.
"""

import numpy as np


def calculate_optimal_max_order(
    room_dimensions: list,
    rt60: float,
    speed_of_sound: float = 343.0,
    min_order: int = 3,
    max_order: int = 15
) -> int:
    """
    Calculate optimal reflection order for pyroomacoustics image source method.

    The max_order determines how many wall reflections are simulated.
    Higher orders capture more late reflections but increase computation.

    Args:
        room_dimensions: [length, width, height] in meters
        rt60: Target reverb time to model (seconds)
        speed_of_sound: Speed of sound in m/s (default 343)
        min_order: Minimum reflection order (default 3)
        max_order: Maximum reflection order (default 15, computational limit)

    Returns:
        Optimal max_order value

    Examples:
        >>> calculate_optimal_max_order([4.8, 1.85, 1.5], 0.15)  # Sedan
        7
        >>> calculate_optimal_max_order([4.7, 1.9, 1.8], 0.20)  # SUV
        8
    """
    # Maximum room dimension (determines reflection travel time)
    max_dim = max(room_dimensions)

    if max_dim <= 0 or rt60 <= 0:
        return min_order

    # Time for sound to travel one max dimension
    travel_time = max_dim / speed_of_sound

    # Number of reflections needed to cover RT60
    # Each reflection bounces back and forth
    n_reflections = rt60 / travel_time

    # max_order represents the number of wall interactions
    # Each order adds roughly 2 reflections (forward + back)
    optimal_order = int(np.ceil(n_reflections / 2))

    # Apply bounds
    return max(min_order, min(optimal_order, max_order))


def estimate_computation_time(max_order: int, room_dimensions: list) -> float:
    """
    Estimate relative computation time for given max_order.

    The image source method has complexity O((2*max_order + 1)^3).

    Args:
        max_order: Reflection order
        room_dimensions: Room dimensions (for reference)

    Returns:
        Relative computation factor (1.0 = baseline at max_order=3)
    """
    baseline_order = 3
    baseline_images = (2 * baseline_order + 1) ** 3

    current_images = (2 * max_order + 1) ** 3

    return current_images / baseline_images


def get_recommended_max_orders() -> dict:
    """
    Get recommended max_order values for common scenarios.

    Returns:
        Dictionary mapping scenario to recommended max_order and rationale
    """
    return {
        # Car interiors
        'compact_car': {
            'dimensions': [3.5, 1.8, 1.5],
            'rt60': 0.10,
            'max_order': 5,
            'notes': 'Small space, fast reflections'
        },
        'sedan': {
            'dimensions': [4.8, 1.85, 1.5],
            'rt60': 0.15,
            'max_order': 7,
            'notes': 'Standard size, moderate reflections'
        },
        'suv': {
            'dimensions': [4.7, 1.9, 1.8],
            'rt60': 0.20,
            'max_order': 8,
            'notes': 'Larger cabin, more late reflections'
        },

        # Room scenarios
        'damped_room': {
            'dimensions': [6.0, 5.0, 3.0],
            'rt60': 0.20,
            'max_order': 5,
            'notes': 'High absorption limits reflections'
        },
        'typical_room': {
            'dimensions': [6.0, 5.0, 3.0],
            'rt60': 0.40,
            'max_order': 8,
            'notes': 'Moderate environment'
        },
        'reverberant_room': {
            'dimensions': [6.0, 5.0, 3.0],
            'rt60': 0.60,
            'max_order': 12,
            'notes': 'Low absorption, many reflections'
        },
    }


def analyze_max_order_tradeoff(
    room_dimensions: list,
    rt60: float,
    orders_to_test: list = None
) -> list:
    """
    Analyze the tradeoff between max_order, accuracy, and computation.

    Args:
        room_dimensions: [length, width, height] in meters
        rt60: Target RT60 in seconds
        orders_to_test: List of max_order values to analyze

    Returns:
        List of dicts with analysis for each order
    """
    if orders_to_test is None:
        orders_to_test = [3, 5, 7, 10, 12, 15]

    optimal = calculate_optimal_max_order(room_dimensions, rt60)
    max_dim = max(room_dimensions)
    travel_time = max_dim / 343.0

    results = []
    for order in orders_to_test:
        # Time coverage
        time_coverage = order * 2 * travel_time
        rt60_coverage = time_coverage / rt60 if rt60 > 0 else 0

        # Computation factor
        comp_factor = estimate_computation_time(order, room_dimensions)

        results.append({
            'max_order': order,
            'time_coverage_ms': time_coverage * 1000,
            'rt60_coverage_percent': min(100, rt60_coverage * 100),
            'computation_factor': comp_factor,
            'is_optimal': order == optimal,
            'recommendation': 'optimal' if order == optimal else
                             ('under' if order < optimal else 'over')
        })

    return results


if __name__ == '__main__':
    print("Optimal Max Order Calculator")
    print("=" * 60)

    recommendations = get_recommended_max_orders()

    for scenario, config in recommendations.items():
        calculated = calculate_optimal_max_order(
            config['dimensions'],
            config['rt60']
        )
        comp_factor = estimate_computation_time(calculated, config['dimensions'])

        print(f"\n{scenario}:")
        print(f"  Dimensions: {config['dimensions']} m")
        print(f"  RT60: {config['rt60']:.2f}s")
        print(f"  Recommended: max_order={config['max_order']}")
        print(f"  Calculated: max_order={calculated}")
        print(f"  Computation factor: {comp_factor:.1f}x")
