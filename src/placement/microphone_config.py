"""
Microphone and Speaker Placement Configuration for Car ANC

Defines positions for:
- Car stereo speaker locations (potential anti-noise sources)
- Reference microphone candidates (noise sensing)
- Error microphone candidates (cancellation measurement)

All positions are in meters relative to car cabin coordinate system:
- X: Front-to-back (0 = front firewall, max = rear)
- Y: Left-to-right (0 = driver side, max = passenger side)
- Z: Floor-to-ceiling (0 = floor, max = roof)

Based on typical sedan dimensions: 4.5m x 1.85m x 1.2m
"""

from typing import Dict, List, Tuple
import itertools


# =============================================================================
# Sedan Car Cabin Configuration
# =============================================================================

SEDAN_DIMENSIONS = [4.5, 1.85, 1.2]  # Length, Width, Height in meters

SEDAN_MATERIALS = {
    'ceiling': 0.38,   # Headliner (fabric)
    'floor': 0.52,     # Carpet
    'east': 0.14,      # Windows (driver side)
    'west': 0.14,      # Windows (passenger side)
    'north': 0.20,     # Dashboard/firewall
    'south': 0.30,     # Rear seats/trunk
}


# =============================================================================
# Car Stereo Speaker Positions
# =============================================================================
# These are the existing speakers in a typical car audio system
# Each can potentially be used to produce anti-noise

SPEAKER_POSITIONS: Dict[str, List[float]] = {
    # Front door speakers (low, in door panel)
    'door_left': [2.0, 0.1, 0.4],       # Driver door, low position
    'door_right': [2.0, 1.75, 0.4],     # Passenger door, low position

    # Dashboard/A-pillar tweeters (high, near windshield)
    'dash_left': [0.8, 0.25, 0.9],      # A-pillar driver side
    'dash_right': [0.8, 1.60, 0.9],     # A-pillar passenger side
    'dash_center': [0.8, 0.92, 0.85],   # Center stack speaker

    # Rear deck speakers (behind rear seats)
    'rear_left': [4.0, 0.40, 0.9],      # Rear deck left
    'rear_right': [4.0, 1.45, 0.9],     # Rear deck right

    # Headrest speakers (premium systems)
    'headrest_driver': [3.2, 0.55, 1.0],     # Driver headrest
    'headrest_passenger': [3.2, 1.30, 1.0],  # Passenger headrest
}


# =============================================================================
# Reference Microphone Positions
# =============================================================================
# Reference mic should be UPSTREAM of the noise to give processing time

REF_MIC_POSITIONS: Dict[str, List[float]] = {
    # Firewall area (best for engine noise)
    'firewall_center': [0.3, 0.92, 0.5],     # Center firewall
    'firewall_driver': [0.3, 0.45, 0.5],     # Driver side firewall

    # A-pillar (good for road/wind noise)
    'a_pillar_left': [0.7, 0.15, 1.0],       # Left A-pillar
    'a_pillar_right': [0.7, 1.70, 1.0],      # Right A-pillar

    # Dashboard (central noise pickup)
    'dashboard': [0.9, 0.92, 0.8],           # Dashboard center

    # Under seat (road noise dominant area)
    'under_driver_seat': [2.5, 0.55, 0.15],  # Under driver seat
    'under_passenger_seat': [2.5, 1.30, 0.15], # Under passenger seat

    # Floor near wheel wells (tire/road noise)
    'floor_front_left': [1.0, 0.15, 0.1],    # Near front left wheel
    'floor_front_right': [1.0, 1.70, 0.1],   # Near front right wheel
}


# =============================================================================
# Error Microphone Positions
# =============================================================================
# Error mic measures cancellation performance at listener position

ERROR_MIC_POSITIONS: Dict[str, List[float]] = {
    # Driver positions (primary targets)
    'driver_headrest': [3.2, 0.55, 1.0],     # Driver headrest
    'driver_ear_left': [3.2, 0.40, 1.0],     # Driver left ear
    'driver_ear_right': [3.2, 0.70, 1.0],    # Driver right ear

    # Passenger positions
    'passenger_headrest': [3.2, 1.30, 1.0],  # Passenger headrest
    'passenger_ear_left': [3.2, 1.15, 1.0],  # Passenger left ear
    'passenger_ear_right': [3.2, 1.45, 1.0], # Passenger right ear

    # Alternative mounting locations
    'sun_visor_driver': [2.8, 0.55, 1.15],   # Driver sun visor
    'sun_visor_passenger': [2.8, 1.30, 1.15], # Passenger sun visor
    'rearview_mirror': [1.5, 0.92, 1.1],     # Central rearview mirror
}


# =============================================================================
# Noise Source Positions (for simulation)
# =============================================================================
# These represent where noise originates in the car

NOISE_SOURCE_POSITIONS: Dict[str, List[float]] = {
    'engine': [0.3, 0.92, 0.4],          # Engine (firewall area)
    'front_left_wheel': [0.8, 0.1, 0.2], # Front left wheel well
    'front_right_wheel': [0.8, 1.75, 0.2], # Front right wheel well
    'rear_left_wheel': [3.7, 0.1, 0.2],  # Rear left wheel well
    'rear_right_wheel': [3.7, 1.75, 0.2], # Rear right wheel well
    'windshield': [0.6, 0.92, 1.0],      # Wind noise (windshield)
}


# =============================================================================
# Driving Scenarios
# =============================================================================

DRIVING_SCENARIOS = {
    'highway': {
        'description': 'Highway cruising at 120 km/h',
        'noise_mix': {'engine': 0.3, 'road': 0.4, 'wind': 0.3},
        'rpm': 2800,
        'speed_kmh': 120,
    },
    'city': {
        'description': 'City driving at 50 km/h',
        'noise_mix': {'engine': 0.6, 'road': 0.3, 'wind': 0.1},
        'rpm': 2000,
        'speed_kmh': 50,
    },
    'acceleration': {
        'description': 'Hard acceleration',
        'noise_mix': {'engine': 0.7, 'road': 0.2, 'wind': 0.1},
        'rpm': 4500,
        'speed_kmh': 80,
    },
    'idle': {
        'description': 'Engine idle in traffic',
        'noise_mix': {'engine': 0.9, 'road': 0.05, 'wind': 0.05},
        'rpm': 800,
        'speed_kmh': 0,
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_placement_config(
    speaker: str,
    ref_mic: str,
    error_mic: str,
    scenario: str = 'highway'
) -> dict:
    """
    Build a complete ANC configuration for a specific placement combination.

    Args:
        speaker: Key from SPEAKER_POSITIONS
        ref_mic: Key from REF_MIC_POSITIONS
        error_mic: Key from ERROR_MIC_POSITIONS
        scenario: Key from DRIVING_SCENARIOS

    Returns:
        Configuration dictionary compatible with CarInteriorANC
    """
    if speaker not in SPEAKER_POSITIONS:
        raise ValueError(f"Unknown speaker: {speaker}. Options: {list(SPEAKER_POSITIONS.keys())}")
    if ref_mic not in REF_MIC_POSITIONS:
        raise ValueError(f"Unknown ref_mic: {ref_mic}. Options: {list(REF_MIC_POSITIONS.keys())}")
    if error_mic not in ERROR_MIC_POSITIONS:
        raise ValueError(f"Unknown error_mic: {error_mic}. Options: {list(ERROR_MIC_POSITIONS.keys())}")
    if scenario not in DRIVING_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}. Options: {list(DRIVING_SCENARIOS.keys())}")

    config = {
        'name': f"{speaker}_{ref_mic}_{error_mic}",
        'description': f"Speaker: {speaker}, Ref: {ref_mic}, Error: {error_mic}",
        'room': {
            'dimensions': SEDAN_DIMENSIONS.copy(),
            'materials': SEDAN_MATERIALS.copy(),
            'max_order': 3,
        },
        'positions': {
            'noise_source': [0.3, 0.92, 0.4],  # Engine position (default)
            'reference_mic': REF_MIC_POSITIONS[ref_mic].copy(),
            'speaker': SPEAKER_POSITIONS[speaker].copy(),
            'error_mic': ERROR_MIC_POSITIONS[error_mic].copy(),
        },
        'scenario': scenario,
        'fxlms': {
            'filter_length': 256,
            'step_size': 0.005,
            'duration': 5.0,
        },
        'placement_info': {
            'speaker_name': speaker,
            'ref_mic_name': ref_mic,
            'error_mic_name': error_mic,
            'scenario_name': scenario,
        }
    }

    return config


def get_all_placement_combinations(
    speakers: List[str] = None,
    ref_mics: List[str] = None,
    error_mics: List[str] = None,
    scenarios: List[str] = None,
) -> List[Tuple[str, str, str, str]]:
    """
    Generate all combinations of placements to test.

    Args:
        speakers: List of speaker keys (default: all)
        ref_mics: List of ref mic keys (default: all)
        error_mics: List of error mic keys (default: all)
        scenarios: List of scenario keys (default: all)

    Returns:
        List of (speaker, ref_mic, error_mic, scenario) tuples
    """
    if speakers is None:
        speakers = list(SPEAKER_POSITIONS.keys())
    if ref_mics is None:
        ref_mics = list(REF_MIC_POSITIONS.keys())
    if error_mics is None:
        error_mics = list(ERROR_MIC_POSITIONS.keys())
    if scenarios is None:
        scenarios = list(DRIVING_SCENARIOS.keys())

    return list(itertools.product(speakers, ref_mics, error_mics, scenarios))


def get_subset_combinations(
    n_speakers: int = 4,
    n_ref_mics: int = 3,
    n_error_mics: int = 2,
    scenarios: List[str] = None,
) -> List[Tuple[str, str, str, str]]:
    """
    Get a reduced subset of combinations for faster testing.

    Selects representative positions to reduce test count while
    maintaining coverage of different placement strategies.

    Args:
        n_speakers: Number of speakers to test
        n_ref_mics: Number of reference mics to test
        n_error_mics: Number of error mics to test
        scenarios: Scenarios to test (default: ['highway', 'city'])

    Returns:
        List of (speaker, ref_mic, error_mic, scenario) tuples
    """
    # Priority order for speakers (most likely to be effective for ANC)
    speaker_priority = [
        'headrest_driver',    # Closest to ear
        'door_left',          # Door speaker
        'dash_left',          # Dashboard driver side
        'dash_center',        # Center
        'rear_left',          # Rear
        'door_right',
        'dash_right',
        'rear_right',
        'headrest_passenger',
    ]

    # Priority order for reference mics (best for early noise detection)
    ref_mic_priority = [
        'firewall_center',    # Engine noise
        'dashboard',          # Central
        'a_pillar_left',      # Wind/road
        'under_driver_seat',  # Road noise
        'floor_front_left',   # Tire noise
        'firewall_driver',
        'a_pillar_right',
        'under_passenger_seat',
        'floor_front_right',
    ]

    # Priority order for error mics (primary listening positions)
    error_mic_priority = [
        'driver_headrest',    # Primary target
        'driver_ear_left',    # Left ear
        'driver_ear_right',   # Right ear
        'passenger_headrest', # Passenger
        'passenger_ear_left',
        'passenger_ear_right',
        'sun_visor_driver',
        'rearview_mirror',
        'sun_visor_passenger',
    ]

    if scenarios is None:
        scenarios = ['highway', 'city']

    # Select top N from each category
    selected_speakers = [s for s in speaker_priority if s in SPEAKER_POSITIONS][:n_speakers]
    selected_ref_mics = [r for r in ref_mic_priority if r in REF_MIC_POSITIONS][:n_ref_mics]
    selected_error_mics = [e for e in error_mic_priority if e in ERROR_MIC_POSITIONS][:n_error_mics]

    return list(itertools.product(
        selected_speakers,
        selected_ref_mics,
        selected_error_mics,
        scenarios
    ))


def calculate_distances(config: dict) -> dict:
    """
    Calculate acoustic distances for a configuration.

    Returns distances that affect ANC timing:
    - Primary path: noise_source -> error_mic
    - Secondary path: speaker -> error_mic
    - Reference path: noise_source -> reference_mic
    - Time budget: (primary - reference) in samples
    """
    import numpy as np

    pos = config['positions']
    fs = 16000  # Sample rate
    c = 343     # Speed of sound m/s

    def dist(p1, p2):
        return np.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))

    d_primary = dist(pos['noise_source'], pos['error_mic'])
    d_secondary = dist(pos['speaker'], pos['error_mic'])
    d_reference = dist(pos['noise_source'], pos['reference_mic'])

    # Time budget: how much time we have to compute anti-noise
    # = (time for noise to reach error mic) - (time to reach ref mic) - (secondary path time)
    t_to_error = d_primary / c
    t_to_ref = d_reference / c
    t_secondary = d_secondary / c

    time_budget = t_to_error - t_to_ref - t_secondary

    return {
        'primary_distance_m': d_primary,
        'secondary_distance_m': d_secondary,
        'reference_distance_m': d_reference,
        'primary_delay_ms': d_primary / c * 1000,
        'secondary_delay_ms': d_secondary / c * 1000,
        'reference_delay_ms': d_reference / c * 1000,
        'time_budget_ms': time_budget * 1000,
        'time_budget_samples': int(time_budget * fs),
        'is_feasible': time_budget > 0,
    }


def validate_placement(config: dict) -> Tuple[bool, str]:
    """
    Validate that a placement configuration is physically feasible.

    Checks:
    1. All positions within room bounds
    2. Reference mic is closer to noise than error mic
    3. Positive time budget exists

    Returns:
        (is_valid, message)
    """
    dims = config['room']['dimensions']
    pos = config['positions']

    # Check positions within bounds
    margin = 0.05  # 5cm minimum from walls
    for name, p in pos.items():
        if not (margin <= p[0] <= dims[0] - margin and
                margin <= p[1] <= dims[1] - margin and
                margin <= p[2] <= dims[2] - margin):
            return False, f"{name} position {p} is outside room bounds"

    # Calculate distances
    distances = calculate_distances(config)

    # Check time budget
    if not distances['is_feasible']:
        return False, f"Negative time budget: {distances['time_budget_ms']:.2f}ms - anti-noise will arrive late"

    if distances['time_budget_ms'] < 0.5:
        return True, f"Warning: Very tight time budget ({distances['time_budget_ms']:.2f}ms)"

    return True, f"Valid placement with {distances['time_budget_ms']:.2f}ms time budget"
