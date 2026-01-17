"""
Room and ANC Configurations for Steps 4-7

Each step has 3 different configurations to demonstrate ANC behavior
across various room sizes, noise types, and acoustic conditions.

All configurations are designed for the 20-300 Hz target frequency range.
"""

import numpy as np


# =============================================================================
# STEP 4: Ideal ANC with Known Paths
# =============================================================================
# Goal: Show perfect cancellation is possible with known acoustic paths

STEP4_CONFIGS = {
    'config_A': {
        'name': 'Small Office - Single Tone',
        'description': 'Compact office with 100 Hz HVAC hum',
        'room': {
            'dimensions': [4.0, 3.0, 2.5],  # Small office
            'absorption': 0.3,               # Moderate absorption (carpet, furniture)
            'max_order': 3,
        },
        'positions': {
            'noise_source': [0.5, 1.5, 1.0],     # HVAC vent location
            'reference_mic': [1.2, 1.5, 1.0],    # Near noise source
            'speaker': [2.8, 1.5, 1.2],          # Control speaker
            'error_mic': [3.5, 1.5, 1.2],        # Desk/listener position
        },
        'noise': {
            'type': 'tonal',
            'frequencies': [100],                # Single HVAC hum
            'amplitudes': [1.0],
        },
    },
    'config_B': {
        'name': 'Living Room - Multi-frequency',
        'description': 'Medium living room with traffic noise (multiple frequencies)',
        'room': {
            'dimensions': [6.0, 4.5, 2.8],  # Living room
            'absorption': 0.25,              # Some soft furnishings
            'max_order': 4,
        },
        'positions': {
            'noise_source': [0.3, 2.25, 1.0],    # Window (traffic noise)
            'reference_mic': [1.5, 2.25, 1.0],   # Near window
            'speaker': [4.0, 2.25, 1.2],         # Near seating
            'error_mic': [5.0, 2.25, 1.2],       # Couch position
        },
        'noise': {
            'type': 'tonal',
            'frequencies': [50, 80, 120],        # Traffic rumble harmonics
            'amplitudes': [0.5, 0.35, 0.15],
        },
    },
    'config_C': {
        'name': 'Industrial Space - Broadband',
        'description': 'Factory floor with machinery broadband noise',
        'room': {
            'dimensions': [10.0, 8.0, 4.0],  # Industrial space
            'absorption': 0.15,               # Hard surfaces (concrete, metal)
            'max_order': 5,
        },
        'positions': {
            'noise_source': [1.0, 4.0, 1.5],     # Machine location
            'reference_mic': [2.5, 4.0, 1.5],    # Near machine
            'speaker': [7.0, 4.0, 1.5],          # Control speaker array
            'error_mic': [8.5, 4.0, 1.5],        # Operator position
        },
        'noise': {
            'type': 'tonal',
            'frequencies': [30, 60, 90, 120, 180, 240],  # Machine harmonics
            'amplitudes': [0.3, 0.25, 0.2, 0.15, 0.1, 0.05],
        },
    },
}


# =============================================================================
# STEP 5: ANC with Latency Problem
# =============================================================================
# Goal: Demonstrate how processing delay affects cancellation

STEP5_CONFIGS = {
    'config_A': {
        'name': 'Tight Timing - Headphones',
        'description': 'Very short acoustic paths (ANC headphone scenario)',
        'room': {
            'dimensions': [3.0, 2.5, 2.2],  # Small room
            'absorption': 0.4,
            'max_order': 2,
        },
        'positions': {
            'noise_source': [0.3, 1.25, 1.1],    # External noise
            'reference_mic': [0.8, 1.25, 1.1],   # Outer mic
            'speaker': [1.5, 1.25, 1.1],         # Driver (close to ear)
            'error_mic': [1.6, 1.25, 1.1],       # Inner mic (at ear)
        },
        'noise': {
            'type': 'tonal',
            'frequencies': [100],
            'amplitudes': [1.0],
        },
        'test_latencies_ms': [0, 0.2, 0.5, 1.0, 2.0],  # Very tight timing
    },
    'config_B': {
        'name': 'Medium Timing - Desktop',
        'description': 'Desktop ANC system with moderate paths',
        'room': {
            'dimensions': [5.0, 4.0, 2.5],
            'absorption': 0.3,
            'max_order': 3,
        },
        'positions': {
            'noise_source': [0.5, 2.0, 1.2],
            'reference_mic': [1.5, 2.0, 1.2],
            'speaker': [3.5, 2.0, 1.2],
            'error_mic': [4.2, 2.0, 1.2],
        },
        'noise': {
            'type': 'tonal',
            'frequencies': [80, 160],
            'amplitudes': [0.7, 0.3],
        },
        'test_latencies_ms': [0, 0.5, 1.0, 2.0, 5.0],
    },
    'config_C': {
        'name': 'Relaxed Timing - Large Room',
        'description': 'Large room with long acoustic paths',
        'room': {
            'dimensions': [8.0, 6.0, 3.0],
            'absorption': 0.2,
            'max_order': 4,
        },
        'positions': {
            'noise_source': [0.5, 3.0, 1.5],
            'reference_mic': [2.0, 3.0, 1.5],
            'speaker': [5.5, 3.0, 1.5],
            'error_mic': [7.0, 3.0, 1.5],
        },
        'noise': {
            'type': 'tonal',
            'frequencies': [60, 120, 180],
            'amplitudes': [0.5, 0.35, 0.15],
        },
        'test_latencies_ms': [0, 1.0, 2.0, 5.0, 10.0],  # More relaxed
    },
}


# =============================================================================
# STEP 6: FxLMS with Realistic Acoustic Paths
# =============================================================================
# Goal: Demonstrate adaptive algorithm performance in different conditions

STEP6_CONFIGS = {
    'config_A': {
        'name': 'Reverberant Room',
        'description': 'High reflection room - challenging for adaptation',
        'room': {
            'dimensions': [5.0, 4.0, 3.0],
            'absorption': 0.1,               # Low absorption = lots of reverb
            'max_order': 6,                  # Many reflections
        },
        'positions': {
            'noise_source': [0.5, 2.0, 1.5],
            'reference_mic': [1.5, 2.0, 1.5],
            'speaker': [3.5, 2.0, 1.5],
            'error_mic': [4.5, 2.0, 1.5],
        },
        'noise': {
            'type': 'tonal',
            'frequencies': [100],
            'amplitudes': [1.0],
        },
        'fxlms': {
            'filter_length': 256,            # Longer filter for reverb
            'step_size': 0.001,              # Small step size for stability in reverberant room
            'duration': 5.0,
        },
    },
    'config_B': {
        'name': 'Typical Room',
        'description': 'Moderate absorption - standard conditions',
        'room': {
            'dimensions': [5.0, 4.0, 2.5],
            'absorption': 0.3,
            'max_order': 3,
        },
        'positions': {
            'noise_source': [0.5, 2.0, 1.2],
            'reference_mic': [1.5, 2.0, 1.2],
            'speaker': [3.5, 2.0, 1.2],
            'error_mic': [4.2, 2.0, 1.2],
        },
        'noise': {
            'type': 'tonal',
            'frequencies': [50, 100, 150],    # Multi-frequency
            'amplitudes': [0.5, 0.35, 0.15],
        },
        'fxlms': {
            'filter_length': 256,            # Longer filter for multi-frequency
            'step_size': 0.003,              # Smaller step to prevent divergence
            'duration': 8.0,                 # Longer duration for convergence
        },
    },
    'config_C': {
        'name': 'Damped Room',
        'description': 'High absorption - easier adaptation',
        'room': {
            'dimensions': [4.0, 3.5, 2.5],
            'absorption': 0.5,               # High absorption
            'max_order': 2,                  # Few reflections
        },
        'positions': {
            'noise_source': [0.4, 1.75, 1.2],
            'reference_mic': [1.2, 1.75, 1.2],
            'speaker': [2.8, 1.75, 1.2],
            'error_mic': [3.5, 1.75, 1.2],
        },
        'noise': {
            'type': 'tonal',
            'frequencies': [80, 120, 200, 280],  # Wider frequency range
            'amplitudes': [0.4, 0.3, 0.2, 0.1],
        },
        'fxlms': {
            'filter_length': 256,             # Longer filter for multi-frequency
            'step_size': 0.02,                # Larger step - damped room is more stable
            'duration': 10.0,                 # Longer duration for multi-freq convergence
        },
    },
}


# =============================================================================
# STEP 7: Car Interior ANC
# =============================================================================
# Goal: Realistic car cabin noise cancellation scenarios

STEP7_CONFIGS = {
    'config_A': {
        'name': 'Compact Car',
        'description': 'Small hatchback - engine-dominant noise',
        'room': {
            'dimensions': [3.8, 1.6, 1.1],   # Compact car cabin
            'materials': {
                'ceiling': 0.35,              # Headliner
                'floor': 0.55,                # Carpet
                'east': 0.12,                 # Windows
                'west': 0.12,
                'north': 0.18,                # Dashboard
                'south': 0.28,                # Rear seats
            },
            'max_order': 3,
        },
        'positions': {
            'noise_source': [0.4, 0.8, 0.35],     # Engine/firewall
            'reference_mic': [0.9, 0.8, 0.7],     # Near firewall
            'speaker': [3.0, 0.8, 0.5],           # Headrest speaker
            'error_mic': [3.4, 0.6, 0.9],         # Driver's ear
        },
        'scenario': 'highway',
        'fxlms': {
            'filter_length': 256,
            'step_size': 0.005,              # Small step size for car acoustics
            'duration': 5.0,
        },
    },
    'config_B': {
        'name': 'Sedan',
        'description': 'Mid-size sedan - balanced noise sources',
        'room': {
            'dimensions': [4.5, 1.85, 1.2],  # Sedan cabin
            'materials': {
                'ceiling': 0.38,
                'floor': 0.52,
                'east': 0.14,
                'west': 0.14,
                'north': 0.20,
                'south': 0.30,
            },
            'max_order': 3,
        },
        'positions': {
            'noise_source': [0.5, 0.92, 0.4],
            'reference_mic': [1.1, 0.92, 0.8],
            'speaker': [3.6, 0.92, 0.6],
            'error_mic': [4.0, 0.75, 1.0],
        },
        'scenario': 'city',
        'fxlms': {
            'filter_length': 256,
            'step_size': 0.005,              # Small step size for car acoustics
            'duration': 5.0,
        },
    },
    'config_C': {
        'name': 'SUV',
        'description': 'Large SUV - more cabin volume, road noise dominant',
        'room': {
            'dimensions': [5.2, 2.1, 1.4],   # SUV cabin
            'materials': {
                'ceiling': 0.32,
                'floor': 0.48,
                'east': 0.15,
                'west': 0.15,
                'north': 0.22,
                'south': 0.35,
            },
            'max_order': 4,
        },
        'positions': {
            'noise_source': [0.6, 1.05, 0.5],
            'reference_mic': [1.3, 1.05, 0.9],
            'speaker': [4.2, 1.05, 0.7],
            'error_mic': [4.7, 0.85, 1.1],
        },
        'scenario': 'acceleration',
        'fxlms': {
            'filter_length': 320,
            'step_size': 0.003,              # Smaller step for larger cabin
            'duration': 6.0,
        },
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

def generate_noise_signal(config: dict, duration: float, fs: int) -> np.ndarray:
    """
    Generate noise signal based on configuration.

    Args:
        config: Noise configuration dict with 'type', 'frequencies', 'amplitudes'
        duration: Signal duration in seconds
        fs: Sampling frequency

    Returns:
        Noise signal array
    """
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs

    noise_type = config.get('type', 'tonal')

    if noise_type == 'tonal':
        # Sum of sinusoids
        signal = np.zeros(n_samples)
        frequencies = config.get('frequencies', [100])
        amplitudes = config.get('amplitudes', [1.0])

        for freq, amp in zip(frequencies, amplitudes):
            signal += amp * np.sin(2 * np.pi * freq * t)

        # Normalize
        signal = signal / np.max(np.abs(signal)) if np.max(np.abs(signal)) > 0 else signal

    elif noise_type == 'broadband':
        # Filtered white noise in 20-300 Hz range
        from scipy import signal as scipy_signal

        noise = np.random.randn(n_samples)

        # Bandpass filter 20-300 Hz
        nyquist = fs / 2
        low = 20 / nyquist
        high = min(300, nyquist * 0.9) / nyquist

        b, a = scipy_signal.butter(4, [low, high], btype='band')
        signal = scipy_signal.filtfilt(b, a, noise)
        signal = signal / np.max(np.abs(signal))

    else:
        signal = np.sin(2 * np.pi * 100 * t)  # Default to 100 Hz

    return signal


def print_config_summary(config: dict, step_name: str):
    """Print a summary of the configuration."""
    print(f"\n{'='*60}")
    print(f"{step_name}: {config['name']}")
    print(f"{'='*60}")
    print(f"Description: {config['description']}")

    room = config['room']
    dims = room['dimensions']
    print(f"\nRoom: {dims[0]}m x {dims[1]}m x {dims[2]}m")

    if 'absorption' in room:
        print(f"Absorption: {room['absorption']}")
    if 'materials' in room:
        print("Materials: Custom per surface")

    print(f"Max reflection order: {room['max_order']}")

    pos = config['positions']
    print(f"\nPositions:")
    for name, p in pos.items():
        print(f"  {name}: [{p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}]")

    if 'noise' in config:
        noise = config['noise']
        print(f"\nNoise: {noise['type']}")
        if noise['type'] == 'tonal':
            freqs = noise.get('frequencies', [])
            print(f"  Frequencies: {freqs} Hz")
