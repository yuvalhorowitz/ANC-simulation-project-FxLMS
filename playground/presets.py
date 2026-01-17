"""
Presets for ANC Playground

Pre-configured room and scenario settings for quick testing.
"""

# Room presets with dimensions and default absorption
ROOM_PRESETS = {
    'Compact Car': {
        'dimensions': [3.8, 1.6, 1.1],
        'absorption': 0.3,
        'max_order': 3,
        'positions': {
            'noise_source': [0.4, 0.8, 0.35],
            'reference_mic': [0.9, 0.8, 0.7],
            'speaker': [3.0, 0.8, 0.5],
            'error_mic': [3.4, 0.6, 0.9],
        },
    },
    'Sedan': {
        'dimensions': [4.5, 1.85, 1.2],
        'absorption': 0.35,
        'max_order': 3,
        'positions': {
            'noise_source': [0.5, 0.92, 0.4],
            'reference_mic': [1.1, 0.92, 0.8],
            'speaker': [3.6, 0.92, 0.6],
            'error_mic': [4.0, 0.75, 1.0],
        },
    },
    'SUV': {
        'dimensions': [5.2, 2.1, 1.4],
        'absorption': 0.3,
        'max_order': 4,
        'positions': {
            'noise_source': [0.6, 1.05, 0.5],
            'reference_mic': [1.3, 1.05, 0.9],
            'speaker': [4.2, 1.05, 0.7],
            'error_mic': [4.7, 0.85, 1.1],
        },
    },
    'Small Room': {
        'dimensions': [4.0, 3.0, 2.5],
        'absorption': 0.3,
        'max_order': 3,
        'positions': {
            'noise_source': [0.5, 1.5, 1.0],
            'reference_mic': [1.2, 1.5, 1.0],
            'speaker': [2.8, 1.5, 1.2],
            'error_mic': [3.5, 1.5, 1.2],
        },
    },
    'Large Room': {
        'dimensions': [8.0, 6.0, 3.0],
        'absorption': 0.2,
        'max_order': 4,
        'positions': {
            'noise_source': [0.5, 3.0, 1.5],
            'reference_mic': [2.0, 3.0, 1.5],
            'speaker': [5.5, 3.0, 1.5],
            'error_mic': [7.0, 3.0, 1.5],
        },
    },
    'Anechoic': {
        'dimensions': [5.0, 4.0, 3.0],
        'absorption': 0.99,
        'max_order': 0,
        'positions': {
            'noise_source': [0.5, 2.0, 1.5],
            'reference_mic': [1.5, 2.0, 1.5],
            'speaker': [3.5, 2.0, 1.5],
            'error_mic': [4.5, 2.0, 1.5],
        },
    },
}

# Driving scenario presets
SCENARIO_PRESETS = {
    'Highway': {
        'rpm': 2800,
        'speed': 120,
        'engine_weight': 0.3,
        'road_weight': 0.4,
        'wind_weight': 0.3,
        'description': 'Highway cruising at 120 km/h',
    },
    'City': {
        'rpm': 2000,
        'speed': 50,
        'engine_weight': 0.5,
        'road_weight': 0.35,
        'wind_weight': 0.15,
        'description': 'City driving at 50 km/h',
    },
    'Acceleration': {
        'rpm': 4500,
        'speed': 80,
        'engine_weight': 0.7,
        'road_weight': 0.2,
        'wind_weight': 0.1,
        'description': 'Hard acceleration',
    },
    'Idle': {
        'rpm': 800,
        'speed': 0,
        'engine_weight': 0.9,
        'road_weight': 0.05,
        'wind_weight': 0.05,
        'description': 'Engine idling, stationary',
    },
    'Custom': {
        'rpm': 2000,
        'speed': 60,
        'engine_weight': 0.4,
        'road_weight': 0.3,
        'wind_weight': 0.3,
        'description': 'Custom settings',
    },
}

# Noise type presets (alternative to driving scenarios)
NOISE_PRESETS = {
    'Single Tone (100 Hz)': {
        'type': 'tonal',
        'frequencies': [100],
        'amplitudes': [1.0],
    },
    'Multi-frequency (50, 100, 150 Hz)': {
        'type': 'tonal',
        'frequencies': [50, 100, 150],
        'amplitudes': [0.5, 0.35, 0.15],
    },
    'Engine Harmonics (30-240 Hz)': {
        'type': 'tonal',
        'frequencies': [30, 60, 90, 120, 180, 240],
        'amplitudes': [0.3, 0.25, 0.2, 0.15, 0.1, 0.05],
    },
    'Broadband (20-300 Hz)': {
        'type': 'broadband',
        'low_freq': 20,
        'high_freq': 300,
    },
}

# FxLMS parameter presets
FXLMS_PRESETS = {
    'Conservative': {
        'filter_length': 256,
        'step_size': 0.001,
        'description': 'Slow but stable convergence',
    },
    'Balanced': {
        'filter_length': 256,
        'step_size': 0.005,
        'description': 'Good balance of speed and stability',
    },
    'Aggressive': {
        'filter_length': 256,
        'step_size': 0.02,
        'description': 'Fast convergence, may be unstable',
    },
    'Short Filter': {
        'filter_length': 64,
        'step_size': 0.01,
        'description': 'Faster computation, less accuracy',
    },
    'Long Filter': {
        'filter_length': 512,
        'step_size': 0.003,
        'description': 'Better for reverberant rooms',
    },
}

# Default values
DEFAULTS = {
    'room_preset': 'Sedan',
    'scenario_preset': 'Highway',
    'noise_mode': 'Driving Scenario',  # or 'Simple Tones'
    'fxlms_preset': 'Balanced',
    'duration': 5.0,
    'sample_rate': 16000,
}
