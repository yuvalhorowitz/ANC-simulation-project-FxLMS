"""
Optimized FxLMS Parameter Tuning Module

This module provides tools for optimizing FxLMS parameters and pyroomacoustics
configuration WITHOUT changing the core algorithm.

Modules:
    optimal_filter_length: Calculate optimal filter length based on RT60
    optimal_max_order: Calculate optimal reflection order for pyroomacoustics
    low_freq_materials: Frequency-dependent material absorption coefficients
    rir_optimization: Energy-based RIR truncation
    optimized_configurations: Pre-tuned configurations for all scenarios
    run_optimized_simulation: Main entry point for running optimized simulations
    compare_results: Tools for comparing baseline vs optimized results

Speaker Configurations:
    Each scenario supports two speaker configurations:
    - SINGLE: Single headrest speaker (baseline)
    - QUAD_STEREO: 4-speaker stereo (front left, front right, rear left, rear right)
"""

from .optimal_filter_length import calculate_optimal_filter_length
from .optimal_max_order import calculate_optimal_max_order
from .rir_optimization import optimal_rir_length, truncate_rir
from .low_freq_materials import (
    get_material_absorption,
    get_low_freq_absorption,
    create_pra_material,
    get_car_interior_materials,
    calculate_mean_absorption,
)
from .optimized_configurations import (
    get_config,
    get_all_configs,
    SpeakerConfig,
    OptimizedConfig,
    get_speaker_positions,
)

__all__ = [
    # Filter/order optimization
    'calculate_optimal_filter_length',
    'calculate_optimal_max_order',

    # RIR optimization
    'optimal_rir_length',
    'truncate_rir',

    # Materials
    'get_material_absorption',
    'get_low_freq_absorption',
    'create_pra_material',
    'get_car_interior_materials',
    'calculate_mean_absorption',

    # Configurations
    'get_config',
    'get_all_configs',
    'SpeakerConfig',
    'OptimizedConfig',
    'get_speaker_positions',
]
