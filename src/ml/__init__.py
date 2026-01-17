"""
Machine Learning Enhancements for FxLMS ANC

This module contains ML-based improvements to the FxLMS Active Noise Cancellation system.

Phases:
    - phase1_step_size: Adaptive step size selection using MLP
    - phase2_classifier: Noise type classification for parameter selection
    - phase3_neural: Neural network-based anti-noise generation
    - common: Shared utilities for metrics and comparison

Note: Imports are lazy to avoid requiring PyTorch for data collection scripts.
"""

# Only import common by default (no torch dependency)
from . import common

__all__ = [
    'common',
    'phase1_step_size',
    'phase2_classifier',
    'phase3_neural',
]


def __getattr__(name):
    """Lazy import for modules that require PyTorch."""
    if name == 'phase1_step_size':
        from . import phase1_step_size
        return phase1_step_size
    elif name == 'phase2_classifier':
        from . import phase2_classifier
        return phase2_classifier
    elif name == 'phase3_neural':
        from . import phase3_neural
        return phase3_neural
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
