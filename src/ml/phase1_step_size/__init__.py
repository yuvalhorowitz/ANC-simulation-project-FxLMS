"""
Phase 1: Adaptive Step Size Selection

This phase replaces the fixed step size (μ) with an ML-based selector
that predicts the optimal step size from signal features.

Note: Imports are lazy to allow data collection without PyTorch.
"""

# Only import feature_extractor by default (no torch dependency)
from .feature_extractor import extract_features

__all__ = [
    'extract_features',
    'StepSizeSelector',
    'StepSizeSelectorTrainer',
    'AdaptiveFxNLMS',
]


def __getattr__(name):
    """Lazy import for classes that require PyTorch."""
    if name == 'StepSizeSelector':
        from .step_size_selector import StepSizeSelector
        return StepSizeSelector
    elif name == 'StepSizeSelectorTrainer':
        from .step_size_selector import StepSizeSelectorTrainer
        return StepSizeSelectorTrainer
    elif name == 'AdaptiveFxNLMS':
        from .adaptive_fxlms import AdaptiveFxNLMS
        return AdaptiveFxNLMS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
