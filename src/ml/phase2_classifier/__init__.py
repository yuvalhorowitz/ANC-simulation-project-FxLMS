"""
Phase 2: Noise Classification

Classifies noise type and selects optimal FxNLMS parameters.
"""

from .spectrogram import (
    compute_mel_spectrogram,
    extract_mel_spectrogram,
    normalize_spectrogram,
    spectrogram_to_tensor,
)

from .noise_classifier import (
    NoiseClassifier,
    NoiseClassifierTrainer,
)

from .parameter_lookup import (
    FxNLMSParams,
    OPTIMAL_PARAMS,
    DEFAULT_PARAMS,
    get_params,
    get_params_dict,
    get_step_size,
    get_filter_length,
    estimate_filter_length,
)

from .classified_fxlms import ClassifiedFxNLMS

__all__ = [
    # Spectrogram
    'compute_mel_spectrogram',
    'extract_mel_spectrogram',
    'normalize_spectrogram',
    'spectrogram_to_tensor',
    # Classifier
    'NoiseClassifier',
    'NoiseClassifierTrainer',
    # Parameters
    'FxNLMSParams',
    'OPTIMAL_PARAMS',
    'DEFAULT_PARAMS',
    'get_params',
    'get_params_dict',
    'get_step_size',
    'get_filter_length',
    'estimate_filter_length',
    # Wrapper
    'ClassifiedFxNLMS',
]
