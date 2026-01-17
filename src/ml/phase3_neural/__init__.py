"""
Phase 3: Neural Anti-Noise Generator

Replaces the FIR adaptive filter with a neural network
that learns to generate anti-noise directly.
"""

from .neural_anc import (
    NeuralANC_CNN,
    NeuralANC_LSTM,
    NeuralANC_MLP,
    create_model,
    save_model,
    load_model,
)

from .neural_anc_trainer import (
    SecondaryPathConv,
    NeuralANCTrainer,
)

from .neural_anc_wrapper import NeuralANCWrapper

__all__ = [
    # Models
    'NeuralANC_CNN',
    'NeuralANC_LSTM',
    'NeuralANC_MLP',
    'create_model',
    'save_model',
    'load_model',
    # Training
    'SecondaryPathConv',
    'NeuralANCTrainer',
    # Inference
    'NeuralANCWrapper',
]
