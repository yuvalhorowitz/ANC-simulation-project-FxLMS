"""
Dynamic Step Size Selector — ML Model for Real-Time Adaptation

Predicts optimal FxNLMS step size from audio features + runtime state.
Trained on real car recordings and synthetic scenarios.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


STEP_SIZES = [0.001, 0.003, 0.005, 0.007, 0.01]
N_AUDIO_FEATURES = 16
N_RUNTIME_FEATURES = 3  # current_mse, weight_norm, mse_slope
N_TOTAL_FEATURES = N_AUDIO_FEATURES + N_RUNTIME_FEATURES
N_CLASSES = len(STEP_SIZES)

MODEL_PATH = Path(__file__).parent.parent.parent / 'output' / 'models' / 'phase2' / 'dynamic_step_selector.pt'


class DynamicStepSelector(nn.Module):
    """
    Multi-class classifier predicting optimal step size from features.

    Input: 19 features (16 audio + 3 runtime)
    Output: 5-class probability (one per candidate step size)
    """

    def __init__(self, n_features=N_TOTAL_FEATURES, n_classes=N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, n_classes),
        )
        self.n_features = n_features
        self.n_classes = n_classes
        self.feature_mean = None
        self.feature_std = None

    def forward(self, x):
        return self.net(x)

    def predict_step_size(self, features: np.ndarray) -> float:
        """
        Predict optimal step size from feature vector.

        Args:
            features: numpy array of shape (19,) — 16 audio + 3 runtime features

        Returns:
            Predicted optimal step size (float)
        """
        self.eval()
        with torch.no_grad():
            if self.feature_mean is not None:
                features = (features - self.feature_mean) / (self.feature_std + 1e-10)

            x = torch.FloatTensor(features).unsqueeze(0)
            logits = self.forward(x)
            class_idx = torch.argmax(logits, dim=1).item()
            return STEP_SIZES[class_idx]

    def predict_with_confidence(self, features: np.ndarray) -> tuple:
        """
        Predict step size with confidence score.

        Returns:
            (step_size, confidence) where confidence is max softmax probability
        """
        self.eval()
        with torch.no_grad():
            if self.feature_mean is not None:
                features = (features - self.feature_mean) / (self.feature_std + 1e-10)

            x = torch.FloatTensor(features).unsqueeze(0)
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            confidence, class_idx = torch.max(probs, dim=1)
            return STEP_SIZES[class_idx.item()], confidence.item()

    def save(self, path: Optional[Path] = None):
        """Save model weights and normalization stats."""
        if path is None:
            path = MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.state_dict(),
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            'step_sizes': STEP_SIZES,
            'n_features': self.n_features,
            'n_classes': self.n_classes,
        }, path)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> 'DynamicStepSelector':
        """Load trained model."""
        if path is None:
            path = MODEL_PATH
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model = cls(
            n_features=checkpoint['n_features'],
            n_classes=checkpoint['n_classes'],
        )
        model.load_state_dict(checkpoint['model_state'])
        model.feature_mean = checkpoint['feature_mean']
        model.feature_std = checkpoint['feature_std']
        model.eval()
        return model
