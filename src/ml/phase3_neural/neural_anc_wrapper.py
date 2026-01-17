"""
Neural ANC Wrapper

Inference wrapper for trained neural ANC models.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Union, List


class NeuralANCWrapper:
    """
    Wrapper for neural ANC inference.

    Maintains a reference signal buffer and generates anti-noise
    sample by sample, similar to FxNLMS interface.

    Usage:
        wrapper = NeuralANCWrapper(model_path='model.pt', buffer_len=256)

        for x in reference_samples:
            y = wrapper.generate_antinoise(x)
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        model: Optional[torch.nn.Module] = None,
        buffer_len: int = 256
    ):
        """
        Initialize wrapper.

        Args:
            model_path: Path to saved model (or provide model directly)
            model: Pre-loaded model (or provide path)
            buffer_len: Reference signal buffer length
        """
        self.buffer_len = buffer_len

        # Load model
        if model is not None:
            self.model = model
        elif model_path is not None:
            from src.ml.phase3_neural.neural_anc import load_model
            self.model = load_model(model_path)
        else:
            raise ValueError("Must provide either model_path or model")

        self.model.eval()

        # Initialize buffer
        self.buffer = np.zeros(buffer_len, dtype=np.float32)
        self.buffer_idx = 0

        # Track MSE for comparison
        self.mse_history: List[float] = []
        self._mse_window = []

    def generate_antinoise(self, x: float) -> float:
        """
        Generate anti-noise for current reference sample.

        Args:
            x: Current reference signal sample

        Returns:
            Anti-noise sample y
        """
        # Update buffer (circular)
        self.buffer[self.buffer_idx] = x
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_len

        # Get buffer in correct order
        ordered_buffer = np.roll(self.buffer, -self.buffer_idx)

        # Generate anti-noise
        with torch.no_grad():
            x_tensor = torch.tensor(ordered_buffer, dtype=torch.float32).unsqueeze(0)
            y = self.model(x_tensor).item()

        return y

    def update_mse(self, e: float) -> None:
        """
        Update MSE history (for compatibility with FxNLMS interface).

        Args:
            e: Error signal sample
        """
        self._mse_window.append(e ** 2)
        if len(self._mse_window) >= 100:
            self.mse_history.append(np.mean(self._mse_window))
            self._mse_window = []

    def get_mse(self, window: int = 100) -> float:
        """Get recent MSE."""
        if len(self.mse_history) == 0:
            return float('inf')
        return np.mean(self.mse_history[-window:])

    def reset(self) -> None:
        """Reset buffer and MSE history."""
        self.buffer = np.zeros(self.buffer_len, dtype=np.float32)
        self.buffer_idx = 0
        self.mse_history = []
        self._mse_window = []

    def get_info(self) -> dict:
        """Get wrapper information."""
        return {
            'buffer_len': self.buffer_len,
            'model_class': self.model.__class__.__name__,
            'model_params': sum(p.numel() for p in self.model.parameters()),
        }
