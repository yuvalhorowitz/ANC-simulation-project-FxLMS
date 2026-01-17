"""
Neural ANC Models

Neural networks that generate anti-noise directly.
Replaces the FIR adaptive filter with a learnable network.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple


class NeuralANC_CNN(nn.Module):
    """
    1D CNN for anti-noise generation.

    Fast inference, good for real-time applications.

    Input: Reference signal buffer [batch, buffer_len]
    Output: Anti-noise sample [batch]
    """

    def __init__(self, buffer_len: int = 256, hidden_channels: int = 32):
        super().__init__()
        self.buffer_len = buffer_len

        self.conv = nn.Sequential(
            # Input: [batch, 1, buffer_len]
            nn.Conv1d(1, hidden_channels, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.fc = nn.Linear(hidden_channels, 1)

    def forward(self, x_buffer: torch.Tensor) -> torch.Tensor:
        """
        Generate anti-noise from reference buffer.

        Args:
            x_buffer: [batch, buffer_len] reference signal history

        Returns:
            [batch] anti-noise samples
        """
        x = x_buffer.unsqueeze(1)  # [batch, 1, buffer_len]
        x = self.conv(x)  # [batch, hidden, 1]
        x = x.squeeze(-1)  # [batch, hidden]
        return self.fc(x).squeeze(-1)  # [batch]


class NeuralANC_LSTM(nn.Module):
    """
    LSTM for anti-noise generation.

    Better for sequential patterns, slower inference.

    Input: Reference signal buffer [batch, buffer_len]
    Output: Anti-noise sample [batch]
    """

    def __init__(self, buffer_len: int = 256, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.buffer_len = buffer_len

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x_buffer: torch.Tensor) -> torch.Tensor:
        """
        Generate anti-noise from reference buffer.

        Args:
            x_buffer: [batch, buffer_len] reference signal history

        Returns:
            [batch] anti-noise samples
        """
        x = x_buffer.unsqueeze(-1)  # [batch, buffer_len, 1]
        lstm_out, _ = self.lstm(x)  # [batch, buffer_len, hidden]
        last_output = lstm_out[:, -1, :]  # [batch, hidden]
        return self.fc(last_output).squeeze(-1)  # [batch]


class NeuralANC_MLP(nn.Module):
    """
    Simple MLP for anti-noise generation.

    Fastest inference, baseline neural model.

    Input: Reference signal buffer [batch, buffer_len]
    Output: Anti-noise sample [batch]
    """

    def __init__(self, buffer_len: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.buffer_len = buffer_len

        self.net = nn.Sequential(
            nn.Linear(buffer_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_buffer: torch.Tensor) -> torch.Tensor:
        """
        Generate anti-noise from reference buffer.

        Args:
            x_buffer: [batch, buffer_len] reference signal history

        Returns:
            [batch] anti-noise samples
        """
        return self.net(x_buffer).squeeze(-1)


def create_model(model_type: str = 'cnn', buffer_len: int = 256, **kwargs) -> nn.Module:
    """
    Factory function to create neural ANC model.

    Args:
        model_type: 'cnn', 'lstm', or 'mlp'
        buffer_len: Input buffer length
        **kwargs: Model-specific parameters

    Returns:
        Neural ANC model
    """
    models = {
        'cnn': NeuralANC_CNN,
        'lstm': NeuralANC_LSTM,
        'mlp': NeuralANC_MLP,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")

    return models[model_type](buffer_len=buffer_len, **kwargs)


def save_model(model: nn.Module, path: Union[str, Path], metadata: Optional[dict] = None):
    """Save model with metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'buffer_len': model.buffer_len,
    }
    if metadata:
        save_dict['metadata'] = metadata

    torch.save(save_dict, path)


def load_model(path: Union[str, Path]) -> nn.Module:
    """Load model from file."""
    checkpoint = torch.load(path, map_location='cpu')

    model_class = checkpoint['model_class']
    buffer_len = checkpoint['buffer_len']

    if model_class == 'NeuralANC_CNN':
        model = NeuralANC_CNN(buffer_len=buffer_len)
    elif model_class == 'NeuralANC_LSTM':
        model = NeuralANC_LSTM(buffer_len=buffer_len)
    elif model_class == 'NeuralANC_MLP':
        model = NeuralANC_MLP(buffer_len=buffer_len)
    else:
        raise ValueError(f"Unknown model class: {model_class}")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model
