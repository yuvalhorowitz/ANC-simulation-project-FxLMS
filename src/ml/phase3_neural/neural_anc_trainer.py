"""
Neural ANC Trainer

Training loop that backpropagates through the secondary path.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple


class SecondaryPathConv(nn.Module):
    """
    Differentiable convolution with secondary path.

    Wraps the secondary path impulse response as a fixed Conv1d
    so gradients can flow through.
    """

    def __init__(self, secondary_path: np.ndarray):
        super().__init__()

        # Secondary path as 1D convolution kernel
        s_len = len(secondary_path)
        self.s_len = s_len

        # Register as buffer (not trainable)
        s_tensor = torch.tensor(secondary_path, dtype=torch.float32)
        s_tensor = s_tensor.flip(0)  # Flip for convolution
        self.register_buffer('s_kernel', s_tensor.view(1, 1, -1))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Convolve anti-noise with secondary path.

        Args:
            y: Anti-noise signal [batch, n_samples]

        Returns:
            y': Anti-noise at error mic [batch, n_samples]
        """
        # Add channel dimension
        y = y.unsqueeze(1)  # [batch, 1, n_samples]

        # Pad for causal convolution
        y_padded = nn.functional.pad(y, (self.s_len - 1, 0))

        # Convolve
        y_prime = nn.functional.conv1d(y_padded, self.s_kernel)

        return y_prime.squeeze(1)  # [batch, n_samples]


class NeuralANCTrainer:
    """
    Trainer for Neural ANC models.

    Key insight: Backpropagates through S(z) to train the network
    to generate anti-noise that cancels after passing through
    the secondary path.
    """

    def __init__(
        self,
        model: nn.Module,
        secondary_path: np.ndarray,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        self.model = model
        self.secondary_conv = SecondaryPathConv(secondary_path)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.history = {'train_loss': [], 'val_loss': []}

    def train_batch(
        self,
        x_buffers: torch.Tensor,
        d_samples: torch.Tensor
    ) -> float:
        """
        Train on a batch.

        Args:
            x_buffers: Reference signal buffers [batch, buffer_len]
            d_samples: Noise at error mic [batch]

        Returns:
            Batch loss
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Generate anti-noise
        y = self.model(x_buffers)  # [batch]

        # Pass through secondary path (simplified: single sample)
        y_expanded = y.unsqueeze(1)  # [batch, 1]
        y_prime = self.secondary_conv(y_expanded).squeeze(1)  # [batch]

        # Error at error mic
        e = d_samples + y_prime

        # Loss = MSE
        loss = torch.mean(e ** 2)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validate(
        self,
        x_buffers: torch.Tensor,
        d_samples: torch.Tensor
    ) -> float:
        """Compute validation loss."""
        self.model.eval()
        with torch.no_grad():
            y = self.model(x_buffers)
            y_expanded = y.unsqueeze(1)
            y_prime = self.secondary_conv(y_expanded).squeeze(1)
            e = d_samples + y_prime
            return torch.mean(e ** 2).item()

    def train(
        self,
        train_x: np.ndarray,
        train_d: np.ndarray,
        val_x: Optional[np.ndarray] = None,
        val_d: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 64,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            train_x: Training reference buffers [n_samples, buffer_len]
            train_d: Training noise samples [n_samples]
            val_x: Validation reference buffers
            val_d: Validation noise samples
            epochs: Number of epochs
            batch_size: Mini-batch size
            early_stopping_patience: Stop if no improvement
            verbose: Print progress

        Returns:
            Training history
        """
        train_x = torch.tensor(train_x, dtype=torch.float32)
        train_d = torch.tensor(train_d, dtype=torch.float32)

        if val_x is not None:
            val_x = torch.tensor(val_x, dtype=torch.float32)
            val_d = torch.tensor(val_d, dtype=torch.float32)

        n_samples = train_x.shape[0]
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            # Shuffle
            indices = torch.randperm(n_samples)

            # Train batches
            total_loss = 0
            n_batches = 0
            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i + batch_size]
                loss = self.train_batch(train_x[batch_idx], train_d[batch_idx])
                total_loss += loss
                n_batches += 1

            avg_train_loss = total_loss / n_batches
            self.history['train_loss'].append(avg_train_loss)

            # Validate
            if val_x is not None:
                val_loss = self.validate(val_x, val_d)
                self.history['val_loss'].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}: train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}: train_loss={avg_train_loss:.6f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.history
