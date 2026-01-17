"""
Step Size Selector Neural Network

PyTorch MLP that predicts optimal step size (μ) from signal features.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Union


class StepSizeSelector(nn.Module):
    """
    MLP that predicts optimal step size from signal features.

    Takes an 8-dimensional feature vector and outputs a step size μ
    in the range [mu_min, mu_max].

    Architecture:
        Input (8) → Linear(32) → ReLU → Dropout →
        Linear(32) → ReLU → Dropout → Linear(1) → Sigmoid → Scale

    Attributes:
        mu_min: Minimum allowed step size
        mu_max: Maximum allowed step size
        input_dim: Number of input features
        hidden_dim: Hidden layer size
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 32,
        mu_min: float = 0.0005,
        mu_max: float = 0.05,
        dropout: float = 0.1
    ):
        """
        Initialize the step size selector.

        Args:
            input_dim: Number of input features (default 8)
            hidden_dim: Hidden layer dimension (default 32)
            mu_min: Minimum step size to output (default 0.0005)
            mu_max: Maximum step size to output (default 0.05)
            dropout: Dropout probability (default 0.1)
        """
        super().__init__()

        self.mu_min = mu_min
        self.mu_max = mu_max
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Store normalization parameters
        self.register_buffer('feature_mean', torch.zeros(input_dim))
        self.register_buffer('feature_std', torch.ones(input_dim))

        # Network architecture
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - predict step size from features.

        Args:
            x: Feature tensor of shape (batch_size, input_dim) or (input_dim,)

        Returns:
            Step size tensor of shape (batch_size, 1) or (1,)
        """
        # Normalize input
        x_normalized = (x - self.feature_mean) / (self.feature_std + 1e-8)

        # Get network output in [0, 1]
        normalized_mu = self.net(x_normalized)

        # Scale to [mu_min, mu_max]
        mu = self.mu_min + (self.mu_max - self.mu_min) * normalized_mu

        return mu

    def predict(self, features: np.ndarray) -> float:
        """
        Predict step size from numpy features.

        Convenience method for inference.

        Args:
            features: Feature array of shape (8,) or (n_samples, 8)

        Returns:
            Predicted step size as float (or array if batched)
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)

            mu = self.forward(x)

            if mu.numel() == 1:
                return mu.item()
            return mu.squeeze().numpy()

    def set_normalization(self, mean: np.ndarray, std: np.ndarray):
        """
        Set feature normalization parameters.

        Should be called after training data statistics are computed.

        Args:
            mean: Feature mean, shape (8,)
            std: Feature standard deviation, shape (8,)
        """
        self.feature_mean = torch.tensor(mean, dtype=torch.float32)
        self.feature_std = torch.tensor(std, dtype=torch.float32)

    def save(self, path: Union[str, Path]):
        """
        Save model to file.

        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.state_dict(),
            'mu_min': self.mu_min,
            'mu_max': self.mu_max,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
        }, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'StepSizeSelector':
        """
        Load model from file.

        Args:
            path: Path to saved model

        Returns:
            Loaded StepSizeSelector instance
        """
        checkpoint = torch.load(path, map_location='cpu')

        model = cls(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            mu_min=checkpoint['mu_min'],
            mu_max=checkpoint['mu_max'],
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.feature_mean = checkpoint['feature_mean']
        model.feature_std = checkpoint['feature_std']

        model.eval()
        return model


class StepSizeSelectorTrainer:
    """
    Trainer for the StepSizeSelector model.
    """

    def __init__(
        self,
        model: StepSizeSelector,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4
    ):
        """
        Initialize trainer.

        Args:
            model: StepSizeSelector model to train
            learning_rate: Learning rate
            weight_decay: L2 regularization
        """
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        self.history = {'train_loss': [], 'val_loss': []}

    def train_epoch(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        batch_size: int = 32
    ) -> float:
        """
        Train for one epoch.

        Args:
            features: Training features, shape (n_samples, 8)
            targets: Target step sizes, shape (n_samples,)
            batch_size: Mini-batch size

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        n_samples = features.shape[0]
        indices = torch.randperm(n_samples)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_features = features[batch_indices]
            batch_targets = targets[batch_indices].unsqueeze(1)

            self.optimizer.zero_grad()
            predictions = self.model(batch_features)
            loss = self.criterion(predictions, batch_targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def validate(
        self,
        features: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """
        Validate model.

        Args:
            features: Validation features
            targets: Validation targets

        Returns:
            Validation loss
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(features)
            loss = self.criterion(predictions, targets.unsqueeze(1))
        return loss.item()

    def train(
        self,
        train_features: np.ndarray,
        train_targets: np.ndarray,
        val_features: Optional[np.ndarray] = None,
        val_targets: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> dict:
        """
        Full training loop.

        Args:
            train_features: Training features, shape (n_samples, 8)
            train_targets: Training targets, shape (n_samples,)
            val_features: Validation features (optional)
            val_targets: Validation targets (optional)
            epochs: Number of epochs
            batch_size: Mini-batch size
            early_stopping_patience: Stop if no improvement for this many epochs
            verbose: Print progress

        Returns:
            Training history
        """
        # Compute and set normalization
        mean = np.mean(train_features, axis=0)
        std = np.std(train_features, axis=0)
        self.model.set_normalization(mean, std)

        # Convert to tensors
        train_features = torch.tensor(train_features, dtype=torch.float32)
        train_targets = torch.tensor(train_targets, dtype=torch.float32)

        if val_features is not None:
            val_features = torch.tensor(val_features, dtype=torch.float32)
            val_targets = torch.tensor(val_targets, dtype=torch.float32)

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_features, train_targets, batch_size)
            self.history['train_loss'].append(train_loss)

            # Validate
            if val_features is not None:
                val_loss = self.validate(val_features, val_targets)
                self.history['val_loss'].append(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}: "
                          f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.6f}")

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.history
