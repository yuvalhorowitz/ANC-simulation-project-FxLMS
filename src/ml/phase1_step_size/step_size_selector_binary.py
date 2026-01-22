"""
Binary Step Size Selector Neural Network

Simplified version that classifies into just 2 classes:
- Low μ: {0.003, 0.005, 0.007} - typical for CITY/HIGHWAY
- High μ: {0.010, 0.015} - typical for IDLE

This tests whether the model can learn basic scenario differentiation.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Union


# Binary classification: Low vs High step size
STEP_SIZE_MAPPING = {
    0.003: 0,  # Low μ
    0.005: 0,  # Low μ
    0.007: 0,  # Low μ
    0.010: 1,  # High μ
    0.015: 1,  # High μ
}

CLASS_LABELS = ['low_mu', 'high_mu']


def step_size_to_binary_class(mu: float) -> int:
    """Convert step size value to binary class (0=low, 1=high)."""
    if mu <= 0.007:
        return 0  # Low μ
    else:
        return 1  # High μ


def binary_class_to_range(class_idx: int) -> str:
    """Convert binary class to descriptive range."""
    return CLASS_LABELS[class_idx]


class BinaryStepSizeSelector(nn.Module):
    """
    Binary MLP classifier: Low μ vs High μ.

    Takes a 12-dimensional feature vector and outputs logits for 2 classes.

    Architecture:
        Input (12) → Linear(32) → ReLU → Dropout →
        Linear(32) → ReLU → Dropout → Linear(2)
    """

    CLASS_LABELS = CLASS_LABELS

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 32,
        dropout: float = 0.2
    ):
        """
        Initialize the binary step size classifier.

        Args:
            input_dim: Number of input features (default 12)
            hidden_dim: Hidden layer dimension (default 32)
            dropout: Dropout probability (default 0.2)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = 2  # Binary classification

        # Store normalization parameters
        self.register_buffer('feature_mean', torch.zeros(input_dim))
        self.register_buffer('feature_std', torch.ones(input_dim))

        # Network architecture (binary classification)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # Output logits for 2 classes
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
        Forward pass - output logits for binary classification.

        Args:
            x: Feature tensor of shape (batch_size, input_dim) or (input_dim,)

        Returns:
            Logits tensor of shape (batch_size, 2) or (2,)
        """
        # Normalize input
        x_normalized = (x - self.feature_mean) / (self.feature_std + 1e-8)

        # Get network output (raw logits)
        logits = self.net(x_normalized)

        return logits

    def predict(self, features: np.ndarray) -> int:
        """
        Predict binary class from numpy features.

        Args:
            features: Feature array of shape (12,) or (n_samples, 12)

        Returns:
            Predicted class (0=low, 1=high) as int or array
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)

            logits = self.forward(x)
            class_idx = torch.argmax(logits, dim=-1)

            if class_idx.numel() == 1:
                return class_idx.item()
            return class_idx.numpy()

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            features: Feature array of shape (12,) or (n_samples, 12)

        Returns:
            Probability array of shape (2,) or (n_samples, 2)
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)

            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)

            return probs.numpy().squeeze()

    def set_normalization(self, mean: np.ndarray, std: np.ndarray):
        """Set feature normalization parameters."""
        self.feature_mean = torch.tensor(mean, dtype=torch.float32)
        self.feature_std = torch.tensor(std, dtype=torch.float32)

    def save(self, path: Union[str, Path]):
        """Save model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
        }, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BinaryStepSizeSelector':
        """Load model from file."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        model = cls(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.feature_mean = checkpoint['feature_mean']
        model.feature_std = checkpoint['feature_std']

        model.eval()
        return model


class BinaryStepSizeSelectorTrainer:
    """Trainer for the BinaryStepSizeSelector model."""

    def __init__(
        self,
        model: BinaryStepSizeSelector,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None
    ):
        """Initialize trainer."""
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def train_epoch(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        batch_size: int = 32
    ) -> tuple:
        """Train for one epoch."""
        self.model.train()
        n_samples = features.shape[0]
        indices = torch.randperm(n_samples)
        total_loss = 0.0
        correct = 0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_features = features[batch_indices]
            batch_targets = targets[batch_indices]

            self.optimizer.zero_grad()
            logits = self.model(batch_features)
            loss = self.criterion(logits, batch_targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == batch_targets).sum().item()
            n_batches += 1

        accuracy = correct / n_samples
        return total_loss / n_batches, accuracy

    def validate(
        self,
        features: torch.Tensor,
        targets: torch.Tensor
    ) -> tuple:
        """Validate model."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(features)
            loss = self.criterion(logits, targets)
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == targets).float().mean().item()
        return loss.item(), accuracy

    def train(
        self,
        train_features: np.ndarray,
        train_targets: np.ndarray,
        val_features: Optional[np.ndarray] = None,
        val_targets: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 15,
        verbose: bool = True
    ) -> dict:
        """Full training loop."""
        # Compute and set normalization
        mean = np.mean(train_features, axis=0)
        std = np.std(train_features, axis=0)
        self.model.set_normalization(mean, std)

        # Convert step sizes to binary class indices
        train_binary_targets = np.array([step_size_to_binary_class(mu) for mu in train_targets])

        # Convert to tensors
        train_features = torch.tensor(train_features, dtype=torch.float32)
        train_targets_tensor = torch.tensor(train_binary_targets, dtype=torch.long)

        if val_features is not None:
            val_binary_targets = np.array([step_size_to_binary_class(mu) for mu in val_targets])
            val_features = torch.tensor(val_features, dtype=torch.float32)
            val_targets_tensor = torch.tensor(val_binary_targets, dtype=torch.long)

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(
                train_features, train_targets_tensor, batch_size
            )
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validate
            if val_features is not None:
                val_loss, val_acc = self.validate(val_features, val_targets_tensor)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}: "
                          f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
                          f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}: "
                          f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}")

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.history
