"""
Binary Step Size Selector - IDLE vs Non-IDLE (Multi-Channel Optimized)

Simple binary classifier that only detects IDLE scenario.
- IDLE → μ=0.007 (optimal for multi-channel 4-speaker setup)
- Non-IDLE → μ=0.005 (conservative baseline)

This avoids misclassifying city/highway/acceleration, which have similar
optimal step sizes close to the baseline anyway.

Note: Multi-channel (4 speakers) changes optimal step sizes compared to
single-channel. IDLE prefers μ=0.007 instead of μ=0.015 due to combined
speaker output at error microphone.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Union


# Binary step sizes (optimized for multi-channel 4-speaker setup)
MU_IDLE = 0.007      # For IDLE scenario (49.3% optimal in multi-channel training)
MU_DEFAULT = 0.005   # For everything else (baseline)


class BinaryStepSizeSelector(nn.Module):
    """
    Binary classifier: IDLE vs non-IDLE.

    Simple architecture optimized for this single decision.
    """

    def __init__(
        self,
        input_dim: int = 12,  # 12 features from averaged 4-mic signal
        hidden_dim: int = 32,
        dropout: float = 0.2
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = 2

        # Store normalization parameters
        self.register_buffer('feature_mean', torch.zeros(input_dim))
        self.register_buffer('feature_std', torch.ones(input_dim))

        # Simple network for binary classification
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # 2 classes: non-idle, idle
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - output logits for [non-idle, idle].

        Args:
            x: Feature tensor of shape (batch_size, input_dim) or (input_dim,)

        Returns:
            Logits tensor of shape (batch_size, 2) or (2,)
        """
        x_normalized = (x - self.feature_mean) / (self.feature_std + 1e-8)
        return self.net(x_normalized)

    def predict(self, features: np.ndarray) -> float:
        """
        Predict step size from numpy features.

        Args:
            features: Feature array of shape (16,) or (n_samples, 16)

        Returns:
            Predicted step size (MU_IDLE or MU_DEFAULT)
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)

            logits = self.forward(x)
            is_idle = torch.argmax(logits, dim=-1)  # 1 = idle

            if is_idle.numel() == 1:
                return MU_IDLE if is_idle.item() == 1 else MU_DEFAULT

            result = np.where(is_idle.numpy() == 1, MU_IDLE, MU_DEFAULT)
            return result.astype(np.float32)

    def predict_class(self, features: np.ndarray) -> int:
        """
        Predict class (0=non-idle, 1=idle).

        Args:
            features: Feature array

        Returns:
            Class index
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
        Predict class probabilities [P(non-idle), P(idle)].

        Args:
            features: Feature array

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
            'model_type': 'binary',
            'mu_idle': MU_IDLE,
            'mu_default': MU_DEFAULT,
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
    """Trainer for the binary IDLE vs non-IDLE classifier."""

    def __init__(
        self,
        model: BinaryStepSizeSelector,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize trainer.

        Args:
            model: BinaryStepSizeSelector model
            learning_rate: Learning rate
            weight_decay: L2 regularization
            class_weights: Optional weights for [non-idle, idle] classes
        """
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
        train_scenarios: list,
        val_features: Optional[np.ndarray] = None,
        val_scenarios: Optional[list] = None,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 15,
        verbose: bool = True
    ) -> dict:
        """
        Full training loop.

        Args:
            train_features: Training features
            train_scenarios: Scenario labels ('idle', 'city', etc.)
            val_features: Validation features
            val_scenarios: Validation scenario labels
            epochs: Number of epochs
            batch_size: Mini-batch size
            early_stopping_patience: Stop if no improvement
            verbose: Print progress
        """
        # Compute and set normalization
        mean = np.mean(train_features, axis=0)
        std = np.std(train_features, axis=0)
        self.model.set_normalization(mean, std)

        # Convert scenario labels to binary (1 = idle, 0 = non-idle)
        train_targets = np.array([1 if s == 'idle' else 0 for s in train_scenarios])

        # Convert to tensors
        train_features_t = torch.tensor(train_features, dtype=torch.float32)
        train_targets_t = torch.tensor(train_targets, dtype=torch.long)

        if val_features is not None:
            val_targets = np.array([1 if s == 'idle' else 0 for s in val_scenarios])
            val_features_t = torch.tensor(val_features, dtype=torch.float32)
            val_targets_t = torch.tensor(val_targets, dtype=torch.long)

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(
                train_features_t, train_targets_t, batch_size
            )
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            if val_features is not None:
                val_loss, val_acc = self.validate(val_features_t, val_targets_t)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

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
                          f"train_loss={train_loss:.4f}, train_acc={train_acc:.1%}, "
                          f"val_loss={val_loss:.4f}, val_acc={val_acc:.1%}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}: "
                          f"train_loss={train_loss:.4f}, train_acc={train_acc:.1%}")

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.history
