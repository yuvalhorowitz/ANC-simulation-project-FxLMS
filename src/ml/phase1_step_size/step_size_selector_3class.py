"""
3-Class Step Size Selector Neural Network

Intermediate complexity between binary and 5-class:
- Class 0 (Low μ): {0.003, 0.005} - Very stable, slow convergence
- Class 1 (Medium μ): {0.007, 0.010} - Balanced
- Class 2 (High μ): {0.015} - Fast convergence, less stable
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Union


# 3-class mapping
STEP_SIZE_MAPPING = {
    0.003: 0,  # Low μ
    0.005: 0,  # Low μ
    0.007: 1,  # Medium μ
    0.010: 1,  # Medium μ
    0.015: 2,  # High μ
}

CLASS_LABELS = ['low_mu', 'medium_mu', 'high_mu']


def step_size_to_3class(mu: float) -> int:
    """Convert step size value to 3-class (0=low, 1=medium, 2=high)."""
    if mu <= 0.005:
        return 0  # Low μ
    elif mu <= 0.010:
        return 1  # Medium μ
    else:
        return 2  # High μ


class ThreeClassStepSizeSelector(nn.Module):
    """
    3-class MLP classifier: Low μ vs Medium μ vs High μ.

    Takes a 12-dimensional feature vector and outputs logits for 3 classes.
    """

    CLASS_LABELS = CLASS_LABELS

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 32,
        dropout: float = 0.2
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = 3

        self.register_buffer('feature_mean', torch.zeros(input_dim))
        self.register_buffer('feature_std', torch.ones(input_dim))

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)  # 3 classes
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_normalized = (x - self.feature_mean) / (self.feature_std + 1e-8)
        return self.net(x_normalized)

    def predict(self, features: np.ndarray) -> int:
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
        self.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            return probs.numpy().squeeze()

    def set_normalization(self, mean: np.ndarray, std: np.ndarray):
        self.feature_mean = torch.tensor(mean, dtype=torch.float32)
        self.feature_std = torch.tensor(std, dtype=torch.float32)

    def save(self, path: Union[str, Path]):
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
    def load(cls, path: Union[str, Path]) -> 'ThreeClassStepSizeSelector':
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


class ThreeClassStepSizeSelectorTrainer:
    """Trainer for 3-class step size selector."""

    def __init__(
        self,
        model: ThreeClassStepSizeSelector,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None
    ):
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
        mean = np.mean(train_features, axis=0)
        std = np.std(train_features, axis=0)
        self.model.set_normalization(mean, std)

        train_3class_targets = np.array([step_size_to_3class(mu) for mu in train_targets])
        train_features = torch.tensor(train_features, dtype=torch.float32)
        train_targets_tensor = torch.tensor(train_3class_targets, dtype=torch.long)

        if val_features is not None:
            val_3class_targets = np.array([step_size_to_3class(mu) for mu in val_targets])
            val_features = torch.tensor(val_features, dtype=torch.float32)
            val_targets_tensor = torch.tensor(val_3class_targets, dtype=torch.long)

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(
                train_features, train_targets_tensor, batch_size
            )
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            if val_features is not None:
                val_loss, val_acc = self.validate(val_features, val_targets_tensor)
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
                          f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
                          f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}: "
                          f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.history
