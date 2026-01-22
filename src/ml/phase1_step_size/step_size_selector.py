"""
Step Size Selector Neural Network (Classification Version)

PyTorch MLP that predicts optimal step size class from signal features.
Uses classification with 11 discrete step size classes instead of regression.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Union


# Step sizes as discrete classes (5 values that actually occur as optimal)
STEP_SIZES = [0.003, 0.005, 0.007, 0.01, 0.015]


def step_size_to_class(mu: float) -> int:
    """Convert step size value to class index."""
    try:
        return STEP_SIZES.index(mu)
    except ValueError:
        # Find closest step size
        distances = [abs(mu - s) for s in STEP_SIZES]
        return distances.index(min(distances))


def class_to_step_size(class_idx: int) -> float:
    """Convert class index to step size value."""
    return STEP_SIZES[class_idx]


class StepSizeSelector(nn.Module):
    """
    MLP classifier that predicts optimal step size class from signal features.

    Takes a 12-dimensional feature vector and outputs logits for 11 step size classes.

    Architecture:
        Input (12) → Linear(32) → ReLU → Dropout →
        Linear(32) → ReLU → Dropout → Linear(11)

    Attributes:
        step_sizes: List of step size values (class labels)
        input_dim: Number of input features
        hidden_dim: Hidden layer size
        n_classes: Number of step size classes
    """

    STEP_SIZES = STEP_SIZES  # Class attribute for access

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 32,
        dropout: float = 0.2
    ):
        """
        Initialize the step size classifier.

        Args:
            input_dim: Number of input features (default 12)
            hidden_dim: Hidden layer dimension (default 32)
            dropout: Dropout probability (default 0.2)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = len(STEP_SIZES)

        # Store normalization parameters
        self.register_buffer('feature_mean', torch.zeros(input_dim))
        self.register_buffer('feature_std', torch.ones(input_dim))

        # Network architecture (classification)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.n_classes)  # Output logits for each class
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
        Forward pass - output logits for each step size class.

        Args:
            x: Feature tensor of shape (batch_size, input_dim) or (input_dim,)

        Returns:
            Logits tensor of shape (batch_size, n_classes) or (n_classes,)
        """
        # Normalize input
        x_normalized = (x - self.feature_mean) / (self.feature_std + 1e-8)

        # Get network output (raw logits)
        logits = self.net(x_normalized)

        return logits

    def predict(self, features: np.ndarray) -> float:
        """
        Predict step size from numpy features.

        Convenience method for inference.

        Args:
            features: Feature array of shape (12,) or (n_samples, 12)

        Returns:
            Predicted step size as float (or array if batched)
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)

            logits = self.forward(x)
            class_idx = torch.argmax(logits, dim=-1)

            if class_idx.numel() == 1:
                return STEP_SIZES[class_idx.item()]
            # Return as float32 to match training data dtype
            return np.array([STEP_SIZES[i.item()] for i in class_idx], dtype=np.float32)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            features: Feature array of shape (12,) or (n_samples, 12)

        Returns:
            Probability array of shape (n_classes,) or (n_samples, n_classes)
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
        """
        Set feature normalization parameters.

        Should be called after training data statistics are computed.

        Args:
            mean: Feature mean, shape (12,)
            std: Feature standard deviation, shape (12,)
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
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'n_classes': self.n_classes,
            'step_sizes': STEP_SIZES,
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


class StepSizeSelectorTrainer:
    """
    Trainer for the StepSizeSelector classification model.
    """

    def __init__(
        self,
        model: StepSizeSelector,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize trainer.

        Args:
            model: StepSizeSelector model to train
            learning_rate: Learning rate
            weight_decay: L2 regularization
            class_weights: Optional class weights for handling imbalance
        """
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        # Use class weights to handle imbalanced classes
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def train_epoch(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        batch_size: int = 32
    ) -> tuple:
        """
        Train for one epoch.

        Args:
            features: Training features, shape (n_samples, 12)
            targets: Target class indices, shape (n_samples,)
            batch_size: Mini-batch size

        Returns:
            Tuple of (average loss, accuracy) for the epoch
        """
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
        """
        Validate model.

        Args:
            features: Validation features
            targets: Validation class indices

        Returns:
            Tuple of (validation loss, accuracy)
        """
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
        """
        Full training loop.

        Args:
            train_features: Training features, shape (n_samples, 12)
            train_targets: Training targets (step size values), shape (n_samples,)
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

        # Convert step sizes to class indices
        train_class_targets = np.array([step_size_to_class(mu) for mu in train_targets])

        # Convert to tensors
        train_features = torch.tensor(train_features, dtype=torch.float32)
        train_targets_tensor = torch.tensor(train_class_targets, dtype=torch.long)

        if val_features is not None:
            val_class_targets = np.array([step_size_to_class(mu) for mu in val_targets])
            val_features = torch.tensor(val_features, dtype=torch.float32)
            val_targets_tensor = torch.tensor(val_class_targets, dtype=torch.long)

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
