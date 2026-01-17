"""
Noise Type Classifier

CNN that classifies noise type from mel spectrogram.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Tuple


class NoiseClassifier(nn.Module):
    """
    CNN that classifies noise type from mel spectrogram.

    Input: [batch, 1, 64, 32] mel spectrogram
    Output: [batch, n_classes] class probabilities

    Classes:
        - idle: Engine at rest, low RPM
        - city: Stop-and-go urban driving
        - highway: High-speed steady driving

    Architecture:
        Conv2d(1→16) → ReLU → MaxPool2d →
        Conv2d(16→32) → ReLU → MaxPool2d →
        Conv2d(32→64) → ReLU → AdaptiveAvgPool2d →
        Flatten → Linear(512→64) → ReLU → Dropout → Linear(64→n_classes)
    """

    CLASSES = ['idle', 'city', 'highway']

    def __init__(
        self,
        n_classes: int = 3,
        input_shape: Tuple[int, int] = (64, 32),
        dropout: float = 0.3
    ):
        """
        Initialize noise classifier.

        Args:
            n_classes: Number of output classes
            input_shape: Expected input shape (n_mels, n_frames)
            dropout: Dropout probability
        """
        super().__init__()

        self.n_classes = n_classes
        self.input_shape = input_shape

        # Convolutional layers
        self.conv = nn.Sequential(
            # Block 1: 64x32 -> 32x16
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2: 32x16 -> 16x8
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3: 16x8 -> 4x2 (adaptive)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 2))
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, n_mels, n_frames)

        Returns:
            Logits of shape (batch, n_classes)
        """
        x = self.conv(x)
        x = self.fc(x)
        return x

    def predict_proba(self, mel_spec: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities from mel spectrogram.

        Args:
            mel_spec: Mel spectrogram of shape (n_mels, n_frames) or
                      (batch, n_mels, n_frames)

        Returns:
            Class probabilities of shape (n_classes,) or (batch, n_classes)
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(mel_spec, dtype=torch.float32)

            # Add dimensions if needed
            if x.dim() == 2:
                x = x.unsqueeze(0).unsqueeze(0)  # (n_mels, n_frames) -> (1, 1, n_mels, n_frames)
            elif x.dim() == 3:
                x = x.unsqueeze(1)  # (batch, n_mels, n_frames) -> (batch, 1, n_mels, n_frames)

            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)

            if probs.shape[0] == 1:
                return probs.squeeze(0).numpy()
            return probs.numpy()

    def predict(self, mel_spec: np.ndarray) -> str:
        """
        Predict noise class from mel spectrogram.

        Args:
            mel_spec: Mel spectrogram of shape (n_mels, n_frames)

        Returns:
            Predicted class name
        """
        probs = self.predict_proba(mel_spec)
        idx = np.argmax(probs)
        return self.CLASSES[idx]

    def predict_batch(self, mel_specs: np.ndarray) -> List[str]:
        """
        Predict noise classes for a batch of spectrograms.

        Args:
            mel_specs: Batch of spectrograms (batch, n_mels, n_frames)

        Returns:
            List of predicted class names
        """
        probs = self.predict_proba(mel_specs)
        indices = np.argmax(probs, axis=1)
        return [self.CLASSES[i] for i in indices]

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
            'n_classes': self.n_classes,
            'input_shape': self.input_shape,
            'classes': self.CLASSES,
        }, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'NoiseClassifier':
        """
        Load model from file.

        Args:
            path: Path to saved model

        Returns:
            Loaded NoiseClassifier instance
        """
        checkpoint = torch.load(path, map_location='cpu')

        model = cls(
            n_classes=checkpoint['n_classes'],
            input_shape=checkpoint.get('input_shape', (64, 32)),
        )

        model.load_state_dict(checkpoint['model_state_dict'])

        # Update classes if saved
        if 'classes' in checkpoint:
            model.CLASSES = checkpoint['classes']

        model.eval()
        return model


class NoiseClassifierTrainer:
    """
    Trainer for NoiseClassifier model.
    """

    def __init__(
        self,
        model: NoiseClassifier,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4
    ):
        """
        Initialize trainer.

        Args:
            model: NoiseClassifier model to train
            learning_rate: Learning rate
            weight_decay: L2 regularization
        """
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def train_epoch(
        self,
        spectrograms: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int = 32
    ) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            spectrograms: Training spectrograms (n_samples, 1, n_mels, n_frames)
            labels: Class labels (n_samples,)
            batch_size: Mini-batch size

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        n_samples = spectrograms.shape[0]
        indices = torch.randperm(n_samples)

        total_loss = 0.0
        correct = 0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_specs = spectrograms[batch_indices]
            batch_labels = labels[batch_indices]

            self.optimizer.zero_grad()
            logits = self.model(batch_specs)
            loss = self.criterion(logits, batch_labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == batch_labels).sum().item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        accuracy = correct / n_samples

        return avg_loss, accuracy

    def validate(
        self,
        spectrograms: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Validate model.

        Args:
            spectrograms: Validation spectrograms
            labels: Validation labels

        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(spectrograms)
            loss = self.criterion(logits, labels).item()
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().mean().item()

        return loss, accuracy

    def train(
        self,
        train_specs: np.ndarray,
        train_labels: np.ndarray,
        val_specs: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> dict:
        """
        Full training loop.

        Args:
            train_specs: Training spectrograms (n_samples, n_mels, n_frames)
            train_labels: Training labels (n_samples,)
            val_specs: Validation spectrograms (optional)
            val_labels: Validation labels (optional)
            epochs: Number of epochs
            batch_size: Mini-batch size
            early_stopping_patience: Stop if no improvement
            verbose: Print progress

        Returns:
            Training history
        """
        # Add channel dimension and convert to tensors
        train_specs = torch.tensor(train_specs, dtype=torch.float32)
        if train_specs.dim() == 3:
            train_specs = train_specs.unsqueeze(1)  # Add channel dim
        train_labels = torch.tensor(train_labels, dtype=torch.long)

        if val_specs is not None:
            val_specs = torch.tensor(val_specs, dtype=torch.float32)
            if val_specs.dim() == 3:
                val_specs = val_specs.unsqueeze(1)
            val_labels = torch.tensor(val_labels, dtype=torch.long)

        best_val_acc = 0.0
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_specs, train_labels, batch_size)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validate
            if val_specs is not None:
                val_loss, val_acc = self.validate(val_specs, val_labels)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break

                if verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}: "
                          f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
                          f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")
            else:
                if verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}: "
                          f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}")

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.history

    def compute_confusion_matrix(
        self,
        spectrograms: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        """
        Compute confusion matrix.

        Args:
            spectrograms: Test spectrograms
            labels: True labels

        Returns:
            Confusion matrix of shape (n_classes, n_classes)
        """
        self.model.eval()
        specs = torch.tensor(spectrograms, dtype=torch.float32)
        if specs.dim() == 3:
            specs = specs.unsqueeze(1)

        with torch.no_grad():
            logits = self.model(specs)
            predictions = torch.argmax(logits, dim=1).numpy()

        n_classes = self.model.n_classes
        confusion = np.zeros((n_classes, n_classes), dtype=int)

        for true_label, pred_label in zip(labels, predictions):
            confusion[true_label, pred_label] += 1

        return confusion
