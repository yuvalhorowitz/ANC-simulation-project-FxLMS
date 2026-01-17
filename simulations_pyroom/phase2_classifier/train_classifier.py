"""
Phase 2: Train Noise Classifier

Trains CNN classifier on mel spectrograms.

Input: output/data/phase2/noise_spectrograms.npz
Output: output/models/phase2/noise_classifier.pt
"""

import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.ml.phase2_classifier.noise_classifier import NoiseClassifier, NoiseClassifierTrainer


def load_data(data_path):
    """Load spectrograms and labels."""
    data = np.load(data_path, allow_pickle=True)
    return data['spectrograms'], data['labels'], list(data['classes'])


def split_data(specs, labels, val_ratio=0.2):
    """Split into train/val sets."""
    n = len(labels)
    n_val = int(n * val_ratio)
    indices = np.random.permutation(n)
    val_idx, train_idx = indices[:n_val], indices[n_val:]
    return specs[train_idx], labels[train_idx], specs[val_idx], labels[val_idx]


def plot_training_history(history, output_path):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    if history['val_loss']:
        axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train')
    if history['val_acc']:
        axes[1].plot(history['val_acc'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def plot_confusion_matrix(confusion, classes, output_path):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(confusion, cmap='Blues')

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')

    # Add text annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(confusion[i, j]), ha='center', va='center')

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    print("=" * 70)
    print("Phase 2: Training Noise Classifier")
    print("=" * 70)

    data_path = Path('output/data/phase2/noise_spectrograms.npz')
    model_path = Path('output/models/phase2/noise_classifier.pt')
    plot_dir = Path('output/plots/phase2')

    if not data_path.exists():
        print(f"Error: Data not found at {data_path}")
        print("Run collect_noise_data.py first.")
        return

    model_path.parent.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\nLoading data from {data_path}...")
    specs, labels, classes = load_data(data_path)
    print(f"  Samples: {len(labels)}")
    print(f"  Classes: {classes}")

    # Split
    train_specs, train_labels, val_specs, val_labels = split_data(specs, labels)
    print(f"  Train: {len(train_labels)}, Val: {len(val_labels)}")

    # Create model
    model = NoiseClassifier(n_classes=len(classes))
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")

    # Train
    trainer = NoiseClassifierTrainer(model, learning_rate=0.001)
    print("\nTraining...")
    history = trainer.train(
        train_specs, train_labels,
        val_specs, val_labels,
        epochs=50,
        batch_size=32,
        early_stopping_patience=10,
        verbose=True
    )

    # Save
    model.save(model_path)
    print(f"\nSaved model to {model_path}")

    # Evaluate
    confusion = trainer.compute_confusion_matrix(val_specs, val_labels)
    accuracy = np.trace(confusion) / np.sum(confusion)
    print(f"\nValidation Accuracy: {accuracy:.1%}")

    # Plots
    plot_training_history(history, plot_dir / 'classifier_training.png')
    plot_confusion_matrix(confusion, classes, plot_dir / 'confusion_matrix.png')

    print("\nDone!")


if __name__ == '__main__':
    main()
