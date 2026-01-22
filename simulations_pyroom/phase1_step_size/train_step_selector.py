"""
Phase 1: Train Step Size Selector (Classification Version)

Trains the neural network classifier to predict optimal step size class
from signal features.

Input: output/data/phase1/step_size_training_data.json
Output: output/models/phase1/step_selector.pt
"""

import numpy as np
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.ml.phase1_step_size.step_size_selector import (
    StepSizeSelector,
    StepSizeSelectorTrainer,
    STEP_SIZES,
    step_size_to_class,
)


def load_training_data(data_path: Path) -> tuple:
    """
    Load and prepare training data.

    Returns:
        Tuple of (features, targets, metadata)
    """
    with open(data_path, 'r') as f:
        data = json.load(f)

    samples = data['samples']

    features = np.array([s['features'] for s in samples], dtype=np.float32)
    targets = np.array([s['best_step_size'] for s in samples], dtype=np.float32)

    return features, targets, data


def split_data(features: np.ndarray, targets: np.ndarray, val_ratio: float = 0.2):
    """
    Split data into training and validation sets (stratified by scenario).

    Returns:
        Tuple of (train_features, train_targets, val_features, val_targets)
    """
    n_samples = features.shape[0]
    n_val = int(n_samples * val_ratio)

    # Shuffle indices
    np.random.seed(42)  # For reproducibility
    indices = np.random.permutation(n_samples)

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    return (
        features[train_indices],
        targets[train_indices],
        features[val_indices],
        targets[val_indices],
    )


def compute_class_weights(targets: np.ndarray) -> torch.Tensor:
    """
    Compute class weights using inverse frequency to handle imbalance.

    Args:
        targets: Training targets (step size values)

    Returns:
        Tensor of class weights for CrossEntropyLoss
    """
    # Count samples per class
    class_counts = np.zeros(len(STEP_SIZES))
    for i, step_size in enumerate(STEP_SIZES):
        class_counts[i] = (targets == step_size).sum()

    # Compute inverse frequency weights
    # weight_i = total_samples / (n_classes * count_i)
    n_samples = len(targets)
    n_classes = len(STEP_SIZES)
    weights = n_samples / (n_classes * class_counts)

    # Normalize so minimum weight is 1.0
    weights = weights / weights.min()

    return torch.tensor(weights, dtype=torch.float32)


def plot_training_history(history: dict, output_path: Path):
    """Plot and save training history (loss and accuracy)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss plot
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    if history['val_loss']:
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (CrossEntropy)')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy plot
    ax = axes[1]
    ax.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
    if history['val_acc']:
        ax.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Classification Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved training plot to {output_path}")
    plt.close()


def plot_confusion_matrix(
    model: StepSizeSelector,
    features: np.ndarray,
    targets: np.ndarray,
    output_path: Path
):
    """Plot confusion matrix for step size classification."""
    predictions = model.predict(features)

    # Convert to class indices
    actual_classes = np.array([step_size_to_class(t) for t in targets])
    pred_classes = np.array([step_size_to_class(p) for p in predictions])

    n_classes = len(STEP_SIZES)
    confusion = np.zeros((n_classes, n_classes), dtype=int)

    for a, p in zip(actual_classes, pred_classes):
        confusion[a, p] += 1

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(confusion, cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    # Labels
    labels = [f'{s:.4f}' for s in STEP_SIZES]
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    # Add text annotations
    for i in range(n_classes):
        for j in range(n_classes):
            text = ax.text(j, i, confusion[i, j],
                          ha='center', va='center',
                          color='white' if confusion[i, j] > confusion.max()/2 else 'black')

    ax.set_xlabel('Predicted Step Size')
    ax.set_ylabel('Actual Step Size')
    ax.set_title('Step Size Classification Confusion Matrix')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved confusion matrix to {output_path}")
    plt.close()


def plot_predictions(
    model: StepSizeSelector,
    features: np.ndarray,
    targets: np.ndarray,
    output_path: Path
):
    """Plot predicted vs actual step sizes."""
    predictions = model.predict(features)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    ax = axes[0]
    # Add jitter for better visualization of discrete values
    jitter = 0.0002 * np.random.randn(len(targets))
    ax.scatter(targets + jitter, predictions + jitter, alpha=0.6, edgecolors='k', linewidth=0.5)

    # Perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')

    ax.set_xlabel('Actual Best Step Size')
    ax.set_ylabel('Predicted Step Size')
    ax.set_title('Predicted vs Actual Step Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Per-class accuracy
    ax = axes[1]
    class_accuracy = {}
    for step_size in STEP_SIZES:
        mask = targets == step_size
        if mask.sum() > 0:
            correct = (predictions[mask] == step_size).sum()
            class_accuracy[step_size] = correct / mask.sum()
        else:
            class_accuracy[step_size] = 0.0

    bars = ax.bar(range(len(STEP_SIZES)),
                  [class_accuracy.get(s, 0) for s in STEP_SIZES],
                  color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(STEP_SIZES)))
    ax.set_xticklabels([f'{s:.4f}' for s in STEP_SIZES], rotation=45, ha='right')
    ax.set_xlabel('Step Size Class')
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Class Classification Accuracy')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved prediction plot to {output_path}")
    plt.close()


def main():
    """Main training script."""
    print("=" * 70)
    print("Phase 1: Training Step Size Selector (Classification)")
    print("=" * 70)

    # Paths
    data_path = Path('output/data/phase1/step_size_training_data.json')
    model_path = Path('output/models/phase1/step_selector.pt')
    plot_dir = Path('output/plots/phase1')

    # Check data exists
    if not data_path.exists():
        print(f"Error: Training data not found at {data_path}")
        print("Run collect_training_data.py first.")
        return

    # Create output directories
    model_path.parent.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\nLoading data from {data_path}...")
    features, targets, metadata = load_training_data(data_path)
    print(f"  Total samples: {len(features)}")
    print(f"  Feature dimension: {features.shape[1]}")
    print(f"  Number of classes: {len(STEP_SIZES)}")

    # Print class distribution
    print("\nClass distribution:")
    for step_size in STEP_SIZES:
        count = (targets == step_size).sum()
        print(f"  μ={step_size:.4f}: {count} samples ({100*count/len(targets):.1f}%)")

    # Split data
    print("\nSplitting data...")
    train_features, train_targets, val_features, val_targets = split_data(features, targets)
    print(f"  Training samples: {len(train_features)}")
    print(f"  Validation samples: {len(val_features)}")

    # Compute class weights to handle imbalance
    print("\nComputing class weights...")
    class_weights = compute_class_weights(train_targets)
    print("  Class weights (inverse frequency):")
    for i, (step_size, weight) in enumerate(zip(STEP_SIZES, class_weights)):
        print(f"    μ={step_size:.4f}: weight={weight:.2f}")

    # Create model
    print("\nCreating model...")
    model = StepSizeSelector(
        input_dim=features.shape[1],
        hidden_dim=32,
        dropout=0.2
    )
    print(f"  Input dimension: {model.input_dim}")
    print(f"  Number of classes: {model.n_classes}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")

    # Create trainer
    trainer = StepSizeSelectorTrainer(
        model=model,
        learning_rate=0.001,
        weight_decay=1e-4,
        class_weights=class_weights  # Use inverse frequency weights to handle imbalance
    )

    # Train
    print("\nTraining...")
    history = trainer.train(
        train_features=train_features,
        train_targets=train_targets,
        val_features=val_features,
        val_targets=val_targets,
        epochs=150,
        batch_size=32,
        early_stopping_patience=20,
        verbose=True
    )

    # Save model
    model.save(model_path)
    print(f"\nModel saved to {model_path}")

    # Final evaluation
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    train_predictions = model.predict(train_features)
    val_predictions = model.predict(val_features)

    # Classification accuracy
    train_correct = (train_predictions == train_targets).sum()
    val_correct = (val_predictions == val_targets).sum()
    train_acc = train_correct / len(train_targets)
    val_acc = val_correct / len(val_targets)

    print(f"\nTraining Accuracy: {train_acc:.3f} ({train_correct}/{len(train_targets)})")
    print(f"Validation Accuracy: {val_acc:.3f} ({val_correct}/{len(val_targets)})")

    # Also compute "within 1 class" accuracy (more lenient)
    train_class_targets = np.array([step_size_to_class(t) for t in train_targets])
    train_class_preds = np.array([step_size_to_class(p) for p in train_predictions])
    val_class_targets = np.array([step_size_to_class(t) for t in val_targets])
    val_class_preds = np.array([step_size_to_class(p) for p in val_predictions])

    train_within_1 = (np.abs(train_class_preds - train_class_targets) <= 1).mean()
    val_within_1 = (np.abs(val_class_preds - val_class_targets) <= 1).mean()

    print(f"\nWithin 1 class accuracy:")
    print(f"  Training: {train_within_1:.3f}")
    print(f"  Validation: {val_within_1:.3f}")

    # Plot results
    print("\nGenerating plots...")
    plot_training_history(history, plot_dir / 'training_history.png')
    plot_predictions(model, features, targets, plot_dir / 'predictions.png')
    plot_confusion_matrix(model, features, targets, plot_dir / 'confusion_matrix.png')

    # Print sample predictions
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)
    print(f"{'Actual':>10} {'Predicted':>10} {'Correct':>10}")
    print("-" * 32)
    for i in range(min(15, len(val_targets))):
        actual = val_targets[i]
        pred = val_predictions[i]
        correct = "Yes" if actual == pred else "No"
        print(f"{actual:>10.5f} {pred:>10.5f} {correct:>10}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nModel: {model_path}")
    print(f"Plots: {plot_dir}")
    print(f"\nFinal validation accuracy: {val_acc:.1%}")
    if val_acc >= 0.70:
        print("Target accuracy (70%) achieved!")
    else:
        print(f"Target accuracy (70%) not yet achieved. Consider more training data.")


if __name__ == '__main__':
    main()
