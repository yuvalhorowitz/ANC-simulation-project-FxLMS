"""
Phase 1: Train Step Size Selector

Trains the neural network to predict optimal step size from signal features.

Input: output/data/phase1/step_size_training_data.json
Output: output/models/phase1/step_selector.pt
"""

import numpy as np
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.ml.phase1_step_size.step_size_selector import (
    StepSizeSelector,
    StepSizeSelectorTrainer
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
    Split data into training and validation sets.

    Returns:
        Tuple of (train_features, train_targets, val_features, val_targets)
    """
    n_samples = features.shape[0]
    n_val = int(n_samples * val_ratio)

    # Shuffle indices
    indices = np.random.permutation(n_samples)

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    return (
        features[train_indices],
        targets[train_indices],
        features[val_indices],
        targets[val_indices],
    )


def plot_training_history(history: dict, output_path: Path):
    """Plot and save training history."""
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(history['train_loss']) + 1)

    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    if history['val_loss']:
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Step Size Selector Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved training plot to {output_path}")
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
    ax.scatter(targets, predictions, alpha=0.6, edgecolors='k', linewidth=0.5)

    # Perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')

    ax.set_xlabel('Actual Best Step Size')
    ax.set_ylabel('Predicted Step Size')
    ax.set_title('Predicted vs Actual Step Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error histogram
    ax = axes[1]
    errors = predictions - targets
    ax.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', label='Zero error')
    ax.set_xlabel('Prediction Error (μ_pred - μ_actual)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Prediction Error Distribution\nMean: {np.mean(errors):.5f}, Std: {np.std(errors):.5f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved prediction plot to {output_path}")
    plt.close()


def main():
    """Main training script."""
    print("=" * 70)
    print("Phase 1: Training Step Size Selector")
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
    print(f"  Target range: [{targets.min():.5f}, {targets.max():.5f}]")

    # Split data
    print("\nSplitting data...")
    train_features, train_targets, val_features, val_targets = split_data(features, targets)
    print(f"  Training samples: {len(train_features)}")
    print(f"  Validation samples: {len(val_features)}")

    # Create model
    print("\nCreating model...")
    model = StepSizeSelector(
        input_dim=features.shape[1],
        hidden_dim=32,
        mu_min=0.0005,
        mu_max=0.05,
        dropout=0.1
    )
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")

    # Create trainer
    trainer = StepSizeSelectorTrainer(
        model=model,
        learning_rate=0.001,
        weight_decay=1e-4
    )

    # Train
    print("\nTraining...")
    history = trainer.train(
        train_features=train_features,
        train_targets=train_targets,
        val_features=val_features,
        val_targets=val_targets,
        epochs=100,
        batch_size=16,
        early_stopping_patience=15,
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

    train_mse = np.mean((train_predictions - train_targets) ** 2)
    val_mse = np.mean((val_predictions - val_targets) ** 2)

    print(f"\nTraining MSE: {train_mse:.6f}")
    print(f"Validation MSE: {val_mse:.6f}")

    train_mae = np.mean(np.abs(train_predictions - train_targets))
    val_mae = np.mean(np.abs(val_predictions - val_targets))

    print(f"Training MAE: {train_mae:.6f}")
    print(f"Validation MAE: {val_mae:.6f}")

    # Plot results
    print("\nGenerating plots...")
    plot_training_history(history, plot_dir / 'training_history.png')
    plot_predictions(model, features, targets, plot_dir / 'predictions.png')

    # Print sample predictions
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)
    print(f"{'Actual':>10} {'Predicted':>10} {'Error':>10}")
    print("-" * 32)
    for i in range(min(10, len(val_targets))):
        actual = val_targets[i]
        pred = val_predictions[i]
        error = pred - actual
        print(f"{actual:>10.5f} {pred:>10.5f} {error:>+10.5f}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nModel: {model_path}")
    print(f"Plots: {plot_dir}")


if __name__ == '__main__':
    main()
