"""
Train Binary Step Size Selector (IDLE vs Non-IDLE)

Simple binary classifier that detects IDLE scenario.
- IDLE → μ=0.015 (aggressive, works well for quiet stable noise)
- Non-IDLE → μ=0.005 (conservative baseline)

This avoids misclassifying city/highway/acceleration.
"""

import numpy as np
import json
import sys
import torch
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ml.phase1_step_size.step_size_selector_binary import (
    BinaryStepSizeSelector,
    BinaryStepSizeSelectorTrainer,
    MU_IDLE,
    MU_DEFAULT,
)


def load_training_data(data_path: Path):
    """Load training data from JSON file."""
    with open(data_path, 'r') as f:
        data = json.load(f)

    samples = data['samples']

    features = np.array([s['features'] for s in samples], dtype=np.float32)
    targets = np.array([s['best_step_size'] for s in samples], dtype=np.float32)
    scenarios = [s['scenario'] for s in samples]

    return features, targets, scenarios


def split_data(features, targets, scenarios, train_ratio=0.8, seed=42):
    """Split data into train/validation sets, stratified by scenario."""
    np.random.seed(seed)

    unique_scenarios = list(set(scenarios))
    train_idx = []
    val_idx = []

    for scenario in unique_scenarios:
        scenario_indices = [i for i, s in enumerate(scenarios) if s == scenario]
        np.random.shuffle(scenario_indices)
        n_train = int(len(scenario_indices) * train_ratio)
        train_idx.extend(scenario_indices[:n_train])
        val_idx.extend(scenario_indices[n_train:])

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)

    return (
        features[train_idx], targets[train_idx], [scenarios[i] for i in train_idx],
        features[val_idx], targets[val_idx], [scenarios[i] for i in val_idx]
    )


def plot_training_history(history: dict, output_path: Path):
    """Plot and save training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train')
    ax.plot(epochs, history['val_loss'], 'r-', label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, [a * 100 for a in history['train_acc']], 'b-', label='Train')
    ax.plot(epochs, [a * 100 for a in history['val_acc']], 'r-', label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Classification Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved training plot to {output_path}")
    plt.close()


def main():
    print("=" * 70)
    print("Training Binary Step Size Selector (IDLE vs Non-IDLE)")
    print("=" * 70)
    print(f"\nStrategy:")
    print(f"  IDLE     → μ = {MU_IDLE} (aggressive)")
    print(f"  Non-IDLE → μ = {MU_DEFAULT} (conservative baseline)")

    # Paths
    data_path = Path('output/data/phase1/step_size_training_data.json')
    model_path = Path('output/models/phase1/step_selector_binary.pt')
    plot_dir = Path('output/plots/phase1')

    plot_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading training data...")
    features, targets, scenarios = load_training_data(data_path)
    print(f"Loaded {len(features)} samples with {features.shape[1]} features")

    # Show scenario distribution
    print("\n" + "=" * 70)
    print("SCENARIO DISTRIBUTION")
    print("=" * 70)

    scenario_counts = Counter(scenarios)
    for scenario in ['idle', 'city', 'highway', 'acceleration']:
        count = scenario_counts.get(scenario, 0)
        label = "IDLE (→ μ=0.015)" if scenario == 'idle' else "Non-IDLE (→ μ=0.005)"
        print(f"  {scenario:12s}: {count:3d} samples  [{label}]")

    # Binary class distribution
    n_idle = scenario_counts.get('idle', 0)
    n_non_idle = len(scenarios) - n_idle
    print(f"\nBinary split: {n_idle} IDLE vs {n_non_idle} Non-IDLE")

    # Class weights (handle imbalance: 1 IDLE vs 3 Non-IDLE scenarios)
    class_weights = torch.tensor([
        1.0,                          # Non-IDLE weight
        n_non_idle / (n_idle + 1e-6)  # IDLE weight (upweight minority class)
    ], dtype=torch.float32)
    print(f"Class weights: [Non-IDLE: {class_weights[0]:.2f}, IDLE: {class_weights[1]:.2f}]")

    # Split data
    print("\nSplitting data (80/20 train/val, stratified)...")
    (train_features, train_targets, train_scenarios,
     val_features, val_targets, val_scenarios) = split_data(features, targets, scenarios)

    print(f"Training set: {len(train_features)} samples")
    print(f"Validation set: {len(val_features)} samples")

    # Verify stratification
    train_idle = sum(1 for s in train_scenarios if s == 'idle')
    val_idle = sum(1 for s in val_scenarios if s == 'idle')
    print(f"  Train IDLE: {train_idle}, Val IDLE: {val_idle}")

    # Create model
    print("\n" + "=" * 70)
    print("MODEL")
    print("=" * 70)

    model = BinaryStepSizeSelector(
        input_dim=features.shape[1],
        hidden_dim=32,
        dropout=0.2
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Architecture: {features.shape[1]} → 32 → 16 → 2")
    print(f"Parameters: {n_params:,}")

    # Create trainer
    trainer = BinaryStepSizeSelectorTrainer(
        model,
        learning_rate=0.001,
        weight_decay=1e-4,
        class_weights=class_weights
    )

    # Train
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    history = trainer.train(
        train_features=train_features,
        train_scenarios=train_scenarios,
        val_features=val_features,
        val_scenarios=val_scenarios,
        epochs=150,
        batch_size=32,
        early_stopping_patience=20,
        verbose=True
    )

    # Save model
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    # Plot training history
    plot_training_history(history, plot_dir / 'training_history_binary.png')

    # Final evaluation
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    # Validation accuracy
    val_predictions = model.predict_class(val_features)
    val_targets_binary = np.array([1 if s == 'idle' else 0 for s in val_scenarios])
    val_correct = np.sum(val_predictions == val_targets_binary)
    val_accuracy = val_correct / len(val_features)

    print(f"\nValidation Accuracy: {val_accuracy:.1%} ({val_correct}/{len(val_features)})")

    # Per-scenario accuracy
    print("\nPer-Scenario Accuracy:")
    for scenario in ['idle', 'city', 'highway', 'acceleration']:
        mask = np.array([s == scenario for s in val_scenarios])
        if mask.sum() > 0:
            scenario_preds = val_predictions[mask]
            expected = 1 if scenario == 'idle' else 0
            correct = np.sum(scenario_preds == expected)
            acc = correct / mask.sum()
            status = "✓" if acc > 0.8 else "✗"
            print(f"  {scenario:12s}: {acc:5.1%} ({correct}/{mask.sum()}) {status}")

    # Confusion matrix
    print("\nConfusion Matrix:")
    print("                  Predicted")
    print("               Non-IDLE  IDLE")
    tp = np.sum((val_targets_binary == 1) & (val_predictions == 1))
    tn = np.sum((val_targets_binary == 0) & (val_predictions == 0))
    fp = np.sum((val_targets_binary == 0) & (val_predictions == 1))
    fn = np.sum((val_targets_binary == 1) & (val_predictions == 0))
    print(f"Actual Non-IDLE   {tn:4d}     {fp:4d}")
    print(f"Actual IDLE       {fn:4d}     {tp:4d}")

    # Precision/Recall for IDLE class
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nIDLE Detection Metrics:")
    print(f"  Precision: {precision:.1%} (of predicted IDLE, how many are correct)")
    print(f"  Recall:    {recall:.1%} (of actual IDLE, how many detected)")
    print(f"  F1 Score:  {f1:.1%}")

    # Expected NR improvement calculation
    print("\n" + "=" * 70)
    print("EXPECTED NR IMPROVEMENT")
    print("=" * 70)

    # From the 5-class evaluation, IDLE gained +1.47 dB
    idle_gain = 1.47
    # With binary model:
    # - If we correctly detect IDLE (recall), we get the gain
    # - If we incorrectly predict IDLE for non-IDLE (false positive), we might lose
    # Assuming non-IDLE with μ=0.015 loses ~0.5 dB on average

    expected_idle_contribution = recall * idle_gain * (n_idle / len(scenarios))
    false_positive_loss = (fp / n_non_idle) * 0.5 * (n_non_idle / len(scenarios))
    expected_mean_improvement = expected_idle_contribution - false_positive_loss

    print(f"IDLE gain (from 5-class eval): +{idle_gain:.2f} dB")
    print(f"IDLE recall: {recall:.1%}")
    print(f"False positive rate: {fp}/{n_non_idle} = {100*fp/n_non_idle:.1f}%")
    print(f"Expected mean improvement: ~{expected_mean_improvement:+.2f} dB")

    print("\n" + "=" * 70)
    if val_accuracy >= 0.90:
        print("SUCCESS: High accuracy binary classifier ready!")
        print("Run evaluate_step_selector_binary.py to verify NR improvement.")
    else:
        print(f"Binary classifier accuracy: {val_accuracy:.1%}")
        print("May need feature engineering or more training data.")
    print("=" * 70)


if __name__ == '__main__':
    main()
