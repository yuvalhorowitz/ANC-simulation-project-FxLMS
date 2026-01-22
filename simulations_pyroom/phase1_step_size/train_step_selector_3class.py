"""
Train 3-Class Step Size Selector (Low μ vs Medium μ vs High μ)

Next incremental step after binary classification success.

3-Class Classification:
- Class 0 (Low μ): {0.003, 0.005} - very stable, slow convergence
- Class 1 (Medium μ): {0.007, 0.010} - balanced
- Class 2 (High μ): {0.015} - fast convergence, less stable
"""

import numpy as np
import json
import sys
from pathlib import Path
from collections import Counter

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ml.phase1_step_size.step_size_selector_3class import (
    ThreeClassStepSizeSelector,
    ThreeClassStepSizeSelectorTrainer,
    step_size_to_3class,
    CLASS_LABELS
)
import torch


def load_training_data(data_path: Path):
    """Load training data from JSON file."""
    with open(data_path, 'r') as f:
        data = json.load(f)

    samples = data['samples']

    # Extract features and targets
    features = np.array([s['features'] for s in samples], dtype=np.float32)
    targets = np.array([s['best_step_size'] for s in samples], dtype=np.float32)
    scenarios = [s['scenario'] for s in samples]

    return features, targets, scenarios


def split_data(features, targets, scenarios, train_ratio=0.8, seed=42):
    """Split data into train/validation sets."""
    np.random.seed(seed)
    n_samples = len(features)
    indices = np.random.permutation(n_samples)

    n_train = int(n_samples * train_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    return (
        features[train_idx], targets[train_idx], [scenarios[i] for i in train_idx],
        features[val_idx], targets[val_idx], [scenarios[i] for i in val_idx]
    )


def compute_class_weights(targets: np.ndarray) -> torch.Tensor:
    """
    Compute class weights for handling imbalanced data.

    Uses inverse frequency weighting.
    """
    # Convert to 3 classes
    class_targets = np.array([step_size_to_3class(mu) for mu in targets])

    class_counts = Counter(class_targets)
    n_classes = 3
    n_samples = len(class_targets)

    # Compute weights: inversely proportional to frequency
    weights = np.array([
        n_samples / (n_classes * class_counts[i])
        for i in range(n_classes)
    ], dtype=np.float32)

    # Normalize to make minimum weight = 1.0
    weights = weights / weights.min()

    return torch.tensor(weights, dtype=torch.float32)


def main():
    print("=" * 70)
    print("Training 3-Class Step Size Selector (Low/Medium/High μ)")
    print("=" * 70)

    # Paths
    data_path = Path('output/data/phase1/step_size_training_data.json')
    model_path = Path('output/models/phase1/step_selector_3class.pt')

    # Load data
    print("\nLoading training data...")
    features, targets, scenarios = load_training_data(data_path)
    print(f"Loaded {len(features)} samples with {features.shape[1]} features each")

    # Convert to 3 classes for analysis
    class_targets = np.array([step_size_to_3class(mu) for mu in targets])

    # Show 3-class distribution
    print("\n" + "=" * 70)
    print("3-CLASS DISTRIBUTION")
    print("=" * 70)

    class_counts = Counter(class_targets)
    for class_idx in [0, 1, 2]:
        count = class_counts[class_idx]
        pct = 100.0 * count / len(class_targets)
        mu_values = [t for t, c in zip(targets, class_targets) if c == class_idx]
        unique_mu = sorted(set(mu_values))
        print(f"\nClass {class_idx} ({CLASS_LABELS[class_idx]}):")
        print(f"  Total samples: {count} ({pct:.1f}%)")
        print(f"  Step sizes: {unique_mu}")
        print(f"  Scenarios breakdown:")
        for scenario in ['idle', 'city', 'highway']:
            scenario_count = sum(1 for s, c in zip(scenarios, class_targets)
                               if s == scenario and c == class_idx)
            print(f"    {scenario}: {scenario_count} samples")

    # Compute class weights
    class_weights = compute_class_weights(targets)
    print("\n" + "=" * 70)
    print("CLASS WEIGHTS (for handling imbalance)")
    print("=" * 70)
    for i in range(3):
        print(f"  {CLASS_LABELS[i]}: weight={class_weights[i]:.2f}")

    # Split data
    print("\nSplitting data (80/20 train/val)...")
    train_features, train_targets, train_scenarios, \
        val_features, val_targets, val_scenarios = split_data(features, targets, scenarios)

    print(f"Training set: {len(train_features)} samples")
    print(f"Validation set: {len(val_features)} samples")

    # Create model
    print("\n" + "=" * 70)
    print("MODEL ARCHITECTURE")
    print("=" * 70)

    model = ThreeClassStepSizeSelector(
        input_dim=12,
        hidden_dim=64,  # Increased from 32
        dropout=0.3     # Increased dropout for better regularization
    )

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: ThreeClassStepSizeSelector")
    print(f"Input features: 12")
    print(f"Hidden dimension: 32")
    print(f"Output classes: 3 (low/medium/high)")
    print(f"Total parameters: {n_params:,}")

    # Create trainer with class weights
    trainer = ThreeClassStepSizeSelectorTrainer(
        model,
        learning_rate=0.0005,  # Reduced from 0.001 for more careful optimization
        weight_decay=1e-4,
        class_weights=class_weights
    )

    # Train
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    history = trainer.train(
        train_features=train_features,
        train_targets=train_targets,
        val_features=val_features,
        val_targets=val_targets,
        epochs=200,  # Increased from 150
        batch_size=32,
        early_stopping_patience=30,  # Increased patience for larger model
        verbose=True
    )

    # Save model
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    # Training set accuracy
    train_class_targets = np.array([step_size_to_3class(mu) for mu in train_targets])
    train_predictions = model.predict(train_features)
    train_correct = np.sum(train_predictions == train_class_targets)
    train_accuracy = train_correct / len(train_features)

    # Validation set accuracy
    val_class_targets = np.array([step_size_to_3class(mu) for mu in val_targets])
    val_predictions = model.predict(val_features)
    val_correct = np.sum(val_predictions == val_class_targets)
    val_accuracy = val_correct / len(val_features)

    print(f"Training Accuracy: {train_accuracy:.3f} ({train_correct}/{len(train_features)})")
    print(f"Validation Accuracy: {val_accuracy:.3f} ({val_correct}/{len(val_features)})")

    # Per-scenario breakdown
    print("\nPer-scenario accuracy:")
    for scenario in ['idle', 'city', 'highway']:
        # Validation set
        scenario_mask = np.array([s == scenario for s in val_scenarios])
        if scenario_mask.sum() > 0:
            scenario_acc = np.mean(
                val_predictions[scenario_mask] == val_class_targets[scenario_mask]
            )
            n_scenario = scenario_mask.sum()
            print(f"  {scenario:8s}: {scenario_acc:.3f} ({n_scenario} samples)")

    # Confusion matrix
    print("\nConfusion Matrix (Validation Set):")
    print("                 Predicted")
    print("                 Low   Med   High")
    for true_class in range(3):
        print(f"Actual {CLASS_LABELS[true_class]:10s}", end=" ")
        for pred_class in range(3):
            count = np.sum((val_class_targets == true_class) & (val_predictions == pred_class))
            print(f"{count:4d}  ", end="")
        print()

    # Per-class accuracy
    print("\nPer-class accuracy:")
    for class_idx in range(3):
        class_mask = val_class_targets == class_idx
        if class_mask.sum() > 0:
            class_acc = np.mean(val_predictions[class_mask] == val_class_targets[class_mask])
            print(f"  {CLASS_LABELS[class_idx]:10s}: {class_acc:.3f} ({class_mask.sum()} samples)")

    # Success criteria
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA")
    print("=" * 70)

    target_accuracy = 0.70  # 70% target for 3-class classification
    passed = val_accuracy >= target_accuracy

    print(f"Target validation accuracy: {target_accuracy:.1%}")
    print(f"Achieved validation accuracy: {val_accuracy:.1%}")
    print(f"Status: {'✓ PASSED' if passed else '✗ FAILED'}")

    if passed:
        print("\n3-class classifier successfully learned to distinguish Low/Medium/High μ!")
        print("This shows:")
        print("  1. Model can handle intermediate complexity")
        print("  2. Incremental approach is working")
        print("  3. Ready to try 4 or 5 classes")
        print("\nNext step: Expand to 4-class or return to full 5-class")
    else:
        print("\n3-class classifier struggled with this complexity level.")
        print("This suggests:")
        print("  1. Jump from 2 to 3 classes may be too large")
        print("  2. May need more training data")
        print("  3. May need better features or architecture")
        print("\nRecommend: Analyze which classes are being confused")


if __name__ == '__main__':
    main()
