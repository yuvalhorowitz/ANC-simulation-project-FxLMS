"""
Train Dynamic Step Size Selector

Trains the multi-class classifier to predict optimal FxNLMS step size
from audio features + runtime state.
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.ml.phase2_dynamic_step.dynamic_step_selector import (
    DynamicStepSelector, STEP_SIZES, N_TOTAL_FEATURES, MODEL_PATH
)

DATA_PATH = Path(__file__).parent.parent.parent / 'output' / 'data' / 'phase2' / 'dynamic_step_training_data.json'


def load_data():
    """Load training data and prepare feature/label tensors."""
    with open(DATA_PATH) as f:
        data = json.load(f)

    samples = data['samples']
    print(f"Loaded {len(samples)} samples")

    X = []
    y = []

    step_to_idx = {s: i for i, s in enumerate(STEP_SIZES)}

    for sample in samples:
        audio_features = np.array(sample['features'], dtype=np.float32)
        runtime = sample['runtime_features']

        # Runtime features: MSE, weight_norm, mse_slope (0 for now — computed at runtime)
        mse_val = min(runtime.get('mse', 0.1), 10.0)  # clip
        weight_norm = min(runtime.get('weight_norm', 0.0), 100.0)
        mse_slope = 0.0  # not available in training data

        all_features = np.concatenate([
            audio_features,
            [mse_val, weight_norm, mse_slope]
        ]).astype(np.float32)

        label = sample['label']
        if label in step_to_idx:
            X.append(all_features)
            y.append(step_to_idx[label])

    X = np.array(X)
    y = np.array(y)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution:")
    for step_idx, count in sorted(Counter(y).items()):
        print(f"  {STEP_SIZES[step_idx]}: {count} ({100*count/len(y):.1f}%)")

    return X, y


def train(epochs=200, batch_size=64, lr=1e-3, val_split=0.2):
    """Train the dynamic step selector."""
    X, y = load_data()

    # Normalize features
    feature_mean = X.mean(axis=0)
    feature_std = X.std(axis=0)
    feature_std[feature_std < 1e-10] = 1.0
    X_norm = (X - feature_mean) / feature_std

    # Split train/val
    n_val = int(len(X) * val_split)
    indices = np.random.permutation(len(X))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train = torch.FloatTensor(X_norm[train_idx])
    y_train = torch.LongTensor(y[train_idx])
    X_val = torch.FloatTensor(X_norm[val_idx])
    y_val = torch.LongTensor(y[val_idx])

    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}")

    # Class weights for imbalanced data
    class_counts = np.bincount(y[train_idx], minlength=len(STEP_SIZES))
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * len(STEP_SIZES)
    class_weights = torch.FloatTensor(class_weights)

    # Model
    model = DynamicStepSelector(n_features=N_TOTAL_FEATURES, n_classes=len(STEP_SIZES))
    model.feature_mean = feature_mean
    model.feature_std = feature_std

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_acc = 0
    patience = 30
    patience_counter = 0

    print(f"\nTraining for up to {epochs} epochs...")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Acc':>7} | {'LR':>8}")
    print("-" * 55)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == batch_y).sum().item()
            total += len(batch_y)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_pred = torch.argmax(val_logits, dim=1)
            val_acc = (val_pred == y_val).float().mean().item()

        scheduler.step(1 - val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{epoch+1:>6} | {train_loss:>10.4f} | {train_acc:>8.1%} | {val_acc:>6.1%} | {current_lr:>8.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            model.save()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print(f"\nBest validation accuracy: {best_val_acc:.1%}")
    print(f"Model saved to: {MODEL_PATH}")

    # Final evaluation
    model = DynamicStepSelector.load()
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        val_pred = torch.argmax(val_logits, dim=1)

    print(f"\nPer-class accuracy:")
    for i, step in enumerate(STEP_SIZES):
        mask = y_val == i
        if mask.sum() > 0:
            acc = (val_pred[mask] == y_val[mask]).float().mean().item()
            print(f"  mu={step}: {acc:.1%} ({mask.sum().item()} samples)")

    return model


if __name__ == '__main__':
    train()
