"""
Phase 3: Train Neural ANC

Trains neural network to generate anti-noise.

Input: output/data/phase3/training_pairs.npz
Output: output/models/phase3/neural_anc.pt
"""

import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.ml.phase3_neural.neural_anc import create_model, save_model
from src.ml.phase3_neural.neural_anc_trainer import NeuralANCTrainer


def load_data(path):
    """Load training data."""
    data = np.load(path)
    return data['x_buffers'], data['d_samples'], data['secondary_path']


def split_data(x, d, val_ratio=0.2):
    """Split into train/val."""
    n = len(d)
    n_val = int(n * val_ratio)
    idx = np.random.permutation(n)
    return x[idx[n_val:]], d[idx[n_val:]], x[idx[:n_val]], d[idx[:n_val]]


def plot_training(history, output_path):
    """Plot training curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train')
    if history['val_loss']:
        plt.plot(history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Neural ANC Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    print("=" * 70)
    print("Phase 3: Training Neural ANC")
    print("=" * 70)

    data_path = Path('output/data/phase3/training_pairs.npz')
    model_path = Path('output/models/phase3/neural_anc.pt')
    plot_dir = Path('output/plots/phase3')

    if not data_path.exists():
        print(f"Error: Data not found at {data_path}")
        print("Run generate_training_pairs.py first.")
        return

    model_path.parent.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Load
    print(f"\nLoading data from {data_path}...")
    x_buffers, d_samples, secondary_path = load_data(data_path)
    print(f"  Samples: {len(d_samples)}, Buffer: {x_buffers.shape[1]}")

    # Split
    train_x, train_d, val_x, val_d = split_data(x_buffers, d_samples)
    print(f"  Train: {len(train_d)}, Val: {len(val_d)}")

    # Create model
    print("\nCreating model...")
    model = create_model('cnn', buffer_len=x_buffers.shape[1])
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")

    # Train
    trainer = NeuralANCTrainer(model, secondary_path, learning_rate=0.001)
    print("\nTraining...")
    history = trainer.train(
        train_x, train_d,
        val_x, val_d,
        epochs=100,
        batch_size=64,
        early_stopping_patience=15,
        verbose=True
    )

    # Save
    save_model(model, model_path)
    print(f"\nSaved model to {model_path}")

    # Plot
    plot_training(history, plot_dir / 'training_loss.png')

    print("\nDone!")


if __name__ == '__main__':
    main()
