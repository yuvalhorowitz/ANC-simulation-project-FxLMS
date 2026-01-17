"""
Phase 2: Collect Noise Data for Classifier Training

Generates labeled mel spectrograms for training the noise classifier.

Output: output/data/phase2/noise_spectrograms.npz
"""

import numpy as np
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.noise.noise_mixer import NoiseMixer
from src.ml.phase2_classifier.spectrogram import extract_mel_spectrogram


# Configuration
FS = 16000
DURATION = 1.0  # 1 second clips
N_SAMPLES_PER_CLASS = 200
SCENARIOS = ['idle', 'city', 'highway']


def collect_spectrograms():
    """Collect labeled mel spectrograms."""
    print("=" * 70)
    print("Phase 2: Collecting Noise Data for Classifier")
    print("=" * 70)

    noise_gen = NoiseMixer(FS)

    all_spectrograms = []
    all_labels = []

    total = len(SCENARIOS) * N_SAMPLES_PER_CLASS

    for class_idx, scenario in enumerate(SCENARIOS):
        print(f"\nCollecting {scenario} samples...")

        for i in range(N_SAMPLES_PER_CLASS):
            print(f"\r  [{class_idx * N_SAMPLES_PER_CLASS + i + 1}/{total}]", end="")

            # Generate noise with variation
            np.random.seed(42 + class_idx * 1000 + i)
            signal = noise_gen.generate_scenario(DURATION, scenario)

            # Add amplitude variation
            amplitude = 0.5 + np.random.rand() * 1.0
            signal = signal * amplitude

            # Extract mel spectrogram
            mel_spec = extract_mel_spectrogram(signal, FS, target_shape=(64, 32))

            all_spectrograms.append(mel_spec)
            all_labels.append(class_idx)

    print("\n")

    spectrograms = np.array(all_spectrograms, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    return spectrograms, labels


def main():
    spectrograms, labels = collect_spectrograms()

    # Save
    output_path = Path('output/data/phase2/noise_spectrograms.npz')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        spectrograms=spectrograms,
        labels=labels,
        classes=SCENARIOS,
        timestamp=datetime.now().isoformat()
    )

    print(f"Saved {len(labels)} samples to {output_path}")
    print(f"Shape: {spectrograms.shape}")
    print(f"Classes: {SCENARIOS}")


if __name__ == '__main__':
    main()
