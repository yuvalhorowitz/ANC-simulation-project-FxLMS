"""
Phase 3: Generate Training Pairs for Neural ANC

Creates (reference_buffer, noise_at_error, secondary_path) training data.

Output: output/data/phase3/training_pairs.npz
"""

import numpy as np
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pyroomacoustics as pra
from src.acoustic.path_generator import FIRPath
from src.noise.noise_mixer import NoiseMixer


# Configuration
ROOM_DIMS = [4.5, 1.85, 1.2]
ROOM_MATERIALS = {
    'ceiling': 0.38, 'floor': 0.52,
    'east': 0.14, 'west': 0.14,
    'north': 0.20, 'south': 0.30,
}
NOISE_SOURCE_POS = [0.3, 0.92, 0.4]
REF_MIC_POS = [0.3, 0.92, 0.5]
SPEAKER_POS = [1.9, 0.55, 1.0]
ERROR_MIC_POS = [1.8, 0.55, 1.0]

FS = 16000
BUFFER_LEN = 256
DURATION = 5.0  # seconds per scenario
N_SCENARIOS = 50  # Total scenarios to generate
SCENARIOS = ['idle', 'city', 'highway']


def create_room():
    """Create room and get paths."""
    materials = {k: pra.Material(v) for k, v in ROOM_MATERIALS.items()}
    room = pra.ShoeBox(ROOM_DIMS, fs=FS, materials=materials, max_order=3, air_absorption=True)
    room.add_source(NOISE_SOURCE_POS)
    room.add_source(SPEAKER_POS)
    room.add_microphone_array(pra.MicrophoneArray(np.array([REF_MIC_POS, ERROR_MIC_POS]).T, fs=FS))
    room.compute_rir()
    return {
        'primary': room.rir[1][0][:512],
        'reference': room.rir[0][0][:512],
        'secondary': room.rir[1][1][:512],
    }


def generate_training_pairs(noise_signal, paths, buffer_len=BUFFER_LEN):
    """Generate (x_buffer, d) pairs from a noise signal."""
    n_samples = len(noise_signal)
    reference_path = FIRPath(paths['reference'])
    primary_path = FIRPath(paths['primary'])

    # Storage
    x_buffers = []
    d_samples = []
    buffer = np.zeros(buffer_len)

    for i in range(n_samples):
        # Get reference and primary signals
        x = reference_path.filter_sample(noise_signal[i])
        d = primary_path.filter_sample(noise_signal[i])

        # Update buffer
        buffer = np.roll(buffer, -1)
        buffer[-1] = x

        # Store pair (after buffer is filled)
        if i >= buffer_len:
            x_buffers.append(buffer.copy())
            d_samples.append(d)

    return np.array(x_buffers), np.array(d_samples)


def main():
    print("=" * 70)
    print("Phase 3: Generating Training Pairs for Neural ANC")
    print("=" * 70)

    paths = create_room()
    noise_gen = NoiseMixer(FS)

    all_x_buffers = []
    all_d_samples = []

    print(f"\nGenerating {N_SCENARIOS} scenarios...")
    for i in range(N_SCENARIOS):
        scenario = SCENARIOS[i % len(SCENARIOS)]
        print(f"\r  [{i+1}/{N_SCENARIOS}] {scenario}", end="")

        np.random.seed(42 + i)
        noise = noise_gen.generate_scenario(DURATION, scenario)

        x_buffers, d_samples = generate_training_pairs(noise, paths)
        all_x_buffers.append(x_buffers)
        all_d_samples.append(d_samples)

    print("\n")

    # Combine
    x_buffers = np.vstack(all_x_buffers).astype(np.float32)
    d_samples = np.hstack(all_d_samples).astype(np.float32)

    # Save
    output_path = Path('output/data/phase3/training_pairs.npz')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        x_buffers=x_buffers,
        d_samples=d_samples,
        secondary_path=paths['secondary'],
        buffer_len=BUFFER_LEN,
        timestamp=datetime.now().isoformat()
    )

    print(f"Saved {len(d_samples)} samples to {output_path}")
    print(f"X shape: {x_buffers.shape}, D shape: {d_samples.shape}")


if __name__ == '__main__':
    main()
