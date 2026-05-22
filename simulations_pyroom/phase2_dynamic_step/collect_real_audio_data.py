"""
Training Data Collection for Dynamic Step Size Prediction

Generates labeled training data by running FxNLMS with multiple step sizes
on 1-second segments of real and synthetic audio. For each segment, the
optimal step size (best noise reduction while stable) becomes the label.
"""

import sys
import json
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pyroomacoustics as pra
from src.acoustic.path_generator import AcousticPathGenerator, FIRPath
from src.core.fxlms import FxNLMS
from src.noise.noise_mixer import NoiseMixer
from src.ml.phase1_step_size.feature_extractor import extract_features

STEP_SIZES = [0.001, 0.003, 0.005, 0.007, 0.01]
SEGMENT_DURATION = 1.0  # seconds
STRIDE_DURATION = 0.5   # seconds
FS = 16000
FILTER_LENGTH = 512

ROOM_CONFIG = {
    'dimensions': [4.5, 1.85, 1.2],
    'absorption': 0.35,
    'max_order': 3,
    'positions': {
        'noise_source': [0.5, 0.92, 0.4],
        'reference_mic': [1.1, 0.92, 0.8],
        'speaker': [0.8, 0.25, 0.9],
        'error_mic': [2.5, 0.55, 1.05],
    }
}

REAL_AUDIO_DIR = Path(__file__).parent.parent.parent / 'real_noises'
REAL_AUDIO_FILES = [
    'realcar1.wav', 'realcar2.wav', 'realcar3.wav', 'realcar4.wav', 'realcar5.wav',
    'la_city_start.wav', 'la_city_stop_go.wav', 'la_quiet_cruise.wav',
    'la_idle.wav', 'la_varying.wav', 'la_medium_cruise.wav',
    'la_loud_low.wav', 'la_late_drive.wav',
]

SYNTHETIC_SCENARIOS = ['idle', 'city', 'highway', 'acceleration']

OUTPUT_PATH = Path(__file__).parent.parent.parent / 'output' / 'data' / 'phase2' / 'dynamic_step_training_data.json'


def create_room():
    """Create pyroomacoustics room and extract paths."""
    dims = ROOM_CONFIG['dimensions']
    absorption = ROOM_CONFIG['absorption']
    positions = ROOM_CONFIG['positions']

    materials = {
        'ceiling': pra.Material(absorption * 1.1),
        'floor': pra.Material(absorption * 1.5),
        'east': pra.Material(absorption * 0.5),
        'west': pra.Material(absorption * 0.5),
        'north': pra.Material(absorption * 0.7),
        'south': pra.Material(absorption * 0.9),
    }

    room = pra.ShoeBox(dims, fs=FS, materials=materials,
                       max_order=ROOM_CONFIG['max_order'], air_absorption=True)

    room.add_source(positions['noise_source'])
    room.add_source(positions['speaker'])

    mic_array = np.array([positions['reference_mic'], positions['error_mic']]).T
    room.add_microphone_array(pra.MicrophoneArray(mic_array, fs=FS))

    room.compute_rir()

    path_gen = AcousticPathGenerator(room)
    paths = path_gen.get_all_anc_paths(modeling_error=0.05)

    max_len = 512
    return {
        'primary': paths['primary'][:max_len],
        'secondary': paths['secondary'][:max_len],
        'secondary_estimate': paths['secondary_estimate'][:max_len],
        'reference': paths['reference'][:max_len],
    }


def run_segment(noise_segment, paths, step_size, initial_weights=None):
    """
    Run FxNLMS on a 1-second noise segment with given step size.
    Returns noise reduction (dB) and final filter state.
    """
    primary_path = FIRPath(paths['primary'])
    secondary_path = FIRPath(paths['secondary'])
    reference_path = FIRPath(paths['reference'])

    fxlms = FxNLMS(
        filter_length=FILTER_LENGTH,
        step_size=step_size,
        secondary_path_estimate=paths['secondary_estimate'],
        regularization=1e-4
    )

    if initial_weights is not None:
        fxlms.weights = initial_weights.copy()

    n_samples = len(noise_segment)
    desired = np.zeros(n_samples)
    error = np.zeros(n_samples)

    for i in range(n_samples):
        sample = noise_segment[i]
        x = reference_path.filter_sample(sample)
        d = primary_path.filter_sample(sample)
        desired[i] = d

        y = fxlms.generate_antinoise(x)
        y_at_error = secondary_path.filter_sample(y)
        e = d + y_at_error
        error[i] = e

        fxlms.filter_reference(x)
        fxlms.update_weights(e)

    # Compute noise reduction (second half for steady state)
    half = n_samples // 2
    d_power = np.mean(desired[half:] ** 2)
    e_power = np.mean(error[half:] ** 2)

    if e_power > 1e-10 and d_power > 1e-10:
        nr_db = 10 * np.log10(d_power / e_power)
    else:
        nr_db = 0.0

    # Check stability
    is_stable = not np.any(np.isnan(error)) and not np.any(np.isinf(error))
    if is_stable:
        is_stable = np.max(np.abs(error)) < 10 * np.max(np.abs(desired) + 1e-10)

    mse_final = np.mean(error[-1000:] ** 2) if is_stable else float('inf')
    weight_norm = np.linalg.norm(fxlms.weights)

    return {
        'nr_db': nr_db if is_stable else -100.0,
        'mse_final': mse_final,
        'is_stable': is_stable,
        'weight_norm': weight_norm,
        'final_weights': fxlms.weights.copy(),
    }


def process_audio_file(filepath, paths, noise_gen=None):
    """Process one audio file into labeled training segments."""
    if filepath is not None:
        noise_source = noise_gen.load_audio_file(str(filepath))
    else:
        return []

    n_samples = len(noise_source)
    segment_samples = int(SEGMENT_DURATION * FS)
    stride_samples = int(STRIDE_DURATION * FS)

    samples = []
    current_weights = None

    for start in range(0, n_samples - segment_samples + 1, stride_samples):
        segment = noise_source[start:start + segment_samples]

        # Extract features from this segment
        # Use reference-path-filtered version for features
        ref_path = FIRPath(paths['reference'])
        ref_signal = np.array([ref_path.filter_sample(s) for s in segment])
        features = extract_features(ref_signal, fs=FS, n_features=16)

        # Run with each step size
        results_per_step = {}
        for step in STEP_SIZES:
            result = run_segment(segment, paths, step, initial_weights=current_weights)
            results_per_step[step] = result

        # Select best step size (highest NR among stable ones)
        best_step = None
        best_nr = -999
        for step, result in results_per_step.items():
            if result['is_stable'] and result['nr_db'] > best_nr:
                best_nr = result['nr_db']
                best_step = step

        if best_step is None:
            best_step = 0.001  # fallback to most conservative

        # Compute runtime features from the best run
        best_result = results_per_step[best_step]
        mse_final = best_result['mse_final']
        weight_norm = best_result['weight_norm']

        # Update weights for next segment (simulates continuous operation)
        current_weights = results_per_step[0.003]['final_weights']  # use moderate step for continuity

        sample = {
            'features': features.tolist(),
            'runtime_features': {
                'mse': float(mse_final) if not np.isinf(mse_final) else 1.0,
                'weight_norm': float(weight_norm),
                'segment_start_s': start / FS,
            },
            'label': best_step,
            'all_nr_db': {str(s): results_per_step[s]['nr_db'] for s in STEP_SIZES},
            'best_nr_db': best_nr,
        }
        samples.append(sample)

    return samples


def process_synthetic(paths, noise_gen, n_variations=50):
    """Generate training data from synthetic scenarios."""
    samples = []

    for scenario in SYNTHETIC_SCENARIOS:
        print(f"  Synthetic: {scenario}...", flush=True)
        for var in range(n_variations):
            noise_source = noise_gen.generate_scenario(
                duration=5.0, scenario=scenario
            )

            n_samples = len(noise_source)
            segment_samples = int(SEGMENT_DURATION * FS)
            stride_samples = int(STRIDE_DURATION * FS)
            current_weights = None

            for start in range(0, n_samples - segment_samples + 1, stride_samples):
                segment = noise_source[start:start + segment_samples]

                ref_path = FIRPath(paths['reference'])
                ref_signal = np.array([ref_path.filter_sample(s) for s in segment])
                features = extract_features(ref_signal, fs=FS, n_features=16)

                results_per_step = {}
                for step in STEP_SIZES:
                    result = run_segment(segment, paths, step, initial_weights=current_weights)
                    results_per_step[step] = result

                best_step = None
                best_nr = -999
                for step, result in results_per_step.items():
                    if result['is_stable'] and result['nr_db'] > best_nr:
                        best_nr = result['nr_db']
                        best_step = step

                if best_step is None:
                    best_step = 0.001

                best_result = results_per_step[best_step]
                current_weights = results_per_step[0.003]['final_weights']

                sample = {
                    'features': features.tolist(),
                    'runtime_features': {
                        'mse': float(best_result['mse_final']) if not np.isinf(best_result['mse_final']) else 1.0,
                        'weight_norm': float(best_result['weight_norm']),
                        'segment_start_s': start / FS,
                    },
                    'label': best_step,
                    'all_nr_db': {str(s): results_per_step[s]['nr_db'] for s in STEP_SIZES},
                    'best_nr_db': best_nr,
                    'source': f'synthetic_{scenario}_var{var}',
                }
                samples.append(sample)

    return samples


def main():
    print("=" * 60)
    print(" Phase 2: Collecting Dynamic Step Size Training Data")
    print("=" * 60)

    # Create room and extract acoustic paths
    print("\nCreating room and computing RIRs...", flush=True)
    paths = create_room()
    noise_gen = NoiseMixer(FS)

    all_samples = []

    # Process real audio files
    print(f"\nProcessing {len(REAL_AUDIO_FILES)} real audio files...")
    for filename in REAL_AUDIO_FILES:
        filepath = REAL_AUDIO_DIR / filename
        if not filepath.exists():
            print(f"  SKIP: {filename} (not found)")
            continue

        print(f"  {filename}...", end=" ", flush=True)
        samples = process_audio_file(filepath, paths, noise_gen)
        for s in samples:
            s['source'] = filename
        all_samples.extend(samples)
        print(f"{len(samples)} segments", flush=True)

    # Process synthetic scenarios
    print(f"\nProcessing synthetic scenarios (4 x 10 variations)...")
    synthetic_samples = process_synthetic(paths, noise_gen, n_variations=10)
    all_samples.extend(synthetic_samples)
    print(f"  {len(synthetic_samples)} synthetic segments")

    # Summary
    print(f"\n{'=' * 60}")
    print(f" Total training samples: {len(all_samples)}")

    # Label distribution
    from collections import Counter
    labels = Counter(s['label'] for s in all_samples)
    print(f"\n Label distribution:")
    for step in sorted(labels.keys()):
        print(f"   mu={step}: {labels[step]} samples ({100*labels[step]/len(all_samples):.1f}%)")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump({
            'metadata': {
                'n_samples': len(all_samples),
                'step_sizes': STEP_SIZES,
                'segment_duration': SEGMENT_DURATION,
                'stride_duration': STRIDE_DURATION,
                'filter_length': FILTER_LENGTH,
                'n_features': 16,
                'room_config': ROOM_CONFIG,
            },
            'samples': all_samples,
        }, f, indent=2)

    print(f"\n Saved to: {OUTPUT_PATH}")
    print(f" File size: {OUTPUT_PATH.stat().st_size / 1024:.0f} KB")


if __name__ == '__main__':
    main()
