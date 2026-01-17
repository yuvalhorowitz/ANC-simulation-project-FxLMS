"""
Phase 1: Collect Training Data for Step Size Selector

Runs FxNLMS simulations with different step sizes across various scenarios
and records the results to train the step size selector model.

Output: output/data/phase1/step_size_training_data.json
"""

import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pyroomacoustics as pra
from src.core.fxlms import FxNLMS
from src.acoustic.path_generator import FIRPath
from src.noise.noise_mixer import NoiseMixer
from src.ml.phase1_step_size.feature_extractor import extract_features
from src.ml.common.metrics import noise_reduction_db, convergence_time, stability_score


# =============================================================================
# Configuration
# =============================================================================

# Room configuration (sedan car)
ROOM_DIMS = [4.5, 1.85, 1.2]
ROOM_MATERIALS = {
    'ceiling': 0.38,
    'floor': 0.52,
    'east': 0.14,
    'west': 0.14,
    'north': 0.20,
    'south': 0.30,
}

# Positions (matching step8)
NOISE_SOURCE_POS = [0.3, 0.92, 0.4]
REF_MIC_POS = [0.3, 0.92, 0.5]
SPEAKER_POS = [1.9, 0.55, 1.0]
ERROR_MIC_POS = [1.8, 0.55, 1.0]

# Step sizes to test
STEP_SIZES = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]

# Scenarios to test
SCENARIOS = ['idle', 'city', 'highway']

# Number of variations per scenario
N_VARIATIONS = 10

# Simulation parameters
FS = 16000
DURATION = 3.0  # seconds
FILTER_LENGTH = 256


def create_room_simulation(fs: int = FS) -> Dict[str, np.ndarray]:
    """
    Create room and compute acoustic paths.

    Returns:
        Dictionary with primary, reference, and secondary path impulse responses
    """
    materials = {
        name: pra.Material(coef)
        for name, coef in ROOM_MATERIALS.items()
    }

    room = pra.ShoeBox(
        ROOM_DIMS,
        fs=fs,
        materials=materials,
        max_order=3,
        air_absorption=True
    )

    # Add noise source
    room.add_source(NOISE_SOURCE_POS)

    # Add speaker
    room.add_source(SPEAKER_POS)

    # Add microphones: [reference, error]
    mic_array = np.array([REF_MIC_POS, ERROR_MIC_POS]).T
    room.add_microphone_array(pra.MicrophoneArray(mic_array, fs=fs))

    # Compute RIRs
    room.compute_rir()

    max_len = 512

    return {
        'primary': room.rir[1][0][:max_len],      # noise -> error mic
        'reference': room.rir[0][0][:max_len],    # noise -> ref mic
        'secondary': room.rir[1][1][:max_len],    # speaker -> error mic
    }


def run_simulation(
    noise_signal: np.ndarray,
    paths: Dict[str, np.ndarray],
    step_size: float,
    filter_length: int = FILTER_LENGTH
) -> Dict[str, Any]:
    """
    Run a single ANC simulation with given step size.

    Returns:
        Dictionary with simulation results
    """
    n_samples = len(noise_signal)

    # Create path filters
    primary_path = FIRPath(paths['primary'])
    reference_path = FIRPath(paths['reference'])
    secondary_path = FIRPath(paths['secondary'])

    # Secondary path estimate with 5% error
    s_hat = paths['secondary'] * (1 + 0.05 * np.random.randn(len(paths['secondary'])))

    # Create FxNLMS
    fxnlms = FxNLMS(
        filter_length=filter_length,
        step_size=step_size,
        secondary_path_estimate=s_hat,
        regularization=1e-6
    )

    # Storage
    desired = np.zeros(n_samples)
    error = np.zeros(n_samples)

    # Simulation loop
    for i in range(n_samples):
        sample = noise_signal[i]

        # Reference signal
        x = reference_path.filter_sample(sample)

        # Noise at error mic (primary path)
        d = primary_path.filter_sample(sample)
        desired[i] = d

        # Generate anti-noise
        y = fxnlms.generate_antinoise(x)

        # Anti-noise through secondary path
        y_at_error = secondary_path.filter_sample(y)

        # Error signal
        e = d + y_at_error
        error[i] = e

        # Update weights
        fxnlms.filter_reference(x)
        fxnlms.update_weights(e)

    # Compute metrics
    nr_db = noise_reduction_db(desired, error)
    conv_time = convergence_time(fxnlms.mse_history)
    stable = stability_score(fxnlms.mse_history)

    return {
        'noise_reduction_db': float(nr_db),
        'convergence_time': int(conv_time),
        'stability_score': float(stable),
        'final_mse': float(fxnlms.mse_history[-1]) if fxnlms.mse_history else float('nan'),
        'desired': desired,
        'error': error,
        'mse_history': fxnlms.mse_history,
    }


def collect_data() -> List[Dict[str, Any]]:
    """
    Collect training data across all scenarios and step sizes.

    Returns:
        List of data samples
    """
    print("=" * 70)
    print("Phase 1: Collecting Training Data for Step Size Selector")
    print("=" * 70)

    # Create room simulation
    print("\nCreating room simulation...")
    paths = create_room_simulation()

    # Initialize noise generator
    noise_gen = NoiseMixer(FS)

    all_data = []
    total_runs = len(SCENARIOS) * len(STEP_SIZES) * N_VARIATIONS
    run_count = 0

    for scenario in SCENARIOS:
        print(f"\n{'='*50}")
        print(f"Scenario: {scenario}")
        print(f"{'='*50}")

        for variation in range(N_VARIATIONS):
            # Generate noise signal
            np.random.seed(42 + variation)
            noise_signal = noise_gen.generate_scenario(DURATION, scenario)

            # Extract features from first second
            features = extract_features(noise_signal[:FS], FS)

            # Test each step size
            results_for_scenario = []

            for step_size in STEP_SIZES:
                run_count += 1
                print(f"\r  [{run_count}/{total_runs}] "
                      f"Scenario={scenario}, Var={variation+1}/{N_VARIATIONS}, "
                      f"μ={step_size:.4f}", end="")

                result = run_simulation(noise_signal, paths, step_size)

                results_for_scenario.append({
                    'step_size': step_size,
                    'noise_reduction_db': result['noise_reduction_db'],
                    'convergence_time': result['convergence_time'],
                    'stability_score': result['stability_score'],
                })

            # Find best step size (highest NR with stability)
            stable_results = [
                r for r in results_for_scenario
                if r['stability_score'] > 0.5
            ]

            if stable_results:
                best = max(stable_results, key=lambda x: x['noise_reduction_db'])
                best_step_size = best['step_size']
            else:
                best_step_size = 0.005  # Fallback

            # Create training sample
            sample = {
                'scenario': scenario,
                'variation': variation,
                'features': features.tolist(),
                'best_step_size': best_step_size,
                'all_results': results_for_scenario,
            }
            all_data.append(sample)

    print("\n")
    return all_data


def save_data(data: List[Dict], output_path: Path):
    """Save collected data to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'step_sizes': STEP_SIZES,
            'scenarios': SCENARIOS,
            'n_variations': N_VARIATIONS,
            'filter_length': FILTER_LENGTH,
            'duration': DURATION,
            'fs': FS,
        },
        'samples': data,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(data)} samples to {output_path}")


def main():
    """Main entry point."""
    # Collect data
    data = collect_data()

    # Save to file
    output_path = Path('output/data/phase1/step_size_training_data.json')
    save_data(data, output_path)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for scenario in SCENARIOS:
        scenario_data = [d for d in data if d['scenario'] == scenario]
        best_sizes = [d['best_step_size'] for d in scenario_data]
        print(f"\n{scenario}:")
        print(f"  Samples: {len(scenario_data)}")
        print(f"  Best μ distribution: {dict(zip(*np.unique(best_sizes, return_counts=True)))}")

    print(f"\nTotal samples: {len(data)}")
    print(f"Output: {output_path}")


if __name__ == '__main__':
    main()
