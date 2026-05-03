"""
Phase 1: Collect Training Data for Step Size Selector

Runs FxNLMS simulations with different step sizes across various scenarios
and records the results to train the step size selector model.

Version 2: Updated for regression model with:
- 4 scenarios (idle, city, highway, acceleration)
- Realistic amplitude scaling
- Multi-objective selection (NR + convergence time)
- T90 convergence metric

Output: output/data/phase1/step_size_training_data.json
"""

import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pyroomacoustics as pra
from src.core.fxlms import FxNLMS
from src.acoustic.path_generator import FIRPath
from src.noise.noise_mixer import NoiseMixer
from src.ml.phase1_step_size.feature_extractor import extract_features, N_FEATURES
from src.ml.common.metrics import (
    noise_reduction_db, convergence_time, stability_score,
    convergence_time_90pct
)


def select_best_step_size_v2(
    results: List[Dict],
    nr_weight: float = 0.5,
    conv_weight: float = 0.5
) -> Tuple[float, float]:
    """
    Select best μ using multi-objective optimization for NR and convergence.

    Balances noise reduction with convergence speed using T90 metric.
    This prevents bias toward slow but high-NR step sizes and encourages
    learning scenario-specific patterns that also converge quickly.

    Args:
        results: List of dicts with step_size, noise_reduction_db,
                 convergence_time_90pct, stability_score
        nr_weight: Weight for noise reduction (default 0.5)
        conv_weight: Weight for convergence speed (default 0.5)

    Returns:
        Tuple of (best_step_size, best_convergence_time_90pct)
    """
    # Filter for highly stable runs (stability > 0.8)
    stable = [r for r in results if r['stability_score'] > 0.8]

    if not stable:
        # If nothing is highly stable, use moderate stability threshold
        stable = [r for r in results if r['stability_score'] > 0.5]

    if not stable:
        # If still nothing stable, return conservative default
        return 0.005, float('inf')

    # Normalize metrics to [0, 1] scale
    nr_vals = [r['noise_reduction_db'] for r in stable]
    conv_vals = [r['convergence_time_90pct'] for r in stable]

    nr_min, nr_max = min(nr_vals), max(nr_vals)
    conv_min, conv_max = min(conv_vals), max(conv_vals)

    # Avoid division by zero
    nr_range = nr_max - nr_min if nr_max > nr_min else 1.0
    conv_range = conv_max - conv_min if conv_max > conv_min else 1.0

    # Normalize: higher is better for both metrics
    nr_norm = [(v - nr_min) / nr_range for v in nr_vals]
    conv_norm = [(conv_max - v) / conv_range for v in conv_vals]  # Invert: lower time = better

    # Combined score
    scores = [nr_weight * nr + conv_weight * conv
              for nr, conv in zip(nr_norm, conv_norm)]

    # Select step size with highest combined score
    best_idx = np.argmax(scores)
    best_result = stable[best_idx]

    return best_result['step_size'], best_result['convergence_time_90pct']


# Legacy function for backward compatibility
def select_best_step_size(results: List[Dict]) -> float:
    """Legacy: Select best μ (for backward compatibility)."""
    best_mu, _ = select_best_step_size_v2(results)
    return best_mu


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

# Multi-channel configuration (matching playground presets.py)
FOUR_SPEAKERS = {
    'door_L': [2.0, 0.1, 0.4],       # Front left door
    'door_R': [2.0, 1.75, 0.4],      # Front right door
    'dash_L': [0.8, 0.25, 0.9],      # Dashboard left
    'dash_R': [0.8, 1.60, 0.9],      # Dashboard right
}

FOUR_REF_MICS = {
    'firewall': [0.3, 0.92, 0.5],    # Engine noise detection
    'floor': [2.0, 0.55, 0.15],      # Road/tire noise
    'a_pillar': [0.5, 0.15, 1.0],    # Wind noise
    'dashboard': [0.9, 0.92, 0.8],   # General
}

ERROR_MIC_POS = [1.8, 0.55, 1.0]  # Driver's ear

SCENARIO_NOISE_POSITIONS = {
    'idle': [0.15, 0.92, 0.5],       # Engine (Firewall)
    'city': [0.5, 0.92, 0.5],        # Combined (Dashboard)
    'highway': [2.0, 0.92, 0.12],    # Road (Floor)
    'acceleration': [0.15, 0.92, 0.5], # Engine (Firewall)
}

# Step sizes to test (5 values that occur as optimal in practice)
STEP_SIZES = [0.003, 0.005, 0.007, 0.01, 0.015]

# Scenarios to test (now includes acceleration)
SCENARIOS = ['idle', 'city', 'highway', 'acceleration']

# Number of variations per scenario (150 × 4 = 600 total samples)
N_VARIATIONS = 150

# Simulation parameters
FS = 16000
DURATION = 3.0  # seconds
FILTER_LENGTH = 256


def create_room_simulation_multi_channel(
    scenario: str,
    fs: int = FS
) -> Dict[str, Any]:
    """
    Create room with 4 speakers + 4 ref mics for given scenario.

    Args:
        scenario: Scenario name ('idle', 'city', 'highway', 'acceleration')
        fs: Sample rate

    Returns:
        Dictionary with:
        - primary: noise -> error mic
        - reference_paths: dict of 4 RIRs (noise -> each ref mic)
        - secondary: combined 4 speaker paths
        - ref_mic_names: list of reference mic names
        - speaker_names: list of speaker names
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

    # Scenario-specific noise source
    room.add_source(SCENARIO_NOISE_POSITIONS[scenario])

    # Add all 4 speakers
    speaker_names = list(FOUR_SPEAKERS.keys())
    for name in speaker_names:
        room.add_source(FOUR_SPEAKERS[name])

    # Build mic array: 4 ref mics + 1 error mic
    ref_mic_names = list(FOUR_REF_MICS.keys())
    mic_positions = [FOUR_REF_MICS[name] for name in ref_mic_names]
    mic_positions.append(ERROR_MIC_POS)
    mic_array = np.array(mic_positions).T
    room.add_microphone_array(pra.MicrophoneArray(mic_array, fs=fs))

    room.compute_rir()
    max_len = 512

    error_mic_idx = len(ref_mic_names)  # Index 4

    # Primary path
    primary = room.rir[error_mic_idx][0][:max_len]

    # Reference paths (will be averaged)
    reference_paths = {
        name: room.rir[i][0][:max_len]
        for i, name in enumerate(ref_mic_names)
    }

    # Secondary paths (sum of all 4 speakers)
    secondary_combined = np.zeros(max_len)
    for i, name in enumerate(speaker_names):
        rir = room.rir[error_mic_idx][i + 1][:max_len]
        secondary_combined[:len(rir)] += rir

    return {
        'primary': primary,
        'reference_paths': reference_paths,
        'secondary': secondary_combined,
        'speaker_names': speaker_names,
        'ref_mic_names': ref_mic_names,
    }


def run_simulation(
    noise_signal: np.ndarray,
    paths: Dict[str, Any],
    step_size: float,
    filter_length: int = FILTER_LENGTH
) -> Dict[str, Any]:
    """
    Run a single ANC simulation with given step size (multi-channel).

    Returns:
        Dictionary with simulation results
    """
    n_samples = len(noise_signal)

    # Create path filters
    primary_path = FIRPath(paths['primary'])

    # Create FIR filter for each reference mic
    reference_path_filters = {
        name: FIRPath(paths['reference_paths'][name])
        for name in paths['ref_mic_names']
    }

    secondary_path = FIRPath(paths['secondary'])

    # Secondary path estimate with 5% error
    s_hat = paths['secondary'] * (1 + 0.05 * np.random.randn(len(paths['secondary'])))

    # Create FxNLMS - CRITICAL: regularization changed to 1e-4 (matching playground)
    fxnlms = FxNLMS(
        filter_length=filter_length,
        step_size=step_size,
        secondary_path_estimate=s_hat,
        regularization=1e-4  # Changed from 1e-6
    )

    # Storage
    desired = np.zeros(n_samples)
    error = np.zeros(n_samples)

    # Simulation loop
    for i in range(n_samples):
        sample = noise_signal[i]

        # AVERAGE 4 reference signals (signal fusion)
        ref_signals = {}
        for name in paths['ref_mic_names']:
            ref_signals[name] = reference_path_filters[name].filter_sample(sample)
        x = np.mean(list(ref_signals.values()))

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

    # Compute T90 convergence time (time to reach 90% of final reduction)
    conv_time_90 = convergence_time_90pct(
        fxnlms.mse_history,
        sample_rate=FS,
        desired=desired,
        error=error
    )

    return {
        'noise_reduction_db': float(nr_db),
        'convergence_time': int(conv_time),
        'convergence_time_90pct': float(conv_time_90),
        'stability_score': float(stable),
        'final_mse': float(fxnlms.mse_history[-1]) if fxnlms.mse_history else float('nan'),
        'desired': desired,
        'error': error,
        'mse_history': fxnlms.mse_history,
    }


def collect_data() -> List[Dict[str, Any]]:
    """
    Collect training data across all scenarios and step sizes (multi-channel).

    Returns:
        List of data samples
    """
    print("=" * 70)
    print("Phase 1: Multi-Channel Training Data Collection")
    print("=" * 70)
    print(f"  4 speakers: {list(FOUR_SPEAKERS.keys())}")
    print(f"  4 ref mics: {list(FOUR_REF_MICS.keys())} (averaged)")
    print(f"  Scenario-specific noise positions")
    print(f"  Regularization: 1e-4")

    # Initialize noise generator
    noise_gen = NoiseMixer(FS)

    all_data = []
    total_runs = len(SCENARIOS) * len(STEP_SIZES) * N_VARIATIONS
    run_count = 0

    for scenario in SCENARIOS:
        print(f"\n{'='*50}")
        print(f"Scenario: {scenario} | Noise at: {SCENARIO_NOISE_POSITIONS[scenario]}")
        print(f"{'='*50}")

        # Create scenario-specific room
        paths = create_room_simulation_multi_channel(scenario)

        for variation in range(N_VARIATIONS):
            # Generate noise signal
            np.random.seed(42 + variation)
            noise_signal = noise_gen.generate_scenario(DURATION, scenario)

            # Extract features from AVERAGED reference signal
            # (matching what the model sees during deployment)
            ref_path_filters = {
                name: FIRPath(paths['reference_paths'][name])
                for name in paths['ref_mic_names']
            }

            # Collect first second from all 4 ref mics
            ref_signals_1s = {name: [] for name in paths['ref_mic_names']}
            for i in range(min(FS, len(noise_signal))):
                sample = noise_signal[i]
                for name in paths['ref_mic_names']:
                    sig = ref_path_filters[name].filter_sample(sample)
                    ref_signals_1s[name].append(sig)

            # Average the 4 reference signals
            ref_signal_averaged = np.mean([
                np.array(ref_signals_1s[name])
                for name in paths['ref_mic_names']
            ], axis=0)

            # Extract 12 features from averaged signal
            features = extract_features(ref_signal_averaged, FS)

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
                    'convergence_time_90pct': result['convergence_time_90pct'],
                    'stability_score': result['stability_score'],
                })

            # Find best step size using multi-objective optimization (NR + convergence)
            best_step_size, best_conv_time = select_best_step_size_v2(results_for_scenario)

            # Create training sample
            sample = {
                'scenario': scenario,
                'variation': variation,
                'features': features.tolist(),
                'best_step_size': best_step_size,
                'best_convergence_time_90pct': best_conv_time,
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
        'version': 3,  # v3: multi-channel with scenario-specific positions
        'config': {
            'step_sizes': STEP_SIZES,
            'scenarios': SCENARIOS,
            'n_variations': N_VARIATIONS,
            'filter_length': FILTER_LENGTH,
            'duration': DURATION,
            'fs': FS,
            'n_features': N_FEATURES,
            'selection_method': 'multi_objective_nr_conv',
            'num_speakers': 4,
            'num_ref_mics': 4,
            'regularization': 1e-4,
            'speaker_positions': FOUR_SPEAKERS,
            'ref_mic_positions': FOUR_REF_MICS,
            'scenario_noise_positions': SCENARIO_NOISE_POSITIONS,
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
        conv_times = [d['best_convergence_time_90pct'] for d in scenario_data
                      if d['best_convergence_time_90pct'] != float('inf')]
        print(f"\n{scenario}:")
        print(f"  Samples: {len(scenario_data)}")
        print(f"  Best μ distribution: {dict(zip(*np.unique(best_sizes, return_counts=True)))}")
        if conv_times:
            print(f"  Avg T90 convergence: {np.mean(conv_times):.2f}s")

    print(f"\nTotal samples: {len(data)}")
    print(f"Features: {N_FEATURES}")
    print(f"Output: {output_path}")


if __name__ == '__main__':
    main()
