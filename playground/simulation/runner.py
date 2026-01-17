"""
Simulation Runner for ANC Playground

Wraps the existing simulation classes to provide a clean interface for the GUI.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pyroomacoustics as pra
from src.acoustic.path_generator import AcousticPathGenerator, FIRPath
from src.core.fxlms import FxNLMS
from src.noise.noise_mixer import NoiseMixer


class PlaygroundSimulation:
    """
    Simplified ANC simulation for the Playground GUI.
    """

    def __init__(self, params: dict):
        """
        Initialize simulation with parameters from GUI.

        Args:
            params: Dictionary containing all simulation parameters
        """
        self.params = params
        self.fs = params.get('sample_rate', 16000)

        # Build room
        self.room = self._create_room()

        # Compute RIRs
        self.room.compute_rir()

        # Extract acoustic paths
        path_gen = AcousticPathGenerator(self.room)
        paths = path_gen.get_all_anc_paths(modeling_error=0.05)

        # Truncate paths for efficiency
        max_len = 512
        self.H_primary = paths['primary'][:max_len]
        self.H_secondary = paths['secondary'][:max_len]
        self.H_secondary_est = paths['secondary_estimate'][:max_len]
        self.H_reference = paths['reference'][:max_len]

        # Create FIR filters
        self.primary_path = FIRPath(self.H_primary)
        self.secondary_path = FIRPath(self.H_secondary)
        self.reference_path = FIRPath(self.H_reference)

        # Create FxNLMS
        self.fxlms = FxNLMS(
            filter_length=params['filter_length'],
            step_size=params['step_size'],
            secondary_path_estimate=self.H_secondary_est,
            regularization=1e-4
        )

        # Noise generator
        self.noise_gen = NoiseMixer(self.fs)

        # Results storage
        self.results = {}

    def _create_room(self) -> pra.ShoeBox:
        """Create pyroomacoustics room from parameters."""
        dims = self.params['dimensions']
        absorption = self.params['absorption']
        max_order = self.params['max_order']
        positions = self.params['positions']

        # Create materials for each wall
        materials = {
            'ceiling': pra.Material(absorption * 1.1),  # Slightly more absorbent
            'floor': pra.Material(absorption * 1.5),    # Carpet/floor more absorbent
            'east': pra.Material(absorption * 0.5),     # Windows less absorbent
            'west': pra.Material(absorption * 0.5),
            'north': pra.Material(absorption * 0.7),    # Dashboard
            'south': pra.Material(absorption * 0.9),    # Rear seats
        }

        room = pra.ShoeBox(
            dims,
            fs=self.fs,
            materials=materials,
            max_order=max_order,
            air_absorption=True
        )

        # Add sources and microphones
        room.add_source(positions['noise_source'])   # Source 0: Noise
        room.add_source(positions['speaker'])        # Source 1: Speaker

        mic_array = np.array([
            positions['reference_mic'],
            positions['error_mic']
        ]).T
        room.add_microphone_array(pra.MicrophoneArray(mic_array, fs=self.fs))

        return room

    def run(self, progress_callback=None) -> dict:
        """
        Run the ANC simulation.

        Args:
            progress_callback: Optional callback function(progress, mse) for progress updates

        Returns:
            Results dictionary with all signals and metrics
        """
        duration = self.params['duration']
        scenario = self.params.get('scenario', 'highway')

        # Generate noise
        noise_source = self.noise_gen.generate_scenario(duration, scenario)
        n_samples = len(noise_source)

        # Reset filters
        self.fxlms.reset()
        self.primary_path.reset()
        self.secondary_path.reset()
        self.reference_path.reset()

        # Storage
        reference = []
        desired = []
        antinoise = []
        error = []
        mse = []

        # Process samples
        for i in range(n_samples):
            sample = noise_source[i]

            # Reference signal
            x = self.reference_path.filter_sample(sample)
            reference.append(x)

            # Noise at error mic
            d = self.primary_path.filter_sample(sample)
            desired.append(d)

            # Generate anti-noise
            y = self.fxlms.generate_antinoise(x)
            antinoise.append(y)

            # Anti-noise through secondary path
            y_at_error = self.secondary_path.filter_sample(y)

            # Error
            e = d + y_at_error
            error.append(e)
            mse.append(e ** 2)

            # Update FxLMS
            self.fxlms.filter_reference(x)
            self.fxlms.update_weights(e)

            # Progress callback
            if progress_callback and (i + 1) % (n_samples // 20) == 0:
                progress = (i + 1) / n_samples
                current_mse = np.mean(mse[-1000:]) if len(mse) > 1000 else np.mean(mse)
                progress_callback(progress, current_mse)

        # Calculate noise reduction
        desired_arr = np.array(desired)
        error_arr = np.array(error)

        steady_start = len(desired_arr) // 2
        d_power = np.mean(desired_arr[steady_start:] ** 2)
        e_power = np.mean(error_arr[steady_start:] ** 2)

        if e_power > 1e-10:
            noise_reduction_db = 10 * np.log10(d_power / e_power)
        else:
            noise_reduction_db = 60.0

        self.results = {
            'reference': np.array(reference),
            'desired': desired_arr,
            'antinoise': np.array(antinoise),
            'error': error_arr,
            'mse': np.array(mse),
            'noise_reduction_db': noise_reduction_db,
            'weights': self.fxlms.weights.copy(),
            'duration': duration,
            'fs': self.fs,
            'params': self.params,
        }

        return self.results


def run_simulation(params: dict, progress_callback=None) -> dict:
    """
    Convenience function to run a simulation.

    Args:
        params: Parameter dictionary from GUI
        progress_callback: Optional progress callback

    Returns:
        Results dictionary
    """
    try:
        sim = PlaygroundSimulation(params)
        results = sim.run(progress_callback)
        results['success'] = True
        results['error_message'] = None
        return results
    except Exception as e:
        return {
            'success': False,
            'error_message': str(e),
            'noise_reduction_db': 0,
            'desired': np.zeros(100),
            'error': np.zeros(100),
            'mse': np.ones(100),
            'weights': np.zeros(params.get('filter_length', 256)),
            'fs': params.get('sample_rate', 16000),
        }


def validate_positions(params: dict) -> tuple:
    """
    Validate that all positions are within the room dimensions.

    Args:
        params: Parameter dictionary

    Returns:
        Tuple of (is_valid, error_message)
    """
    dims = params['dimensions']
    positions = params['positions']

    for name, pos in positions.items():
        for i, (coord, dim) in enumerate(zip(pos, dims)):
            if coord < 0.1 or coord > dim - 0.1:
                axis = ['x', 'y', 'z'][i]
                return False, f"{name} {axis}-coordinate ({coord:.2f}) is outside room bounds (0.1 to {dim-0.1:.2f})"

    return True, None
