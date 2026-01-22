"""
Simulation Runner for ANC Playground

Wraps the existing simulation classes to provide a clean interface for the GUI.
Supports both single-speaker and multi-speaker (4-speaker) ANC modes.
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pyroomacoustics as pra
from src.acoustic.path_generator import AcousticPathGenerator, FIRPath
from src.core.fxlms import FxNLMS
from src.noise.noise_mixer import NoiseMixer
from src.ml.common.metrics import convergence_time_90pct


class PlaygroundSimulation:
    """
    Simplified ANC simulation for the Playground GUI.
    Supports single-speaker mode.
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

        # Generate noise (handle dynamic ride separately)
        if scenario == 'dynamic ride':
            noise_source, scenario_order = self.noise_gen.generate_dynamic_scenario(duration)
        else:
            noise_source = self.noise_gen.generate_scenario(duration, scenario)
            scenario_order = None
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

        # Calculate convergence time (time to reach 90% of final reduction)
        conv_time = convergence_time_90pct(
            mse, sample_rate=self.fs,
            desired=desired_arr, error=error_arr
        )

        self.results = {
            'noise_source': noise_source,  # Raw noise signal
            'reference': np.array(reference),
            'desired': desired_arr,
            'antinoise': np.array(antinoise),
            'error': error_arr,
            'mse': np.array(mse),
            'noise_reduction_db': noise_reduction_db,
            'convergence_time': conv_time,
            'weights': self.fxlms.weights.copy(),
            'duration': duration,
            'fs': self.fs,
            'params': self.params,
            'scenario_order': scenario_order,  # For dynamic ride
        }

        return self.results


class MultiRefMicSimulation:
    """
    Multi-reference-mic ANC simulation for the Playground GUI.
    Multiple reference mics with signals averaged for FxLMS input.
    """

    def __init__(self, params: dict):
        """
        Initialize multi-reference-mic simulation.

        Args:
            params: Dictionary containing all simulation parameters
                   Must include 'ref_mics' dict with reference mic positions
        """
        self.params = params
        self.fs = params.get('sample_rate', 16000)
        self.ref_mics = params.get('ref_mics', {})

        if not self.ref_mics:
            raise ValueError("No reference mics defined for multi-ref-mic mode")

        self.ref_mic_names = list(self.ref_mics.keys())

        # Build room
        self.room = self._create_room()

        # Compute RIRs
        self.room.compute_rir()

        # Extract paths
        max_len = 512
        positions = params['positions']

        # Reference paths: noise -> each ref mic
        self.H_reference = {}
        for i, name in enumerate(self.ref_mic_names):
            self.H_reference[name] = self.room.rir[i][0][:max_len]

        # Primary path: noise -> error mic (last mic index)
        error_mic_idx = len(self.ref_mic_names)
        self.H_primary = self.room.rir[error_mic_idx][0][:max_len]

        # Secondary path: speaker -> error mic
        self.H_secondary = self.room.rir[error_mic_idx][1][:max_len]

        # Estimate with 5% error
        self.H_secondary_est = self.H_secondary * (
            1 + 0.05 * np.random.randn(len(self.H_secondary))
        )

        # Create FIR filters
        self.primary_path = FIRPath(self.H_primary)
        self.secondary_path = FIRPath(self.H_secondary)
        self.reference_paths = {
            name: FIRPath(self.H_reference[name]) for name in self.ref_mic_names
        }

        # Create FxNLMS
        self.fxlms = FxNLMS(
            filter_length=params['filter_length'],
            step_size=params['step_size'],
            secondary_path_estimate=self.H_secondary_est,
            regularization=1e-4
        )

        # Noise generator
        self.noise_gen = NoiseMixer(self.fs)

        self.results = {}

    def _create_room(self) -> pra.ShoeBox:
        """Create room with multiple reference mics."""
        dims = self.params['dimensions']
        absorption = self.params['absorption']
        max_order = self.params['max_order']
        positions = self.params['positions']

        materials = {
            'ceiling': pra.Material(absorption * 1.1),
            'floor': pra.Material(absorption * 1.5),
            'east': pra.Material(absorption * 0.5),
            'west': pra.Material(absorption * 0.5),
            'north': pra.Material(absorption * 0.7),
            'south': pra.Material(absorption * 0.9),
        }

        room = pra.ShoeBox(
            dims,
            fs=self.fs,
            materials=materials,
            max_order=max_order,
            air_absorption=True
        )

        # Add noise source (source 0)
        room.add_source(positions['noise_source'])

        # Add speaker (source 1)
        room.add_source(positions['speaker'])

        # Build mic array: all ref mics + error mic
        mic_positions = [self.ref_mics[name] for name in self.ref_mic_names]
        mic_positions.append(positions['error_mic'])
        mic_array = np.array(mic_positions).T
        room.add_microphone_array(pra.MicrophoneArray(mic_array, fs=self.fs))

        return room

    def run(self, progress_callback=None) -> dict:
        """Run multi-reference-mic ANC simulation."""
        duration = self.params['duration']
        scenario = self.params.get('scenario', 'highway')

        # Generate noise (handle dynamic ride separately)
        if scenario == 'dynamic ride':
            noise_source, scenario_order = self.noise_gen.generate_dynamic_scenario(duration)
        else:
            noise_source = self.noise_gen.generate_scenario(duration, scenario)
            scenario_order = None
        n_samples = len(noise_source)

        # Reset filters
        self.fxlms.reset()
        self.primary_path.reset()
        self.secondary_path.reset()
        for path in self.reference_paths.values():
            path.reset()

        # Storage
        reference = []
        desired = []
        antinoise = []
        error = []
        mse = []

        # Storage for individual reference mic signals
        ref_mic_signals = {name: [] for name in self.ref_mic_names}

        for i in range(n_samples):
            sample = noise_source[i]

            # Filter through all reference paths and AVERAGE
            ref_signals_sample = {}
            for name in self.ref_mic_names:
                ref_sig = self.reference_paths[name].filter_sample(sample)
                ref_signals_sample[name] = ref_sig
                ref_mic_signals[name].append(ref_sig)

            x = np.mean(list(ref_signals_sample.values()))  # Signal fusion: average
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

        # Calculate convergence time (time to reach 90% of final reduction)
        conv_time = convergence_time_90pct(
            mse, sample_rate=self.fs,
            desired=desired_arr, error=error_arr
        )

        # Convert individual ref mic signals to arrays
        ref_mic_signals_arr = {name: np.array(sig) for name, sig in ref_mic_signals.items()}

        self.results = {
            'noise_source': noise_source,  # Raw noise signal
            'reference': np.array(reference),
            'desired': desired_arr,
            'antinoise': np.array(antinoise),
            'error': error_arr,
            'mse': np.array(mse),
            'noise_reduction_db': noise_reduction_db,
            'convergence_time': conv_time,
            'weights': self.fxlms.weights.copy(),
            'duration': duration,
            'fs': self.fs,
            'params': self.params,
            'num_ref_mics': len(self.ref_mics),
            'ref_mic_names': list(self.ref_mics.keys()),
            'ref_mic_signals': ref_mic_signals_arr,  # Individual ref mic signals
            'scenario_order': scenario_order,  # For dynamic ride
        }

        return self.results


class MultiRefMicMultiSpeakerSimulation:
    """
    Multi-reference-mic + Multi-speaker ANC simulation.
    Combines both features: multiple ref mics (averaged) and multiple speakers.
    """

    def __init__(self, params: dict):
        """
        Initialize combined multi-ref-mic and multi-speaker simulation.

        Args:
            params: Dictionary containing all simulation parameters
        """
        self.params = params
        self.fs = params.get('sample_rate', 16000)
        self.ref_mics = params.get('ref_mics', {})
        self.speakers = params.get('speakers', {})

        if not self.ref_mics:
            raise ValueError("No reference mics defined")
        if not self.speakers:
            raise ValueError("No speakers defined")

        self.ref_mic_names = list(self.ref_mics.keys())
        self.speaker_names = list(self.speakers.keys())

        # Build room
        self.room = self._create_room()

        # Compute RIRs
        self.room.compute_rir()

        # Extract paths
        max_len = 512
        error_mic_idx = len(self.ref_mic_names)

        # Reference paths: noise -> each ref mic
        self.H_reference = {}
        for i, name in enumerate(self.ref_mic_names):
            self.H_reference[name] = self.room.rir[i][0][:max_len]

        # Primary path: noise -> error mic
        self.H_primary = self.room.rir[error_mic_idx][0][:max_len]

        # Secondary paths: each speaker -> error mic
        self.H_secondary = {}
        for i, name in enumerate(self.speaker_names):
            rir = self.room.rir[error_mic_idx][i + 1][:max_len]  # +1 because source 0 is noise
            self.H_secondary[name] = rir

        # Combined secondary path (sum of all speaker contributions)
        self.H_secondary_combined = np.zeros(max_len)
        for name in self.speaker_names:
            path = self.H_secondary[name]
            self.H_secondary_combined[:len(path)] += path

        # Estimate with 5% error
        self.H_secondary_est = self.H_secondary_combined * (
            1 + 0.05 * np.random.randn(len(self.H_secondary_combined))
        )

        # Create FIR filters
        self.primary_path = FIRPath(self.H_primary)
        self.reference_paths = {
            name: FIRPath(self.H_reference[name]) for name in self.ref_mic_names
        }
        self.secondary_paths = {
            name: FIRPath(self.H_secondary[name]) for name in self.speaker_names
        }

        # Create FxNLMS with combined secondary path
        self.fxlms = FxNLMS(
            filter_length=params['filter_length'],
            step_size=params['step_size'],
            secondary_path_estimate=self.H_secondary_est,
            regularization=1e-4
        )

        # Noise generator
        self.noise_gen = NoiseMixer(self.fs)

        self.results = {}

    def _create_room(self) -> pra.ShoeBox:
        """Create room with multiple ref mics and multiple speakers."""
        dims = self.params['dimensions']
        absorption = self.params['absorption']
        max_order = self.params['max_order']
        positions = self.params['positions']

        materials = {
            'ceiling': pra.Material(absorption * 1.1),
            'floor': pra.Material(absorption * 1.5),
            'east': pra.Material(absorption * 0.5),
            'west': pra.Material(absorption * 0.5),
            'north': pra.Material(absorption * 0.7),
            'south': pra.Material(absorption * 0.9),
        }

        room = pra.ShoeBox(
            dims,
            fs=self.fs,
            materials=materials,
            max_order=max_order,
            air_absorption=True
        )

        # Add noise source (source 0)
        room.add_source(positions['noise_source'])

        # Add all speakers (sources 1, 2, 3, ...)
        for name in self.speaker_names:
            room.add_source(self.speakers[name])

        # Build mic array: all ref mics + error mic
        mic_positions = [self.ref_mics[name] for name in self.ref_mic_names]
        mic_positions.append(positions['error_mic'])
        mic_array = np.array(mic_positions).T
        room.add_microphone_array(pra.MicrophoneArray(mic_array, fs=self.fs))

        return room

    def run(self, progress_callback=None) -> dict:
        """Run combined multi-ref-mic + multi-speaker ANC simulation."""
        duration = self.params['duration']
        scenario = self.params.get('scenario', 'highway')

        # Generate noise (handle dynamic ride separately)
        if scenario == 'dynamic ride':
            noise_source, scenario_order = self.noise_gen.generate_dynamic_scenario(duration)
        else:
            noise_source = self.noise_gen.generate_scenario(duration, scenario)
            scenario_order = None
        n_samples = len(noise_source)

        # Reset filters
        self.fxlms.reset()
        self.primary_path.reset()
        for path in self.reference_paths.values():
            path.reset()
        for path in self.secondary_paths.values():
            path.reset()

        # Storage
        reference = []
        desired = []
        antinoise = []
        error = []
        mse = []

        # Storage for individual reference mic signals
        ref_mic_signals = {name: [] for name in self.ref_mic_names}

        for i in range(n_samples):
            sample = noise_source[i]

            # Filter through all reference paths and AVERAGE
            ref_signals_sample = {}
            for name in self.ref_mic_names:
                ref_sig = self.reference_paths[name].filter_sample(sample)
                ref_signals_sample[name] = ref_sig
                ref_mic_signals[name].append(ref_sig)

            x = np.mean(list(ref_signals_sample.values()))  # Signal fusion: average
            reference.append(x)

            # Noise at error mic
            d = self.primary_path.filter_sample(sample)
            desired.append(d)

            # Generate anti-noise (same signal to all speakers)
            y = self.fxlms.generate_antinoise(x)
            antinoise.append(y)

            # Anti-noise through ALL secondary paths (sum contributions)
            y_at_error = 0.0
            for name in self.speaker_names:
                y_at_error += self.secondary_paths[name].filter_sample(y)

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

        # Calculate convergence time (time to reach 90% of final reduction)
        conv_time = convergence_time_90pct(
            mse, sample_rate=self.fs,
            desired=desired_arr, error=error_arr
        )

        # Convert individual ref mic signals to arrays
        ref_mic_signals_arr = {name: np.array(sig) for name, sig in ref_mic_signals.items()}

        self.results = {
            'noise_source': noise_source,  # Raw noise signal
            'reference': np.array(reference),
            'desired': desired_arr,
            'antinoise': np.array(antinoise),
            'error': error_arr,
            'mse': np.array(mse),
            'noise_reduction_db': noise_reduction_db,
            'convergence_time': conv_time,
            'weights': self.fxlms.weights.copy(),
            'duration': duration,
            'fs': self.fs,
            'params': self.params,
            'num_ref_mics': len(self.ref_mics),
            'ref_mic_names': list(self.ref_mics.keys()),
            'num_speakers': len(self.speakers),
            'speaker_names': list(self.speakers.keys()),
            'ref_mic_signals': ref_mic_signals_arr,  # Individual ref mic signals
            'scenario_order': scenario_order,  # For dynamic ride
        }

        return self.results


class MultiSpeakerSimulation:
    """
    Multi-speaker (4-speaker) ANC simulation for the Playground GUI.
    All speakers receive the same anti-noise signal.
    """

    def __init__(self, params: dict):
        """
        Initialize multi-speaker simulation.

        Args:
            params: Dictionary containing all simulation parameters
                   Must include 'speakers' dict with speaker positions
        """
        self.params = params
        self.fs = params.get('sample_rate', 16000)
        self.speakers = params.get('speakers', {})

        if not self.speakers:
            raise ValueError("No speakers defined for multi-speaker mode")

        # Build room with multiple speakers
        self.room = self._create_room()

        # Compute RIRs
        self.room.compute_rir()

        # Extract paths
        max_len = 512
        positions = params['positions']

        # Primary path: noise -> error mic (mic index 1)
        self.H_primary = self.room.rir[1][0][:max_len]

        # Reference path: noise -> reference mic (mic index 0)
        self.H_reference = self.room.rir[0][0][:max_len]

        # Secondary paths: each speaker -> error mic
        self.speaker_names = list(self.speakers.keys())
        self.H_secondary = {}
        for i, name in enumerate(self.speaker_names):
            rir = self.room.rir[1][i + 1][:max_len]  # +1 because source 0 is noise
            self.H_secondary[name] = rir

        # Combined secondary path (sum of all speaker contributions)
        self.H_secondary_combined = np.zeros(max_len)
        for name in self.speaker_names:
            path = self.H_secondary[name]
            self.H_secondary_combined[:len(path)] += path

        # Estimate with 5% error
        self.H_secondary_est = self.H_secondary_combined * (
            1 + 0.05 * np.random.randn(len(self.H_secondary_combined))
        )

        # Create FIR filters
        self.primary_path = FIRPath(self.H_primary)
        self.reference_path = FIRPath(self.H_reference)
        self.secondary_paths = {
            name: FIRPath(self.H_secondary[name]) for name in self.speaker_names
        }

        # Create FxNLMS with combined secondary path
        self.fxlms = FxNLMS(
            filter_length=params['filter_length'],
            step_size=params['step_size'],
            secondary_path_estimate=self.H_secondary_est,
            regularization=1e-4
        )

        # Noise generator
        self.noise_gen = NoiseMixer(self.fs)

        self.results = {}

    def _create_room(self) -> pra.ShoeBox:
        """Create room with multiple speakers."""
        dims = self.params['dimensions']
        absorption = self.params['absorption']
        max_order = self.params['max_order']
        positions = self.params['positions']

        materials = {
            'ceiling': pra.Material(absorption * 1.1),
            'floor': pra.Material(absorption * 1.5),
            'east': pra.Material(absorption * 0.5),
            'west': pra.Material(absorption * 0.5),
            'north': pra.Material(absorption * 0.7),
            'south': pra.Material(absorption * 0.9),
        }

        room = pra.ShoeBox(
            dims,
            fs=self.fs,
            materials=materials,
            max_order=max_order,
            air_absorption=True
        )

        # Add noise source (source 0)
        room.add_source(positions['noise_source'])

        # Add all speakers (sources 1, 2, 3, 4, ...)
        for name in self.speakers:
            room.add_source(self.speakers[name])

        # Add microphones: [0] = reference, [1] = error
        mic_array = np.array([
            positions['reference_mic'],
            positions['error_mic']
        ]).T
        room.add_microphone_array(pra.MicrophoneArray(mic_array, fs=self.fs))

        return room

    def run(self, progress_callback=None) -> dict:
        """Run multi-speaker ANC simulation."""
        duration = self.params['duration']
        scenario = self.params.get('scenario', 'highway')

        # Generate noise (handle dynamic ride separately)
        if scenario == 'dynamic ride':
            noise_source, scenario_order = self.noise_gen.generate_dynamic_scenario(duration)
        else:
            noise_source = self.noise_gen.generate_scenario(duration, scenario)
            scenario_order = None
        n_samples = len(noise_source)

        # Reset filters
        self.fxlms.reset()
        self.primary_path.reset()
        self.reference_path.reset()
        for path in self.secondary_paths.values():
            path.reset()

        # Storage
        reference = []
        desired = []
        antinoise = []
        error = []
        mse = []

        for i in range(n_samples):
            sample = noise_source[i]

            # Reference signal
            x = self.reference_path.filter_sample(sample)
            reference.append(x)

            # Noise at error mic
            d = self.primary_path.filter_sample(sample)
            desired.append(d)

            # Generate anti-noise (same signal to all speakers)
            y = self.fxlms.generate_antinoise(x)
            antinoise.append(y)

            # Anti-noise through ALL secondary paths (sum contributions)
            y_at_error = 0.0
            for name in self.speaker_names:
                y_at_error += self.secondary_paths[name].filter_sample(y)

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

        # Calculate convergence time (time to reach 90% of final reduction)
        conv_time = convergence_time_90pct(
            mse, sample_rate=self.fs,
            desired=desired_arr, error=error_arr
        )

        self.results = {
            'noise_source': noise_source,  # Raw noise signal
            'reference': np.array(reference),
            'desired': desired_arr,
            'antinoise': np.array(antinoise),
            'error': error_arr,
            'mse': np.array(mse),
            'noise_reduction_db': noise_reduction_db,
            'convergence_time': conv_time,
            'weights': self.fxlms.weights.copy(),
            'duration': duration,
            'fs': self.fs,
            'params': self.params,
            'num_speakers': len(self.speakers),
            'speaker_names': list(self.speakers.keys()),
            'scenario_order': scenario_order,  # For dynamic ride
        }

        return self.results


def run_simulation(params: dict, progress_callback=None) -> dict:
    """
    Convenience function to run a simulation.
    Automatically selects the appropriate simulation class based on params.

    Args:
        params: Parameter dictionary from GUI
        progress_callback: Optional progress callback

    Returns:
        Results dictionary
    """
    try:
        speaker_mode = params.get('speaker_mode', 'Single Speaker')
        ref_mic_mode = params.get('ref_mic_mode', 'Single Reference Mic')

        is_multi_speaker = speaker_mode == '4-Speaker System'
        is_multi_ref_mic = ref_mic_mode == '4-Reference Mic System'

        # Select appropriate simulation class
        if is_multi_ref_mic and is_multi_speaker:
            sim = MultiRefMicMultiSpeakerSimulation(params)
        elif is_multi_ref_mic:
            sim = MultiRefMicSimulation(params)
        elif is_multi_speaker:
            sim = MultiSpeakerSimulation(params)
        else:
            sim = PlaygroundSimulation(params)

        results = sim.run(progress_callback)
        results['success'] = True
        results['error_message'] = None
        results['speaker_mode'] = speaker_mode
        results['ref_mic_mode'] = ref_mic_mode
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
            'speaker_mode': params.get('speaker_mode', 'Single Speaker'),
            'ref_mic_mode': params.get('ref_mic_mode', 'Single Reference Mic'),
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

    # Validate main positions
    for name, pos in positions.items():
        for i, (coord, dim) in enumerate(zip(pos, dims)):
            if coord < 0.1 or coord > dim - 0.1:
                axis = ['x', 'y', 'z'][i]
                return False, f"{name} {axis}-coordinate ({coord:.2f}) is outside room bounds (0.1 to {dim-0.1:.2f})"

    # Validate speaker positions if in 4-speaker mode
    if params.get('speaker_mode') == '4-Speaker System' and 'speakers' in params:
        for name, pos in params['speakers'].items():
            for i, (coord, dim) in enumerate(zip(pos, dims)):
                if coord < 0.1 or coord > dim - 0.1:
                    axis = ['x', 'y', 'z'][i]
                    return False, f"Speaker {name} {axis}-coordinate ({coord:.2f}) is outside room bounds"

    # Validate ref mic positions if in 4-ref-mic mode
    if params.get('ref_mic_mode') == '4-Reference Mic System' and 'ref_mics' in params:
        for name, pos in params['ref_mics'].items():
            for i, (coord, dim) in enumerate(zip(pos, dims)):
                if coord < 0.1 or coord > dim - 0.1:
                    axis = ['x', 'y', 'z'][i]
                    return False, f"Ref mic {name} {axis}-coordinate ({coord:.2f}) is outside room bounds"

    return True, None
