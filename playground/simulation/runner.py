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


def _generate_noise(noise_gen, params):
    """
    Generate or load noise signal based on params.
    Returns (noise_source, scenario_order).
    For real audio: scenario_order is None.
    For dynamic ride: scenario_order is the list of scenario names.
    """
    audio_file = params.get('audio_file')
    if audio_file:
        noise_source = noise_gen.load_audio_file(audio_file, duration=params['duration'])
        return noise_source, None

    scenario = params.get('scenario', 'highway')
    if scenario == 'dynamic ride':
        noise_source, scenario_order = noise_gen.generate_dynamic_scenario(params['duration'])
        return noise_source, scenario_order
    else:
        noise_source = noise_gen.generate_scenario(params['duration'], scenario)
        return noise_source, None


WEIGHT_SNAPSHOT_INTERVAL_S = 0.5


def _create_fxlms(params, secondary_path_estimate, sample_rate):
    """Create FxNLMS adaptive filter from params."""
    return FxNLMS(
        filter_length=params['filter_length'],
        step_size=params['step_size'],
        secondary_path_estimate=secondary_path_estimate,
        regularization=1e-4,
        leakage=params.get('leakage', 0.0)
    )


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
        self.fxlms = _create_fxlms(params, self.H_secondary_est, self.fs)

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
        noise_source, scenario_order = _generate_noise(self.noise_gen, self.params)
        n_samples = len(noise_source)
        duration = n_samples / self.fs

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
        weights_history = []
        snapshot_interval = int(WEIGHT_SNAPSHOT_INTERVAL_S * self.fs)

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

            if (i + 1) % snapshot_interval == 0:
                weights_history.append(self.fxlms.weights.copy())

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
            'noise_source': noise_source,
            'reference': np.array(reference),
            'desired': desired_arr,
            'antinoise': np.array(antinoise),
            'error': error_arr,
            'mse': np.array(mse),
            'noise_reduction_db': noise_reduction_db,
            'convergence_time': conv_time,
            'weights': self.fxlms.weights.copy(),
            'weights_history': np.array(weights_history),
            'duration': duration,
            'fs': self.fs,
            'params': self.params,
            'scenario_order': scenario_order,
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
        self.fxlms = _create_fxlms(params, self.H_secondary_est, self.fs)

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
        noise_source, scenario_order = _generate_noise(self.noise_gen, self.params)
        n_samples = len(noise_source)
        duration = n_samples / self.fs

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
        weights_history = []
        snapshot_interval = int(WEIGHT_SNAPSHOT_INTERVAL_S * self.fs)

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

            if (i + 1) % snapshot_interval == 0:
                weights_history.append(self.fxlms.weights.copy())

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
            'noise_source': noise_source,
            'reference': np.array(reference),
            'desired': desired_arr,
            'antinoise': np.array(antinoise),
            'error': error_arr,
            'mse': np.array(mse),
            'noise_reduction_db': noise_reduction_db,
            'convergence_time': conv_time,
            'weights': self.fxlms.weights.copy(),
            'weights_history': np.array(weights_history),
            'duration': duration,
            'fs': self.fs,
            'params': self.params,
            'num_ref_mics': len(self.ref_mics),
            'ref_mic_names': list(self.ref_mics.keys()),
            'ref_mic_signals': ref_mic_signals_arr,
            'scenario_order': scenario_order,
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
        self.fxlms = _create_fxlms(params, self.H_secondary_est, self.fs)

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
        noise_source, scenario_order = _generate_noise(self.noise_gen, self.params)
        n_samples = len(noise_source)
        duration = n_samples / self.fs

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
        weights_history = []
        snapshot_interval = int(WEIGHT_SNAPSHOT_INTERVAL_S * self.fs)

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

            if (i + 1) % snapshot_interval == 0:
                weights_history.append(self.fxlms.weights.copy())

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
            'noise_source': noise_source,
            'reference': np.array(reference),
            'desired': desired_arr,
            'antinoise': np.array(antinoise),
            'error': error_arr,
            'mse': np.array(mse),
            'noise_reduction_db': noise_reduction_db,
            'convergence_time': conv_time,
            'weights': self.fxlms.weights.copy(),
            'weights_history': np.array(weights_history),
            'duration': duration,
            'fs': self.fs,
            'params': self.params,
            'num_ref_mics': len(self.ref_mics),
            'ref_mic_names': list(self.ref_mics.keys()),
            'num_speakers': len(self.speakers),
            'speaker_names': list(self.speakers.keys()),
            'ref_mic_signals': ref_mic_signals_arr,
            'scenario_order': scenario_order,
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
        self.fxlms = _create_fxlms(params, self.H_secondary_est, self.fs)

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
        noise_source, scenario_order = _generate_noise(self.noise_gen, self.params)
        n_samples = len(noise_source)
        duration = n_samples / self.fs

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
        weights_history = []
        snapshot_interval = int(WEIGHT_SNAPSHOT_INTERVAL_S * self.fs)

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

            if (i + 1) % snapshot_interval == 0:
                weights_history.append(self.fxlms.weights.copy())

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
            'noise_source': noise_source,
            'reference': np.array(reference),
            'desired': desired_arr,
            'antinoise': np.array(antinoise),
            'error': error_arr,
            'mse': np.array(mse),
            'noise_reduction_db': noise_reduction_db,
            'convergence_time': conv_time,
            'weights': self.fxlms.weights.copy(),
            'weights_history': np.array(weights_history),
            'duration': duration,
            'fs': self.fs,
            'params': self.params,
            'num_speakers': len(self.speakers),
            'speaker_names': list(self.speakers.keys()),
            'scenario_order': scenario_order,
        }

        return self.results


# Scenario to noise source position mapping (matching presets.py)
SCENARIO_NOISE_POSITIONS = {
    'idle': [0.15, 0.92, 0.5],        # Engine (Firewall)
    'city': [0.5, 0.92, 0.5],         # Combined (Dashboard)
    'highway': [2.0, 0.92, 0.12],     # Road (Floor)
    'acceleration': [0.15, 0.92, 0.5], # Engine (Firewall)
}


class DynamicRideSimulation:
    """
    Dynamic Ride simulation with multiple noise source positions.

    Pre-computes RIRs for all 4 noise positions and switches between them
    as the scenario changes, with crossfade transitions.
    """

    def __init__(self, params: dict):
        """
        Initialize dynamic ride simulation.

        Args:
            params: Dictionary containing all simulation parameters
        """
        self.params = params
        self.fs = params.get('sample_rate', 16000)
        self.crossfade_ms = 100.0  # Crossfade duration in ms

        # Build room with all noise source positions
        self.room = self._create_room()

        # Compute RIRs
        self.room.compute_rir()

        # Extract acoustic paths for each noise source position
        max_len = 512
        positions = params['positions']

        # Source indices: 0=idle/engine, 1=city/combined, 2=highway/road, 3=acceleration/engine, 4=speaker
        # Note: idle and acceleration share the same position (engine/firewall)
        self.scenario_source_idx = {
            'idle': 0,
            'city': 1,
            'highway': 2,
            'acceleration': 3,
        }

        # Primary paths: each noise source -> error mic (mic index 1)
        self.H_primary = {}
        for scenario, src_idx in self.scenario_source_idx.items():
            self.H_primary[scenario] = self.room.rir[1][src_idx][:max_len]

        # Reference paths: each noise source -> reference mic (mic index 0)
        self.H_reference = {}
        for scenario, src_idx in self.scenario_source_idx.items():
            self.H_reference[scenario] = self.room.rir[0][src_idx][:max_len]

        # Secondary path: speaker -> error mic (speaker is source 4)
        self.H_secondary = self.room.rir[1][4][:max_len]

        # Estimate with 5% error
        self.H_secondary_est = self.H_secondary * (
            1 + 0.05 * np.random.randn(len(self.H_secondary))
        )

        # Create FIR filters for each scenario
        self.primary_paths = {
            scenario: FIRPath(self.H_primary[scenario])
            for scenario in self.scenario_source_idx.keys()
        }
        self.reference_paths = {
            scenario: FIRPath(self.H_reference[scenario])
            for scenario in self.scenario_source_idx.keys()
        }
        self.secondary_path = FIRPath(self.H_secondary)

        # Create FxNLMS
        self.fxlms = _create_fxlms(params, self.H_secondary_est, self.fs)

        # Noise generator
        self.noise_gen = NoiseMixer(self.fs)

        self.results = {}

    def _create_room(self) -> pra.ShoeBox:
        """Create room with all noise source positions."""
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

        # Add all noise source positions as separate sources
        # Source 0: Idle (Engine/Firewall)
        room.add_source(SCENARIO_NOISE_POSITIONS['idle'])
        # Source 1: City (Combined/Dashboard)
        room.add_source(SCENARIO_NOISE_POSITIONS['city'])
        # Source 2: Highway (Road/Floor)
        room.add_source(SCENARIO_NOISE_POSITIONS['highway'])
        # Source 3: Acceleration (Engine/Firewall - same as idle)
        room.add_source(SCENARIO_NOISE_POSITIONS['acceleration'])
        # Source 4: Speaker
        room.add_source(positions['speaker'])

        # Add microphones: [0] = reference, [1] = error
        mic_array = np.array([
            positions['reference_mic'],
            positions['error_mic']
        ]).T
        room.add_microphone_array(pra.MicrophoneArray(mic_array, fs=self.fs))

        return room

    def run(self, progress_callback=None) -> dict:
        """Run dynamic ride simulation with position switching."""
        duration = self.params['duration']
        crossfade_samples = int(self.crossfade_ms / 1000.0 * self.fs)

        # Generate dynamic noise with crossfade
        noise_source, scenario_order, segment_boundaries = \
            self.noise_gen.generate_dynamic_scenario(duration, crossfade_ms=self.crossfade_ms)
        n_samples = len(noise_source)

        # Calculate segment length (accounting for crossfade reduction)
        segment_len = n_samples // len(scenario_order)

        # Reset filters
        self.fxlms.reset()
        self.secondary_path.reset()
        for path in self.primary_paths.values():
            path.reset()
        for path in self.reference_paths.values():
            path.reset()

        # Storage
        reference = []
        desired = []
        antinoise = []
        error = []
        mse = []
        weights_history = []
        snapshot_interval = int(WEIGHT_SNAPSHOT_INTERVAL_S * self.fs)

        # Track current and next scenario for crossfade
        def get_scenario_at_sample(sample_idx):
            """Get the scenario name for a given sample index."""
            for i, boundary in enumerate(segment_boundaries):
                if i == len(segment_boundaries) - 1:
                    return scenario_order[i]
                if sample_idx < segment_boundaries[i + 1]:
                    return scenario_order[i]
            return scenario_order[-1]

        def get_crossfade_weight(sample_idx):
            """Get crossfade weight (0-1) if in transition zone, else None."""
            for i in range(1, len(segment_boundaries)):
                boundary = segment_boundaries[i]
                adjusted_boundary = boundary - (i * crossfade_samples)
                if adjusted_boundary - crossfade_samples <= sample_idx < adjusted_boundary:
                    progress = (sample_idx - (adjusted_boundary - crossfade_samples)) / crossfade_samples
                    return progress, scenario_order[i-1], scenario_order[i]
            return None

        # Process samples
        for i in range(n_samples):
            sample = noise_source[i]

            crossfade_info = get_crossfade_weight(i)

            if crossfade_info is not None:
                weight, from_scenario, to_scenario = crossfade_info
                x_from = self.reference_paths[from_scenario].filter_sample(sample)
                x_to = self.reference_paths[to_scenario].filter_sample(sample)
                x = (1 - weight) * x_from + weight * x_to
                d_from = self.primary_paths[from_scenario].filter_sample(sample)
                d_to = self.primary_paths[to_scenario].filter_sample(sample)
                d = (1 - weight) * d_from + weight * d_to
            else:
                current_scenario = get_scenario_at_sample(i)
                x = self.reference_paths[current_scenario].filter_sample(sample)
                d = self.primary_paths[current_scenario].filter_sample(sample)

            reference.append(x)
            desired.append(d)

            y = self.fxlms.generate_antinoise(x)
            antinoise.append(y)

            y_at_error = self.secondary_path.filter_sample(y)

            e = d + y_at_error
            error.append(e)
            mse.append(e ** 2)

            self.fxlms.filter_reference(x)
            self.fxlms.update_weights(e)

            if (i + 1) % snapshot_interval == 0:
                weights_history.append(self.fxlms.weights.copy())

            if progress_callback and (i + 1) % (n_samples // 20) == 0:
                progress = (i + 1) / n_samples
                current_mse = np.mean(mse[-1000:]) if len(mse) > 1000 else np.mean(mse)
                progress_callback(progress, current_mse)

        desired_arr = np.array(desired)
        error_arr = np.array(error)

        steady_start = len(desired_arr) // 2
        d_power = np.mean(desired_arr[steady_start:] ** 2)
        e_power = np.mean(error_arr[steady_start:] ** 2)

        if e_power > 1e-10:
            noise_reduction_db = 10 * np.log10(d_power / e_power)
        else:
            noise_reduction_db = 60.0

        conv_time = convergence_time_90pct(
            mse, sample_rate=self.fs,
            desired=desired_arr, error=error_arr
        )

        self.results = {
            'noise_source': noise_source,
            'reference': np.array(reference),
            'desired': desired_arr,
            'antinoise': np.array(antinoise),
            'error': error_arr,
            'mse': np.array(mse),
            'noise_reduction_db': noise_reduction_db,
            'convergence_time': conv_time,
            'weights': self.fxlms.weights.copy(),
            'weights_history': np.array(weights_history),
            'duration': duration,
            'fs': self.fs,
            'params': self.params,
            'scenario_order': scenario_order,
            'segment_boundaries': segment_boundaries,
        }

        return self.results


class DynamicRideMultiRefMicSimulation:
    """
    Dynamic Ride simulation with multiple noise source positions AND multiple reference mics.
    Combines dynamic position switching with 4-ref-mic averaging.
    """

    def __init__(self, params: dict):
        self.params = params
        self.fs = params.get('sample_rate', 16000)
        self.crossfade_ms = 100.0
        self.ref_mics = params.get('ref_mics', {})

        if not self.ref_mics:
            raise ValueError("No reference mics defined for multi-ref-mic mode")

        self.ref_mic_names = list(self.ref_mics.keys())

        # Build room with all noise source positions and all ref mics
        self.room = self._create_room()
        self.room.compute_rir()

        max_len = 512
        positions = params['positions']

        # Source indices for noise positions
        self.scenario_source_idx = {
            'idle': 0,
            'city': 1,
            'highway': 2,
            'acceleration': 3,
        }

        # Error mic is the last mic
        error_mic_idx = len(self.ref_mic_names)

        # Reference paths: each noise source -> each ref mic
        # Structure: H_reference[scenario][ref_mic_name] = RIR
        self.H_reference = {}
        for scenario, src_idx in self.scenario_source_idx.items():
            self.H_reference[scenario] = {}
            for i, name in enumerate(self.ref_mic_names):
                self.H_reference[scenario][name] = self.room.rir[i][src_idx][:max_len]

        # Primary paths: each noise source -> error mic
        self.H_primary = {}
        for scenario, src_idx in self.scenario_source_idx.items():
            self.H_primary[scenario] = self.room.rir[error_mic_idx][src_idx][:max_len]

        # Secondary path: speaker -> error mic (speaker is source 4)
        self.H_secondary = self.room.rir[error_mic_idx][4][:max_len]
        self.H_secondary_est = self.H_secondary * (
            1 + 0.05 * np.random.randn(len(self.H_secondary))
        )

        # Create FIR filters
        self.primary_paths = {
            scenario: FIRPath(self.H_primary[scenario])
            for scenario in self.scenario_source_idx.keys()
        }
        # Reference paths: nested dict [scenario][ref_mic_name]
        self.reference_paths = {}
        for scenario in self.scenario_source_idx.keys():
            self.reference_paths[scenario] = {
                name: FIRPath(self.H_reference[scenario][name])
                for name in self.ref_mic_names
            }
        self.secondary_path = FIRPath(self.H_secondary)

        # Create FxNLMS
        self.fxlms = _create_fxlms(params, self.H_secondary_est, self.fs)

        self.noise_gen = NoiseMixer(self.fs)
        self.results = {}

    def _create_room(self) -> pra.ShoeBox:
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
            dims, fs=self.fs, materials=materials,
            max_order=max_order, air_absorption=True
        )

        # Add all noise source positions as separate sources (0-3)
        room.add_source(SCENARIO_NOISE_POSITIONS['idle'])
        room.add_source(SCENARIO_NOISE_POSITIONS['city'])
        room.add_source(SCENARIO_NOISE_POSITIONS['highway'])
        room.add_source(SCENARIO_NOISE_POSITIONS['acceleration'])
        # Source 4: Speaker
        room.add_source(positions['speaker'])

        # Build mic array: all ref mics + error mic
        mic_positions = [self.ref_mics[name] for name in self.ref_mic_names]
        mic_positions.append(positions['error_mic'])
        mic_array = np.array(mic_positions).T
        room.add_microphone_array(pra.MicrophoneArray(mic_array, fs=self.fs))

        return room

    def run(self, progress_callback=None) -> dict:
        duration = self.params['duration']
        crossfade_samples = int(self.crossfade_ms / 1000.0 * self.fs)

        noise_source, scenario_order, segment_boundaries = \
            self.noise_gen.generate_dynamic_scenario(duration, crossfade_ms=self.crossfade_ms)
        n_samples = len(noise_source)

        # Reset filters
        self.fxlms.reset()
        self.secondary_path.reset()
        for path in self.primary_paths.values():
            path.reset()
        for scenario_paths in self.reference_paths.values():
            for path in scenario_paths.values():
                path.reset()

        # Storage
        reference = []
        desired = []
        antinoise = []
        error = []
        mse = []
        weights_history = []
        snapshot_interval = int(WEIGHT_SNAPSHOT_INTERVAL_S * self.fs)
        ref_mic_signals = {name: [] for name in self.ref_mic_names}

        def get_scenario_at_sample(sample_idx):
            for i, boundary in enumerate(segment_boundaries):
                if i == len(segment_boundaries) - 1:
                    return scenario_order[i]
                if sample_idx < segment_boundaries[i + 1]:
                    return scenario_order[i]
            return scenario_order[-1]

        def get_crossfade_weight(sample_idx):
            for i in range(1, len(segment_boundaries)):
                boundary = segment_boundaries[i]
                adjusted_boundary = boundary - (i * crossfade_samples)
                if adjusted_boundary - crossfade_samples <= sample_idx < adjusted_boundary:
                    progress = (sample_idx - (adjusted_boundary - crossfade_samples)) / crossfade_samples
                    return progress, scenario_order[i-1], scenario_order[i]
            return None

        for i in range(n_samples):
            sample = noise_source[i]
            crossfade_info = get_crossfade_weight(i)

            if crossfade_info is not None:
                weight, from_scenario, to_scenario = crossfade_info

                ref_signals_sample = {}
                for name in self.ref_mic_names:
                    x_from = self.reference_paths[from_scenario][name].filter_sample(sample)
                    x_to = self.reference_paths[to_scenario][name].filter_sample(sample)
                    ref_sig = (1 - weight) * x_from + weight * x_to
                    ref_signals_sample[name] = ref_sig
                    ref_mic_signals[name].append(ref_sig)

                x = np.mean(list(ref_signals_sample.values()))

                d_from = self.primary_paths[from_scenario].filter_sample(sample)
                d_to = self.primary_paths[to_scenario].filter_sample(sample)
                d = (1 - weight) * d_from + weight * d_to
            else:
                current_scenario = get_scenario_at_sample(i)

                ref_signals_sample = {}
                for name in self.ref_mic_names:
                    ref_sig = self.reference_paths[current_scenario][name].filter_sample(sample)
                    ref_signals_sample[name] = ref_sig
                    ref_mic_signals[name].append(ref_sig)

                x = np.mean(list(ref_signals_sample.values()))
                d = self.primary_paths[current_scenario].filter_sample(sample)

            reference.append(x)
            desired.append(d)

            y = self.fxlms.generate_antinoise(x)
            antinoise.append(y)

            y_at_error = self.secondary_path.filter_sample(y)
            e = d + y_at_error
            error.append(e)
            mse.append(e ** 2)

            self.fxlms.filter_reference(x)
            self.fxlms.update_weights(e)

            if (i + 1) % snapshot_interval == 0:
                weights_history.append(self.fxlms.weights.copy())

            if progress_callback and (i + 1) % (n_samples // 20) == 0:
                progress = (i + 1) / n_samples
                current_mse = np.mean(mse[-1000:]) if len(mse) > 1000 else np.mean(mse)
                progress_callback(progress, current_mse)

        desired_arr = np.array(desired)
        error_arr = np.array(error)

        steady_start = len(desired_arr) // 2
        d_power = np.mean(desired_arr[steady_start:] ** 2)
        e_power = np.mean(error_arr[steady_start:] ** 2)

        if e_power > 1e-10:
            noise_reduction_db = 10 * np.log10(d_power / e_power)
        else:
            noise_reduction_db = 60.0

        conv_time = convergence_time_90pct(
            mse, sample_rate=self.fs,
            desired=desired_arr, error=error_arr
        )

        ref_mic_signals_arr = {name: np.array(sig) for name, sig in ref_mic_signals.items()}

        self.results = {
            'noise_source': noise_source,
            'reference': np.array(reference),
            'desired': desired_arr,
            'antinoise': np.array(antinoise),
            'error': error_arr,
            'mse': np.array(mse),
            'noise_reduction_db': noise_reduction_db,
            'convergence_time': conv_time,
            'weights': self.fxlms.weights.copy(),
            'weights_history': np.array(weights_history),
            'duration': duration,
            'fs': self.fs,
            'params': self.params,
            'scenario_order': scenario_order,
            'segment_boundaries': segment_boundaries,
            'num_ref_mics': len(self.ref_mics),
            'ref_mic_names': list(self.ref_mics.keys()),
            'ref_mic_signals': ref_mic_signals_arr,
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
        scenario = params.get('scenario', 'highway')
        mimo_mode = params.get('mimo_mode', 'Off')

        is_multi_speaker = speaker_mode == '4-Speaker System'
        is_multi_ref_mic = ref_mic_mode == '4-Reference Mic System'
        is_dynamic_ride = scenario == 'dynamic ride'
        use_mimo = mimo_mode != 'Off' and is_multi_speaker and not is_dynamic_ride

        # Select appropriate simulation class
        if use_mimo:
            # Lazy imports — only load if MIMO mode is active, so existing
            # default behavior is unchanged.
            if mimo_mode == 'Stage 1 SIMO (1×M×1)':
                from playground.simulation.mimo_runner import MIMOSimulation
                sim = MIMOSimulation(params)
            elif mimo_mode == 'Stage 2 SIMO+multi-error (1×M×K)':
                from playground.simulation.mimo_runner_multierror import MIMOSimulationMultiError
                sim = MIMOSimulationMultiError(params)
            elif mimo_mode == 'Stage 3 Full MIMO (N×M×K)':
                from playground.simulation.mimo_runner_full import MIMOSimulationFull
                sim = MIMOSimulationFull(params)
            else:
                # Unknown mimo_mode value — fall back to default dispatch
                use_mimo = False

        if not use_mimo:
            if is_dynamic_ride:
                # Dynamic ride uses special simulation with multi-position RIRs
                if is_multi_ref_mic:
                    sim = DynamicRideMultiRefMicSimulation(params)
                else:
                    sim = DynamicRideSimulation(params)
            elif is_multi_ref_mic and is_multi_speaker:
                sim = MultiRefMicMultiSpeakerSimulation(params)
            elif is_multi_ref_mic:
                sim = MultiRefMicSimulation(params)
            elif is_multi_speaker:
                sim = MultiSpeakerSimulation(params)
            else:
                sim = PlaygroundSimulation(params)

        results = sim.run(progress_callback)

        # For Stage 2/3 MIMO, add aggregate fields for plot compatibility
        if use_mimo and 'error_per_mic' in results:
            error_per_mic = results['error_per_mic']
            desired_per_mic = results['desired_per_mic']
            results['error'] = np.mean(error_per_mic, axis=1)
            results['desired'] = np.mean(desired_per_mic, axis=1)

        results['success'] = True
        results['error_message'] = None
        results['speaker_mode'] = speaker_mode
        results['ref_mic_mode'] = ref_mic_mode
        results['mimo_mode'] = mimo_mode if use_mimo else 'Off'
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
