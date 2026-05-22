"""
MIMO Simulation Runner — Stage 1 (1 ref × M speakers × 1 error mic)

Runs true MIMO ANC: M independent adaptive filters, one per speaker, each with
its own secondary path estimate. Each speaker emits a unique anti-noise signal
optimized for its own acoustic path. The single error mic provides a scalar
error signal that drives all M filter updates simultaneously.

Mirrors the structure of `MultiSpeakerSimulation` (in playground/simulation/runner.py)
but uses MIMOFxNLMS instead of scalar FxNLMS, and per-speaker secondary path
estimates instead of a combined sum.

This file is isolated from the existing runner.py — no modifications to existing
simulation classes. See docs/mimo_plan.md for the design and isolation guarantees.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pyroomacoustics as pra
from src.acoustic.path_generator import FIRPath
from src.core.mimo_fxnlms import MIMOFxNLMS
from src.noise.noise_mixer import NoiseMixer
from src.ml.common.metrics import convergence_time_90pct
from playground.simulation.runner import _generate_noise, WEIGHT_SNAPSHOT_INTERVAL_S


class MIMOSimulation:
    """
    True MIMO ANC: 1 reference mic, M speakers, 1 error mic.

    Each of the M speakers has its own adaptive filter with its own secondary
    path estimate. Anti-noise contributions from all speakers sum at the error
    mic; the resulting scalar error drives independent updates of each filter's
    weights.
    """

    def __init__(self, params: dict):
        self.params = params
        self.fs = params.get('sample_rate', 16000)
        self.speakers = params.get('speakers', {})

        if not self.speakers:
            raise ValueError("MIMOSimulation requires 'speakers' dict in params")

        self.room = self._create_room()
        self.room.compute_rir()

        max_len = 512
        positions = params['positions']

        # Primary and reference paths (same as MultiSpeakerSimulation)
        self.H_primary = self.room.rir[1][0][:max_len]
        self.H_reference = self.room.rir[0][0][:max_len]

        # Per-speaker secondary paths (actual)
        self.speaker_names = list(self.speakers.keys())
        self.H_secondary = {}
        for i, name in enumerate(self.speaker_names):
            rir = self.room.rir[1][i + 1][:max_len]
            self.H_secondary[name] = rir

        # Per-speaker secondary path ESTIMATES (5% modeling error each)
        # This is the key MIMO difference: each speaker has its own s_hat_m,
        # not a combined sum.
        self.H_secondary_est = {}
        for name in self.speaker_names:
            path = self.H_secondary[name]
            self.H_secondary_est[name] = path * (1 + 0.05 * np.random.randn(len(path)))

        self.primary_path = FIRPath(self.H_primary)
        self.reference_path = FIRPath(self.H_reference)
        self.secondary_paths = {
            name: FIRPath(self.H_secondary[name]) for name in self.speaker_names
        }

        # Build MIMO adaptive filter with per-speaker secondary path estimates
        self.fxlms = MIMOFxNLMS(
            filter_length=params['filter_length'],
            step_size=params['step_size'],
            secondary_path_estimates=[
                self.H_secondary_est[name] for name in self.speaker_names
            ],
            regularization=1e-4,
            leakage=params.get('leakage', 0.0),
        )

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
            dims,
            fs=self.fs,
            materials=materials,
            max_order=max_order,
            air_absorption=True,
        )

        # Source 0: noise. Sources 1..M: speakers.
        room.add_source(positions['noise_source'])
        for name in self.speakers:
            room.add_source(self.speakers[name])

        # Mic 0: reference, Mic 1: error.
        mic_array = np.array([positions['reference_mic'], positions['error_mic']]).T
        room.add_microphone_array(pra.MicrophoneArray(mic_array, fs=self.fs))

        return room

    def run(self, progress_callback=None) -> dict:
        noise_source, scenario_order = _generate_noise(self.noise_gen, self.params)
        n_samples = len(noise_source)
        duration = n_samples / self.fs
        M = len(self.speaker_names)

        self.fxlms.reset()
        self.primary_path.reset()
        self.reference_path.reset()
        for path in self.secondary_paths.values():
            path.reset()

        reference = []
        desired = []
        antinoise = []  # store summed anti-noise for compatibility with existing plots
        antinoise_per_speaker = np.zeros((n_samples, M), dtype=np.float64)
        error = []
        mse = []
        weights_history = []
        snapshot_interval = int(WEIGHT_SNAPSHOT_INTERVAL_S * self.fs)

        for i in range(n_samples):
            sample = noise_source[i]

            x = self.reference_path.filter_sample(sample)
            reference.append(x)

            d = self.primary_path.filter_sample(sample)
            desired.append(d)

            # Generate one anti-noise sample per speaker
            y_per_speaker = self.fxlms.generate_antinoise(x)
            antinoise_per_speaker[i] = y_per_speaker

            # Each speaker's anti-noise propagates through its OWN secondary path
            y_at_error = 0.0
            for m, name in enumerate(self.speaker_names):
                y_at_error += self.secondary_paths[name].filter_sample(y_per_speaker[m])
            antinoise.append(y_per_speaker.sum())  # for plot compatibility

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
        nr_db = 10 * np.log10(d_power / e_power) if e_power > 1e-10 else 60.0

        conv_time = convergence_time_90pct(
            mse, sample_rate=self.fs, desired=desired_arr, error=error_arr
        )

        self.results = {
            'noise_source': noise_source,
            'reference': np.array(reference),
            'desired': desired_arr,
            'antinoise': np.array(antinoise),
            'antinoise_per_speaker': antinoise_per_speaker,
            'error': error_arr,
            'mse': np.array(mse),
            'noise_reduction_db': nr_db,
            'convergence_time': conv_time,
            'weights': self.fxlms.weights.copy(),
            'weights_history': np.array(weights_history) if weights_history else np.array([]),
            'duration': duration,
            'fs': self.fs,
            'params': self.params,
            'num_speakers': M,
            'speaker_names': list(self.speaker_names),
            'scenario_order': scenario_order,
            'algorithm': 'true_mimo',
        }

        return self.results
