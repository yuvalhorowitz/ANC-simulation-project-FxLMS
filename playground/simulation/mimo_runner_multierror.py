"""
MIMO Stage 2 Simulation Runner — 1 ref × M speakers × K error mics

Builds a room with K error mics distributed around a head-sized region
(centered on the canonical driver headrest). Trains MIMOFxNLMSMultiError
with the cost function J = sum_k e_k(n)^2 — minimizes noise across all K
error mics simultaneously, producing a wider quiet zone.

Default error-mic configuration: 4 mics arranged in a 2×2 grid in the y-z
plane, ±5 cm offsets in y and z from the original error_mic position.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pyroomacoustics as pra
from src.acoustic.path_generator import FIRPath
from src.core.mimo_fxnlms_multierror import MIMOFxNLMSMultiError
from src.noise.noise_mixer import NoiseMixer
from src.ml.common.metrics import convergence_time_90pct
from playground.simulation.runner import _generate_noise, WEIGHT_SNAPSHOT_INTERVAL_S


HEAD_ZONE_OFFSETS = [
    (0.05, 0.05),   # y +5cm, z +5cm
    (-0.05, 0.05),  # y -5cm, z +5cm
    (0.05, -0.05),  # y +5cm, z -5cm
    (-0.05, -0.05), # y -5cm, z -5cm
]


def make_head_zone_error_mics(center, offsets=HEAD_ZONE_OFFSETS):
    """Generate K error mic positions around `center` using y/z offsets."""
    cx, cy, cz = center
    mics = []
    for dy, dz in offsets:
        mics.append([cx, cy + dy, cz + dz])
    return mics


class MIMOSimulationMultiError:
    """
    True MIMO with K error mics (Stage 2). 1 reference, M speakers, K error mics.
    """

    def __init__(self, params: dict):
        self.params = params
        self.fs = params.get('sample_rate', 16000)
        self.speakers = params.get('speakers', {})
        if not self.speakers:
            raise ValueError("MIMOSimulationMultiError requires 'speakers' dict")

        # Build K error mic positions around the driver headrest
        center = params['positions']['error_mic']
        self.error_mics = params.get('error_mics_positions',
                                     make_head_zone_error_mics(center))
        self.K = len(self.error_mics)

        self.room = self._create_room()
        self.room.compute_rir()

        max_len = 512
        positions = params['positions']

        # Mic indices: 0 = reference, 1..K = error mics
        # Source indices: 0 = noise, 1..M = speakers
        self.H_reference = self.room.rir[0][0][:max_len]

        self.speaker_names = list(self.speakers.keys())
        self.M = len(self.speaker_names)

        # Per-error-mic primary paths (noise → error mic k)
        self.H_primary = [self.room.rir[1 + k][0][:max_len] for k in range(self.K)]

        # Per-(speaker, error_mic) secondary path: speaker m → error mic k
        # Indexed as H_secondary[m][k]
        self.H_secondary = [
            [self.room.rir[1 + k][1 + m][:max_len] for k in range(self.K)]
            for m in range(self.M)
        ]

        # Per-(speaker, error_mic) secondary path estimates with 5% modeling error
        self.H_secondary_est = [
            [path * (1 + 0.05 * np.random.randn(len(path))) for path in row]
            for row in self.H_secondary
        ]

        # FIR filters for the actual paths (used during simulation)
        self.primary_paths = [FIRPath(p) for p in self.H_primary]
        self.reference_path = FIRPath(self.H_reference)
        self.secondary_paths = [
            [FIRPath(p) for p in row] for row in self.H_secondary
        ]

        # MIMO multi-error adaptive filter
        self.fxlms = MIMOFxNLMSMultiError(
            filter_length=params['filter_length'],
            step_size=params['step_size'],
            secondary_path_estimates=self.H_secondary_est,
            regularization=1e-4,
            leakage=params.get('leakage', 0.0),
        )

        self.noise_gen = NoiseMixer(self.fs)
        self.results = {}

    def _create_room(self):
        dims = self.params['dimensions']
        absorption = self.params['absorption']
        max_order = self.params['max_order']
        positions = self.params['positions']

        materials = {
            'ceiling': pra.Material(absorption * 1.1),
            'floor':   pra.Material(absorption * 1.5),
            'east':    pra.Material(absorption * 0.5),
            'west':    pra.Material(absorption * 0.5),
            'north':   pra.Material(absorption * 0.7),
            'south':   pra.Material(absorption * 0.9),
        }

        room = pra.ShoeBox(dims, fs=self.fs, materials=materials,
                           max_order=max_order, air_absorption=True)

        # Source 0: noise. Sources 1..M: speakers
        room.add_source(positions['noise_source'])
        for name in self.speakers:
            room.add_source(self.speakers[name])

        # Mic 0: reference. Mics 1..K: error mics around head zone
        all_mics = [positions['reference_mic']] + self.error_mics
        room.add_microphone_array(pra.MicrophoneArray(np.array(all_mics).T, fs=self.fs))
        return room

    def run(self, progress_callback=None) -> dict:
        noise_source, scenario_order = _generate_noise(self.noise_gen, self.params)
        n_samples = len(noise_source)
        duration = n_samples / self.fs

        self.fxlms.reset()
        self.reference_path.reset()
        for p in self.primary_paths:
            p.reset()
        for row in self.secondary_paths:
            for p in row:
                p.reset()

        reference = []
        desired_per_k = np.zeros((n_samples, self.K))
        antinoise_per_speaker = np.zeros((n_samples, self.M))
        error_per_k = np.zeros((n_samples, self.K))
        mse = []
        weights_history = []
        snapshot_interval = int(WEIGHT_SNAPSHOT_INTERVAL_S * self.fs)

        for i in range(n_samples):
            sample = noise_source[i]

            x = self.reference_path.filter_sample(sample)
            reference.append(x)

            # Noise propagates through K primary paths to K error mics
            for k in range(self.K):
                desired_per_k[i, k] = self.primary_paths[k].filter_sample(sample)

            # Each speaker produces its own anti-noise
            y_per = self.fxlms.generate_antinoise(x)
            antinoise_per_speaker[i] = y_per

            # Anti-noise from each speaker propagates to each error mic
            antinoise_per_k = np.zeros(self.K)
            for m in range(self.M):
                for k in range(self.K):
                    antinoise_per_k[k] += self.secondary_paths[m][k].filter_sample(y_per[m])

            # Error per error mic
            errors = desired_per_k[i] + antinoise_per_k
            error_per_k[i] = errors

            mse.append(float(np.sum(errors ** 2)))

            self.fxlms.filter_reference(x)
            self.fxlms.update_weights(errors)

            if (i + 1) % snapshot_interval == 0:
                weights_history.append(self.fxlms.weights.copy())

            if progress_callback and (i + 1) % (n_samples // 20) == 0:
                progress = (i + 1) / n_samples
                current_mse = np.mean(mse[-1000:]) if len(mse) > 1000 else np.mean(mse)
                progress_callback(progress, current_mse)

        # Per-error-mic and aggregate metrics
        steady_start = n_samples // 2
        per_mic_nr = []
        for k in range(self.K):
            d_p = np.mean(desired_per_k[steady_start:, k] ** 2)
            e_p = np.mean(error_per_k[steady_start:, k] ** 2)
            nr_k = 10 * np.log10(d_p / e_p) if e_p > 1e-10 else 60.0
            per_mic_nr.append(nr_k)

        # Headline NR = mean across error mics
        nr_db = float(np.mean(per_mic_nr))

        # Convergence time using sum-MSE
        conv_time = convergence_time_90pct(
            mse, sample_rate=self.fs,
            desired=desired_per_k[:, 0], error=error_per_k[:, 0]
        )

        self.results = {
            'noise_source': noise_source,
            'reference': np.array(reference),
            'desired_per_mic': desired_per_k,
            'antinoise_per_speaker': antinoise_per_speaker,
            'error_per_mic': error_per_k,
            'mse': np.array(mse),
            'noise_reduction_db': nr_db,
            'noise_reduction_per_mic_db': per_mic_nr,
            'convergence_time': conv_time,
            'weights': self.fxlms.weights.copy(),
            'weights_history': np.array(weights_history) if weights_history else np.array([]),
            'duration': duration,
            'fs': self.fs,
            'params': self.params,
            'num_speakers': self.M,
            'num_error_mics': self.K,
            'speaker_names': list(self.speaker_names),
            'error_mic_positions': self.error_mics,
            'scenario_order': scenario_order,
            'algorithm': 'true_mimo_multi_error',
        }

        return self.results
