"""
MIMO Stage 3 Simulation Runner — Full N×M×K MIMO

True MIMO with N independent reference mics + M speakers + K error mics.

Reference signals are kept independent (no averaging at the input). Each
speaker has N filters, one per reference mic. The total weight tensor is
shape (M, N, L) — significantly more degrees of freedom than Stage 2's
(M, L).

Default config: 4 reference mics from the playground's FOUR_REF_MIC_CONFIG
(firewall, floor, a-pillar, dashboard) — capturing engine, road, wind,
and combined noise respectively.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pyroomacoustics as pra
from src.acoustic.path_generator import FIRPath
from src.core.mimo_fxnlms_full import MIMOFxNLMSFull
from src.noise.noise_mixer import NoiseMixer
from src.ml.common.metrics import convergence_time_90pct
from playground.simulation.runner import _generate_noise, WEIGHT_SNAPSHOT_INTERVAL_S
from playground.simulation.mimo_runner_multierror import HEAD_ZONE_OFFSETS, make_head_zone_error_mics


DEFAULT_REF_MICS = {
    'firewall':  [0.3, 0.92, 0.5],
    'floor':     [2.0, 0.55, 0.15],
    'a_pillar':  [0.5, 0.15, 1.0],
    'dashboard': [0.9, 0.92, 0.8],
}


class MIMOSimulationFull:
    """
    Full MIMO ANC: N reference mics, M speakers, K error mics.
    """

    def __init__(self, params: dict):
        self.params = params
        self.fs = params.get('sample_rate', 16000)
        self.speakers = params.get('speakers', {})
        self.ref_mics = params.get('ref_mics') or DEFAULT_REF_MICS

        if not self.speakers:
            raise ValueError("MIMOSimulationFull requires 'speakers' dict")

        center = params['positions']['error_mic']
        self.error_mics = params.get('error_mics_positions',
                                     make_head_zone_error_mics(center))
        self.K = len(self.error_mics)
        self.N = len(self.ref_mics)
        self.M = len(self.speakers)

        # Set names early; needed in _create_room
        self.ref_mic_names = list(self.ref_mics.keys())
        self.speaker_names = list(self.speakers.keys())

        self.room = self._create_room()
        self.room.compute_rir()

        max_len = 512

        # Layout:
        # Source 0: noise
        # Sources 1..M: speakers
        # Mic 0..N-1: reference mics
        # Mic N..N+K-1: error mics

        # Reference paths: noise → ref mic n
        self.H_reference = [self.room.rir[n][0][:max_len] for n in range(self.N)]

        # Primary paths: noise → error mic k
        self.H_primary = [self.room.rir[self.N + k][0][:max_len] for k in range(self.K)]

        # Secondary paths: speaker m → error mic k
        # H_secondary[m][k]
        self.H_secondary = [
            [self.room.rir[self.N + k][1 + m][:max_len] for k in range(self.K)]
            for m in range(self.M)
        ]

        # Per-(speaker, error_mic) secondary path estimates with 5% modeling error
        self.H_secondary_est = [
            [path * (1 + 0.05 * np.random.randn(len(path))) for path in row]
            for row in self.H_secondary
        ]

        self.reference_paths = [FIRPath(p) for p in self.H_reference]
        self.primary_paths = [FIRPath(p) for p in self.H_primary]
        self.secondary_paths = [
            [FIRPath(p) for p in row] for row in self.H_secondary
        ]

        self.fxlms = MIMOFxNLMSFull(
            filter_length=params['filter_length'],
            step_size=params['step_size'],
            num_reference_mics=self.N,
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

        room.add_source(positions['noise_source'])
        for name in self.speakers:
            room.add_source(self.speakers[name])

        # Mics: N reference mics, then K error mics
        ref_positions = [self.ref_mics[name] for name in self.ref_mic_names]
        all_mics = ref_positions + self.error_mics
        room.add_microphone_array(pra.MicrophoneArray(np.array(all_mics).T, fs=self.fs))
        return room

    def run(self, progress_callback=None) -> dict:
        noise_source, scenario_order = _generate_noise(self.noise_gen, self.params)
        n_samples = len(noise_source)
        duration = n_samples / self.fs

        self.fxlms.reset()
        for p in self.reference_paths:
            p.reset()
        for p in self.primary_paths:
            p.reset()
        for row in self.secondary_paths:
            for p in row:
                p.reset()

        ref_signals = np.zeros((n_samples, self.N))
        desired_per_k = np.zeros((n_samples, self.K))
        antinoise_per_speaker = np.zeros((n_samples, self.M))
        error_per_k = np.zeros((n_samples, self.K))
        mse = []
        weights_history = []
        snapshot_interval = int(WEIGHT_SNAPSHOT_INTERVAL_S * self.fs)

        for i in range(n_samples):
            sample = noise_source[i]

            x_vec = np.array([self.reference_paths[n].filter_sample(sample)
                              for n in range(self.N)])
            ref_signals[i] = x_vec

            for k in range(self.K):
                desired_per_k[i, k] = self.primary_paths[k].filter_sample(sample)

            y_per = self.fxlms.generate_antinoise(x_vec)
            antinoise_per_speaker[i] = y_per

            antinoise_per_k = np.zeros(self.K)
            for m in range(self.M):
                for k in range(self.K):
                    antinoise_per_k[k] += self.secondary_paths[m][k].filter_sample(y_per[m])

            errors = desired_per_k[i] + antinoise_per_k
            error_per_k[i] = errors
            mse.append(float(np.sum(errors ** 2)))

            self.fxlms.filter_reference(x_vec)
            self.fxlms.update_weights(errors)

            if (i + 1) % snapshot_interval == 0:
                weights_history.append(self.fxlms.weights.copy())

            if progress_callback and (i + 1) % (n_samples // 20) == 0:
                progress = (i + 1) / n_samples
                current_mse = np.mean(mse[-1000:]) if len(mse) > 1000 else np.mean(mse)
                progress_callback(progress, current_mse)

        steady_start = n_samples // 2
        per_mic_nr = []
        for k in range(self.K):
            d_p = np.mean(desired_per_k[steady_start:, k] ** 2)
            e_p = np.mean(error_per_k[steady_start:, k] ** 2)
            per_mic_nr.append(10 * np.log10(d_p / e_p) if e_p > 1e-10 else 60.0)
        nr_db = float(np.mean(per_mic_nr))

        conv_time = convergence_time_90pct(
            mse, sample_rate=self.fs,
            desired=desired_per_k[:, 0], error=error_per_k[:, 0]
        )

        self.results = {
            'noise_source': noise_source,
            'reference_signals': ref_signals,
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
            'num_reference_mics': self.N,
            'num_speakers': self.M,
            'num_error_mics': self.K,
            'speaker_names': list(self.speaker_names),
            'ref_mic_names': list(self.ref_mic_names),
            'error_mic_positions': self.error_mics,
            'scenario_order': scenario_order,
            'algorithm': 'true_mimo_full',
        }

        return self.results
