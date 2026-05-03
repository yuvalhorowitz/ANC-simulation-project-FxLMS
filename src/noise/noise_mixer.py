"""
Noise Mixer

Combines multiple noise sources to create realistic car interior noise.
"""

import numpy as np
from typing import Dict
from scipy.io import wavfile

from .engine_noise import EngineNoiseGenerator
from .road_noise import RoadNoiseGenerator
from .wind_noise import WindNoiseGenerator


class NoiseMixer:
    """
    Combines engine, road, and wind noise for realistic car interior sound.
    """

    def __init__(self, sample_rate: float = 16000):
        """
        Initialize noise mixer.

        Args:
            sample_rate: Sampling rate in Hz
        """
        self.fs = sample_rate
        self.engine = EngineNoiseGenerator(sample_rate)
        self.road = RoadNoiseGenerator(sample_rate)
        self.wind = WindNoiseGenerator(sample_rate)

    def generate(
        self,
        duration: float,
        rpm: float = 2500,
        speed_kmh: float = 80,
        mix_weights: Dict[str, float] = None,
        amplitude: float = 1.0
    ) -> np.ndarray:
        """
        Generate combined car interior noise.

        Args:
            duration: Duration in seconds
            rpm: Engine RPM
            speed_kmh: Vehicle speed
            mix_weights: Dict with 'engine', 'road', 'wind' weights
            amplitude: Overall amplitude

        Returns:
            Combined noise signal
        """
        if mix_weights is None:
            mix_weights = {
                'engine': 0.5,
                'road': 0.35,
                'wind': 0.15
            }

        # Generate individual components
        engine_noise = self.engine.generate(duration, rpm=rpm, amplitude=1.0)
        road_noise = self.road.generate(duration, speed_kmh=speed_kmh, amplitude=1.0)
        wind_noise = self.wind.generate(duration, speed_kmh=speed_kmh, amplitude=1.0)

        # Mix
        combined = (
            mix_weights.get('engine', 0) * engine_noise +
            mix_weights.get('road', 0) * road_noise +
            mix_weights.get('wind', 0) * wind_noise
        )

        # Normalize
        if np.max(np.abs(combined)) > 0:
            combined = amplitude * combined / np.max(np.abs(combined))

        return combined

    def generate_scenario(self, duration: float, scenario: str = 'highway') -> np.ndarray:
        """
        Generate noise for predefined driving scenarios.

        Args:
            duration: Duration in seconds
            scenario: 'highway', 'city', 'acceleration', 'idle'

        Returns:
            Noise signal for scenario
        """
        scenarios = {
            'highway': {'rpm': 2800, 'speed': 120, 'amplitude': 0.8,
                       'mix': {'engine': 0.3, 'road': 0.4, 'wind': 0.3}},
            'city': {'rpm': 2000, 'speed': 50, 'amplitude': 0.5,
                    'mix': {'engine': 0.5, 'road': 0.35, 'wind': 0.15}},
            'acceleration': {'rpm': 4500, 'speed': 80, 'amplitude': 1.0,
                           'mix': {'engine': 0.7, 'road': 0.2, 'wind': 0.1}},
            'idle': {'rpm': 800, 'speed': 0, 'amplitude': 0.2,
                    'mix': {'engine': 0.9, 'road': 0.05, 'wind': 0.05}}
        }

        params = scenarios.get(scenario, scenarios['highway'])

        return self.generate(
            duration,
            rpm=params['rpm'],
            speed_kmh=params['speed'],
            mix_weights=params['mix'],
            amplitude=params['amplitude']
        )

    def generate_dynamic_scenario(
        self,
        duration: float,
        seed: int = None,
        crossfade_ms: float = 100.0
    ) -> tuple:
        """
        Generate a dynamic ride with randomly ordered scenarios and crossfade transitions.

        Creates a sequence of all 4 scenarios (idle, city, highway, acceleration)
        in random order, useful for testing ANC adaptation to changing conditions.

        Args:
            duration: Total duration in seconds (divided equally among scenarios)
            seed: Optional random seed for reproducibility
            crossfade_ms: Crossfade duration in milliseconds (default 100ms)

        Returns:
            Tuple of (noise_signal, scenario_order, segment_boundaries) where:
            - noise_signal: The combined audio signal
            - scenario_order: List of scenario names in order
            - segment_boundaries: List of sample indices where each segment starts
        """
        import random
        if seed is not None:
            random.seed(seed)

        scenario_names = ['idle', 'city', 'highway', 'acceleration']
        random.shuffle(scenario_names)

        segment_duration = duration / len(scenario_names)
        crossfade_samples = int(crossfade_ms / 1000.0 * self.fs)

        # Generate all segments
        segments = []
        for scenario in scenario_names:
            segment = self.generate_scenario(segment_duration, scenario)
            segments.append(segment)

        # Calculate segment boundaries (sample indices)
        segment_len = len(segments[0])
        segment_boundaries = [i * segment_len for i in range(len(scenario_names))]

        # Apply crossfade between segments
        combined = self._crossfade_segments(segments, crossfade_samples)

        return combined, scenario_names, segment_boundaries

    def load_audio_file(self, filepath: str, duration: float = None) -> np.ndarray:
        """
        Load a WAV file and return as normalized mono signal.

        Args:
            filepath: Path to WAV file
            duration: Max duration in seconds (None = full file)

        Returns:
            Normalized audio signal resampled to self.fs
        """
        file_fs, data = wavfile.read(filepath)

        samples = data.astype(np.float64)
        if samples.dtype == np.float64 and np.max(np.abs(data)) > 1:
            info = np.iinfo(data.dtype)
            samples = samples / max(abs(info.min), abs(info.max))

        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        if file_fs != self.fs:
            ratio = self.fs / file_fs
            new_len = int(len(samples) * ratio)
            indices = np.arange(new_len) / ratio
            indices = np.clip(indices, 0, len(samples) - 1).astype(int)
            samples = samples[indices]

        if duration is not None:
            max_samples = int(duration * self.fs)
            if len(samples) > max_samples:
                samples = samples[:max_samples]

        if np.max(np.abs(samples)) > 0:
            samples = samples / np.max(np.abs(samples))

        return samples

    def _crossfade_segments(
        self,
        segments: list,
        crossfade_samples: int
    ) -> np.ndarray:
        """
        Concatenate segments with crossfade transitions.

        Args:
            segments: List of numpy arrays (audio segments)
            crossfade_samples: Number of samples for crossfade

        Returns:
            Combined signal with smooth transitions
        """
        if len(segments) == 0:
            return np.array([])

        if len(segments) == 1:
            return segments[0]

        # Start with first segment
        result = segments[0].copy()

        for i in range(1, len(segments)):
            next_seg = segments[i]

            if crossfade_samples > 0 and crossfade_samples < len(result) and crossfade_samples < len(next_seg):
                # Create crossfade weights
                fade_out = np.linspace(1.0, 0.0, crossfade_samples)
                fade_in = np.linspace(0.0, 1.0, crossfade_samples)

                # Apply crossfade to overlap region
                result[-crossfade_samples:] = (
                    result[-crossfade_samples:] * fade_out +
                    next_seg[:crossfade_samples] * fade_in
                )

                # Append the rest of the next segment (after crossfade region)
                result = np.concatenate([result, next_seg[crossfade_samples:]])
            else:
                # No crossfade, just concatenate
                result = np.concatenate([result, next_seg])

        return result
