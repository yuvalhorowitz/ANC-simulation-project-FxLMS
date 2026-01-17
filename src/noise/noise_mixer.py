"""
Noise Mixer

Combines multiple noise sources to create realistic car interior noise.
"""

import numpy as np
from typing import Dict

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
            'highway': {'rpm': 2800, 'speed': 120,
                       'mix': {'engine': 0.3, 'road': 0.4, 'wind': 0.3}},
            'city': {'rpm': 2000, 'speed': 50,
                    'mix': {'engine': 0.5, 'road': 0.35, 'wind': 0.15}},
            'acceleration': {'rpm': 4500, 'speed': 80,
                           'mix': {'engine': 0.7, 'road': 0.2, 'wind': 0.1}},
            'idle': {'rpm': 800, 'speed': 0,
                    'mix': {'engine': 0.9, 'road': 0.05, 'wind': 0.05}}
        }

        params = scenarios.get(scenario, scenarios['highway'])

        return self.generate(
            duration,
            rpm=params['rpm'],
            speed_kmh=params['speed'],
            mix_weights=params['mix']
        )
