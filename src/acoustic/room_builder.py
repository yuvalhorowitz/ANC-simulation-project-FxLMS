"""
Room Builder Module

Factory functions and classes for creating pyroomacoustics room simulations.
"""

import numpy as np
import pyroomacoustics as pra
from typing import Dict, List, Tuple, Optional, Union


class RoomBuilder:
    """
    Factory for creating pyroomacoustics room simulations.

    Provides pre-configured room templates and customization options
    for ANC simulation scenarios.
    """

    # Speed of sound in air at 20C
    SPEED_OF_SOUND = 343.0

    @staticmethod
    def simple_room(
        dimensions: List[float],
        fs: int = 16000,
        absorption: float = 0.2,
        max_order: int = 3
    ) -> pra.ShoeBox:
        """
        Create a simple rectangular room with uniform absorption.

        Args:
            dimensions: [length, width, height] in meters
            fs: Sampling frequency in Hz
            absorption: Absorption coefficient (0-1)
            max_order: Maximum reflection order for image source method

        Returns:
            pyroomacoustics ShoeBox room
        """
        room = pra.ShoeBox(
            dimensions,
            fs=fs,
            materials=pra.Material(absorption),
            max_order=max_order
        )
        return room

    @staticmethod
    def room_with_rt60(
        dimensions: List[float],
        rt60: float,
        fs: int = 16000
    ) -> pra.ShoeBox:
        """
        Create a room with specified reverberation time (RT60).

        Uses the Sabine formula to calculate required absorption.

        Args:
            dimensions: [length, width, height] in meters
            rt60: Desired reverberation time in seconds
            fs: Sampling frequency in Hz

        Returns:
            pyroomacoustics ShoeBox room
        """
        # Calculate absorption from RT60 using inverse Sabine
        e_absorption, max_order = pra.inverse_sabine(rt60, dimensions)

        room = pra.ShoeBox(
            dimensions,
            fs=fs,
            materials=pra.Material(e_absorption),
            max_order=max_order
        )
        return room

    @staticmethod
    def car_cabin(
        fs: int = 16000,
        max_order: int = 3
    ) -> pra.ShoeBox:
        """
        Create a pre-configured car cabin room.

        Dimensions and absorption coefficients based on typical
        sedan interior measurements.

        Args:
            fs: Sampling frequency in Hz
            max_order: Maximum reflection order

        Returns:
            pyroomacoustics ShoeBox room configured as car cabin
        """
        # Typical car cabin dimensions (meters)
        # Length x Width x Height
        dimensions = [4.5, 2.0, 1.2]

        # Wall-specific materials (absorption coefficients)
        # Based on typical car interior materials
        materials = {
            'ceiling': pra.Material(0.3),   # Headliner (fabric)
            'floor': pra.Material(0.5),     # Carpet
            'east': pra.Material(0.15),     # Passenger side windows
            'west': pra.Material(0.15),     # Driver side windows
            'north': pra.Material(0.2),     # Dashboard/windshield
            'south': pra.Material(0.25),    # Rear seats/window
        }

        room = pra.ShoeBox(
            dimensions,
            fs=fs,
            materials=materials,
            max_order=max_order,
            air_absorption=True
        )
        return room

    @staticmethod
    def anechoic_room(
        dimensions: List[float],
        fs: int = 16000
    ) -> pra.ShoeBox:
        """
        Create a nearly anechoic room (no reflections).

        Useful for testing direct path only.

        Args:
            dimensions: [length, width, height] in meters
            fs: Sampling frequency in Hz

        Returns:
            pyroomacoustics ShoeBox with max absorption
        """
        room = pra.ShoeBox(
            dimensions,
            fs=fs,
            materials=pra.Material(0.99),  # Near-perfect absorption
            max_order=0  # No reflections
        )
        return room

    @staticmethod
    def add_anc_components(
        room: pra.ShoeBox,
        noise_source_pos: List[float],
        speaker_pos: List[float],
        ref_mic_pos: List[float],
        error_mic_pos: List[float],
        noise_signal: np.ndarray = None
    ) -> pra.ShoeBox:
        """
        Add standard ANC components to a room.

        Sets up:
        - Source 0: Noise source
        - Source 1: Control speaker (initially silent)
        - Mic 0: Reference microphone
        - Mic 1: Error microphone

        Args:
            room: pyroomacoustics room to modify
            noise_source_pos: [x, y, z] position of noise source
            speaker_pos: [x, y, z] position of control speaker
            ref_mic_pos: [x, y, z] position of reference microphone
            error_mic_pos: [x, y, z] position of error microphone
            noise_signal: Optional noise signal array

        Returns:
            Modified room with ANC components
        """
        # Default noise signal if not provided
        if noise_signal is None:
            noise_signal = np.zeros(room.fs)  # 1 second of silence

        # Add noise source (index 0)
        room.add_source(noise_source_pos, signal=noise_signal)

        # Add control speaker (index 1) - initially silent
        room.add_source(speaker_pos, signal=np.zeros_like(noise_signal))

        # Add microphones
        mic_positions = np.array([ref_mic_pos, error_mic_pos]).T
        mic_array = pra.MicrophoneArray(mic_positions, fs=room.fs)
        room.add_microphone_array(mic_array)

        return room


def calculate_delay_samples(distance: float, fs: int, c: float = 343.0) -> int:
    """
    Calculate propagation delay in samples.

    Args:
        distance: Distance in meters
        fs: Sampling frequency in Hz
        c: Speed of sound in m/s

    Returns:
        Delay in samples
    """
    delay_seconds = distance / c
    return int(delay_seconds * fs)


def calculate_distance(pos1: List[float], pos2: List[float]) -> float:
    """
    Calculate Euclidean distance between two 3D positions.

    Args:
        pos1: [x, y, z] first position
        pos2: [x, y, z] second position

    Returns:
        Distance in meters
    """
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))
