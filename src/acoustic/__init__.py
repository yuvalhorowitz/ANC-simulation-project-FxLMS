"""
Acoustic modeling module.

Provides pyroomacoustics-based realistic room simulation for ANC systems.
"""

# pyroomacoustics-based models
from .room_builder import RoomBuilder, calculate_delay_samples, calculate_distance
from .path_generator import AcousticPathGenerator, FIRPath
