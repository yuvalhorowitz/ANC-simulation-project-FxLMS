"""
Microphone Placement Optimization Module

Contains configurations and testing utilities for finding optimal
speaker and microphone positions for car ANC systems.
"""

from .microphone_config import (
    SPEAKER_POSITIONS,
    REF_MIC_POSITIONS,
    ERROR_MIC_POSITIONS,
    SEDAN_DIMENSIONS,
    SEDAN_MATERIALS,
    get_placement_config,
    get_all_placement_combinations,
)
