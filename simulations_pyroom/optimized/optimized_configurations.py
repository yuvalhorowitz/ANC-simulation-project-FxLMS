"""
Optimized Configurations for FxLMS ANC Simulation

Provides pre-tuned configurations combining:
- Optimal filter length (based on RT60)
- Optimal max_order (based on room dimensions)
- Frequency-dependent materials
- Speaker configurations: single speaker AND 4-speaker stereo

Each scenario includes both speaker configurations for A/B testing.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .optimal_filter_length import calculate_optimal_filter_length
from .optimal_max_order import calculate_optimal_max_order
from .low_freq_materials import get_car_interior_materials, calculate_mean_absorption


class SpeakerConfig(Enum):
    """Speaker configuration types."""
    SINGLE = "single"
    QUAD_STEREO = "quad_stereo"


@dataclass
class SpeakerPosition:
    """Single speaker position."""
    position: List[float]  # [x, y, z] in meters
    name: str = ""


@dataclass
class MicrophonePosition:
    """Single microphone position."""
    position: List[float]  # [x, y, z] in meters
    name: str = ""
    mic_type: str = "reference"  # "reference" or "error"


@dataclass
class RoomConfig:
    """Room/cabin configuration."""
    dimensions: List[float]  # [length, width, height] in meters
    rt60: float  # Reverb time in seconds
    materials: Dict = field(default_factory=dict)
    name: str = ""


@dataclass
class OptimizedConfig:
    """Complete optimized configuration for a scenario."""
    name: str
    room: RoomConfig

    # FxLMS parameters
    filter_length: int
    step_size: float
    secondary_path_length: int

    # Pyroomacoustics parameters
    max_order: int
    fs: int = 16000

    # Single speaker configuration
    single_speaker: SpeakerPosition = None

    # 4-speaker stereo configuration
    quad_speakers: List[SpeakerPosition] = field(default_factory=list)

    # Microphone positions
    reference_mic: MicrophonePosition = None
    error_mic: MicrophonePosition = None

    # Noise source
    noise_source: List[float] = field(default_factory=list)

    # Metadata
    notes: str = ""


def create_sedan_config() -> OptimizedConfig:
    """
    Create optimized configuration for sedan interior.

    Returns:
        OptimizedConfig with single and 4-speaker setups
    """
    # Room dimensions (meters)
    dimensions = [4.8, 1.85, 1.5]
    rt60 = 0.15

    # Calculate optimal parameters
    filter_length = calculate_optimal_filter_length(rt60, fs=16000)
    max_order = calculate_optimal_max_order(dimensions, rt60)

    room = RoomConfig(
        dimensions=dimensions,
        rt60=rt60,
        materials=get_car_interior_materials(),
        name="Sedan Cabin"
    )

    # Microphone positions (relative to room origin at front-left-bottom)
    # Reference mic near firewall (front of cabin)
    reference_mic = MicrophonePosition(
        position=[0.3, 0.925, 0.8],
        name="Reference (Firewall)",
        mic_type="reference"
    )

    # Error mic at driver's head position
    error_mic = MicrophonePosition(
        position=[1.8, 0.6, 1.1],
        name="Error (Driver Head)",
        mic_type="error"
    )

    # Noise source (engine/road noise typically from front-bottom)
    noise_source = [0.1, 0.925, 0.3]

    # Single speaker configuration
    # Speaker at driver's headrest
    single_speaker = SpeakerPosition(
        position=[1.6, 0.5, 1.2],
        name="Headrest Speaker"
    )

    # 4-speaker stereo configuration
    # Front left, front right, rear left, rear right door speakers
    quad_speakers = [
        SpeakerPosition(
            position=[1.0, 0.1, 0.6],
            name="Front Left Door"
        ),
        SpeakerPosition(
            position=[1.0, 1.75, 0.6],
            name="Front Right Door"
        ),
        SpeakerPosition(
            position=[3.0, 0.1, 0.6],
            name="Rear Left Door"
        ),
        SpeakerPosition(
            position=[3.0, 1.75, 0.6],
            name="Rear Right Door"
        ),
    ]

    return OptimizedConfig(
        name="sedan",
        room=room,
        filter_length=filter_length,
        step_size=0.005,  # Tuned for sedan
        secondary_path_length=filter_length // 2,
        max_order=max_order,
        single_speaker=single_speaker,
        quad_speakers=quad_speakers,
        reference_mic=reference_mic,
        error_mic=error_mic,
        noise_source=noise_source,
        notes="Standard sedan cabin with moderate RT60"
    )


def create_suv_config() -> OptimizedConfig:
    """
    Create optimized configuration for SUV interior.

    Returns:
        OptimizedConfig with single and 4-speaker setups
    """
    dimensions = [4.7, 1.9, 1.8]
    rt60 = 0.20

    filter_length = calculate_optimal_filter_length(rt60, fs=16000)
    max_order = calculate_optimal_max_order(dimensions, rt60)

    room = RoomConfig(
        dimensions=dimensions,
        rt60=rt60,
        materials=get_car_interior_materials(),
        name="SUV Cabin"
    )

    reference_mic = MicrophonePosition(
        position=[0.3, 0.95, 0.9],
        name="Reference (Firewall)",
        mic_type="reference"
    )

    error_mic = MicrophonePosition(
        position=[1.8, 0.6, 1.3],
        name="Error (Driver Head)",
        mic_type="error"
    )

    noise_source = [0.1, 0.95, 0.3]

    # Single speaker at headrest
    single_speaker = SpeakerPosition(
        position=[1.6, 0.5, 1.4],
        name="Headrest Speaker"
    )

    # 4-speaker configuration for SUV
    quad_speakers = [
        SpeakerPosition(
            position=[1.0, 0.1, 0.7],
            name="Front Left Door"
        ),
        SpeakerPosition(
            position=[1.0, 1.8, 0.7],
            name="Front Right Door"
        ),
        SpeakerPosition(
            position=[3.2, 0.1, 0.7],
            name="Rear Left Door"
        ),
        SpeakerPosition(
            position=[3.2, 1.8, 0.7],
            name="Rear Right Door"
        ),
    ]

    return OptimizedConfig(
        name="suv",
        room=room,
        filter_length=filter_length,
        step_size=0.003,  # Lower step size for larger reverb
        secondary_path_length=filter_length // 2,
        max_order=max_order,
        single_speaker=single_speaker,
        quad_speakers=quad_speakers,
        reference_mic=reference_mic,
        error_mic=error_mic,
        noise_source=noise_source,
        notes="SUV cabin with longer RT60"
    )


def create_compact_config() -> OptimizedConfig:
    """
    Create optimized configuration for compact car interior.

    Returns:
        OptimizedConfig with single and 4-speaker setups
    """
    dimensions = [3.5, 1.8, 1.5]
    rt60 = 0.10

    filter_length = calculate_optimal_filter_length(rt60, fs=16000)
    max_order = calculate_optimal_max_order(dimensions, rt60)

    room = RoomConfig(
        dimensions=dimensions,
        rt60=rt60,
        materials=get_car_interior_materials(),
        name="Compact Car Cabin"
    )

    reference_mic = MicrophonePosition(
        position=[0.25, 0.9, 0.75],
        name="Reference (Firewall)",
        mic_type="reference"
    )

    error_mic = MicrophonePosition(
        position=[1.4, 0.55, 1.0],
        name="Error (Driver Head)",
        mic_type="error"
    )

    noise_source = [0.1, 0.9, 0.25]

    single_speaker = SpeakerPosition(
        position=[1.2, 0.45, 1.1],
        name="Headrest Speaker"
    )

    # Compact car 4-speaker setup (closer together)
    quad_speakers = [
        SpeakerPosition(
            position=[0.8, 0.1, 0.5],
            name="Front Left Door"
        ),
        SpeakerPosition(
            position=[0.8, 1.7, 0.5],
            name="Front Right Door"
        ),
        SpeakerPosition(
            position=[2.3, 0.1, 0.5],
            name="Rear Left Door"
        ),
        SpeakerPosition(
            position=[2.3, 1.7, 0.5],
            name="Rear Right Door"
        ),
    ]

    return OptimizedConfig(
        name="compact",
        room=room,
        filter_length=filter_length,
        step_size=0.008,  # Higher step size for shorter reverb
        secondary_path_length=filter_length // 2,
        max_order=max_order,
        single_speaker=single_speaker,
        quad_speakers=quad_speakers,
        reference_mic=reference_mic,
        error_mic=error_mic,
        noise_source=noise_source,
        notes="Compact car with short RT60"
    )


def create_damped_room_config() -> OptimizedConfig:
    """
    Create optimized configuration for damped room (test environment).

    Returns:
        OptimizedConfig with single and 4-speaker setups
    """
    dimensions = [6.0, 5.0, 3.0]
    rt60 = 0.20

    filter_length = calculate_optimal_filter_length(rt60, fs=16000)
    max_order = calculate_optimal_max_order(dimensions, rt60)

    room = RoomConfig(
        dimensions=dimensions,
        rt60=rt60,
        materials={
            'ceiling': 'acoustic_tile',
            'floor': 'car_carpet',
            'front': 'acoustic_tile',
            'back': 'acoustic_tile',
            'left': 'drywall',
            'right': 'drywall',
        },
        name="Damped Test Room"
    )

    reference_mic = MicrophonePosition(
        position=[1.0, 2.5, 1.5],
        name="Reference",
        mic_type="reference"
    )

    error_mic = MicrophonePosition(
        position=[3.0, 2.5, 1.2],
        name="Error (Listener)",
        mic_type="error"
    )

    noise_source = [0.5, 2.5, 1.5]

    single_speaker = SpeakerPosition(
        position=[2.5, 2.5, 1.5],
        name="Center Speaker"
    )

    quad_speakers = [
        SpeakerPosition(
            position=[2.0, 1.0, 1.5],
            name="Front Left"
        ),
        SpeakerPosition(
            position=[2.0, 4.0, 1.5],
            name="Front Right"
        ),
        SpeakerPosition(
            position=[5.0, 1.0, 1.5],
            name="Rear Left"
        ),
        SpeakerPosition(
            position=[5.0, 4.0, 1.5],
            name="Rear Right"
        ),
    ]

    return OptimizedConfig(
        name="damped_room",
        room=room,
        filter_length=filter_length,
        step_size=0.007,
        secondary_path_length=filter_length // 2,
        max_order=max_order,
        single_speaker=single_speaker,
        quad_speakers=quad_speakers,
        reference_mic=reference_mic,
        error_mic=error_mic,
        noise_source=noise_source,
        notes="Acoustically treated test room"
    )


def create_reverberant_room_config() -> OptimizedConfig:
    """
    Create optimized configuration for reverberant room (challenging scenario).

    Returns:
        OptimizedConfig with single and 4-speaker setups
    """
    dimensions = [6.0, 5.0, 3.0]
    rt60 = 0.60

    filter_length = calculate_optimal_filter_length(rt60, fs=16000)
    max_order = calculate_optimal_max_order(dimensions, rt60)

    room = RoomConfig(
        dimensions=dimensions,
        rt60=rt60,
        materials={
            'ceiling': 'concrete',
            'floor': 'wood_floor',
            'front': 'drywall',
            'back': 'drywall',
            'left': 'glass_window',
            'right': 'glass_window',
        },
        name="Reverberant Room"
    )

    reference_mic = MicrophonePosition(
        position=[1.0, 2.5, 1.5],
        name="Reference",
        mic_type="reference"
    )

    error_mic = MicrophonePosition(
        position=[3.0, 2.5, 1.2],
        name="Error (Listener)",
        mic_type="error"
    )

    noise_source = [0.5, 2.5, 1.5]

    single_speaker = SpeakerPosition(
        position=[2.5, 2.5, 1.5],
        name="Center Speaker"
    )

    quad_speakers = [
        SpeakerPosition(
            position=[2.0, 1.0, 1.5],
            name="Front Left"
        ),
        SpeakerPosition(
            position=[2.0, 4.0, 1.5],
            name="Front Right"
        ),
        SpeakerPosition(
            position=[5.0, 1.0, 1.5],
            name="Rear Left"
        ),
        SpeakerPosition(
            position=[5.0, 4.0, 1.5],
            name="Rear Right"
        ),
    ]

    return OptimizedConfig(
        name="reverberant_room",
        room=room,
        filter_length=filter_length,
        step_size=0.002,  # Lower step size for stability
        secondary_path_length=filter_length // 2,
        max_order=max_order,
        single_speaker=single_speaker,
        quad_speakers=quad_speakers,
        reference_mic=reference_mic,
        error_mic=error_mic,
        noise_source=noise_source,
        notes="Challenging reverberant environment"
    )


# All available configurations
ALL_CONFIGS = {
    'sedan': create_sedan_config,
    'suv': create_suv_config,
    'compact': create_compact_config,
    'damped_room': create_damped_room_config,
    'reverberant_room': create_reverberant_room_config,
}


def get_config(name: str) -> OptimizedConfig:
    """
    Get optimized configuration by name.

    Args:
        name: Configuration name (sedan, suv, compact, damped_room, reverberant_room)

    Returns:
        OptimizedConfig instance
    """
    if name not in ALL_CONFIGS:
        available = ', '.join(ALL_CONFIGS.keys())
        raise ValueError(f"Unknown configuration '{name}'. Available: {available}")

    return ALL_CONFIGS[name]()


def get_all_configs() -> Dict[str, OptimizedConfig]:
    """
    Get all available configurations.

    Returns:
        Dictionary of configuration name to OptimizedConfig
    """
    return {name: factory() for name, factory in ALL_CONFIGS.items()}


def get_speaker_positions(
    config: OptimizedConfig,
    speaker_config: SpeakerConfig
) -> List[List[float]]:
    """
    Get speaker positions for a specific configuration.

    Args:
        config: OptimizedConfig instance
        speaker_config: SINGLE or QUAD_STEREO

    Returns:
        List of [x, y, z] positions
    """
    if speaker_config == SpeakerConfig.SINGLE:
        return [config.single_speaker.position]
    elif speaker_config == SpeakerConfig.QUAD_STEREO:
        return [sp.position for sp in config.quad_speakers]
    else:
        raise ValueError(f"Unknown speaker config: {speaker_config}")


def print_config_summary(config: OptimizedConfig):
    """Print a formatted summary of a configuration."""
    print(f"\n{'='*60}")
    print(f"Configuration: {config.name}")
    print(f"{'='*60}")

    print(f"\nRoom: {config.room.name}")
    print(f"  Dimensions: {config.room.dimensions} m")
    print(f"  RT60: {config.room.rt60:.2f} s")

    print(f"\nFxLMS Parameters:")
    print(f"  Filter length: {config.filter_length} taps ({config.filter_length/config.fs*1000:.1f} ms)")
    print(f"  Step size: {config.step_size}")
    print(f"  Secondary path length: {config.secondary_path_length}")

    print(f"\nPyroomacoustics:")
    print(f"  max_order: {config.max_order}")

    print(f"\nSingle Speaker Configuration:")
    print(f"  {config.single_speaker.name}: {config.single_speaker.position}")

    print(f"\n4-Speaker Stereo Configuration:")
    for sp in config.quad_speakers:
        print(f"  {sp.name}: {sp.position}")

    print(f"\nMicrophones:")
    print(f"  Reference: {config.reference_mic.position}")
    print(f"  Error: {config.error_mic.position}")

    print(f"\nNoise Source: {config.noise_source}")
    print(f"Notes: {config.notes}")


if __name__ == '__main__':
    print("Optimized Configurations Summary")
    print("=" * 60)

    for name in ALL_CONFIGS:
        config = get_config(name)
        print_config_summary(config)
