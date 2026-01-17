"""
Parameter Lookup Table

Maps noise class to optimal FxNLMS parameters.
Includes both step size (μ) and filter length (L).

Filter Length Selection Rationale:
    - Filter length determines how long an impulse response can be captured
    - Longer filters = better low-frequency cancellation but slower convergence
    - Rule of thumb: L ≈ RT60 × fs × 0.5
    - For car interiors (RT60 ≈ 0.05-0.1s at 16kHz): L ≈ 400-800
    - But we use shorter filters (128-256) for faster adaptation

    Per noise type:
    - Idle: Low-frequency dominant, needs longer filter (256)
    - City: Mid-frequency mix, medium filter (192)
    - Highway: Broadband noise, standard filter (256)
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class FxNLMSParams:
    """Container for FxNLMS parameters."""
    step_size: float      # Learning rate μ
    filter_length: int    # Number of FIR taps L
    regularization: float = 1e-6

    def to_dict(self) -> Dict[str, Any]:
        return {
            'step_size': self.step_size,
            'filter_length': self.filter_length,
            'regularization': self.regularization,
        }


# Optimal parameters determined empirically from Phase 1 analysis
# and theoretical considerations for filter length
OPTIMAL_PARAMS: Dict[str, FxNLMSParams] = {
    # Idle: Engine at rest, low RPM
    # - Dominant frequencies: 30-200 Hz (engine harmonics)
    # - Characteristics: Narrowband, predictable, low amplitude
    # - Strategy: Lower μ for stability, longer filter for low-freq
    'idle': FxNLMSParams(
        step_size=0.002,
        filter_length=256,
    ),

    # City: Stop-and-go urban driving
    # - Dominant frequencies: 100-1000 Hz (engine + road)
    # - Characteristics: Variable, intermittent, mixed sources
    # - Strategy: Moderate μ for adaptability, shorter filter for speed
    'city': FxNLMSParams(
        step_size=0.008,
        filter_length=192,
    ),

    # Highway: High-speed steady driving
    # - Dominant frequencies: 200-2000 Hz (tire + wind noise)
    # - Characteristics: Broadband, relatively steady
    # - Strategy: Medium μ, standard filter length
    'highway': FxNLMSParams(
        step_size=0.005,
        filter_length=256,
    ),
}

# Default parameters (used if classification fails)
DEFAULT_PARAMS = FxNLMSParams(
    step_size=0.005,
    filter_length=256,
)


def get_params(noise_class: str) -> FxNLMSParams:
    """
    Get optimal FxNLMS parameters for a noise class.

    Args:
        noise_class: One of 'idle', 'city', 'highway'

    Returns:
        FxNLMSParams with step_size and filter_length
    """
    return OPTIMAL_PARAMS.get(noise_class.lower(), DEFAULT_PARAMS)


def get_params_dict(noise_class: str) -> Dict[str, Any]:
    """
    Get optimal parameters as dictionary.

    Args:
        noise_class: One of 'idle', 'city', 'highway'

    Returns:
        Dictionary with 'step_size', 'filter_length', 'regularization'
    """
    return get_params(noise_class).to_dict()


def get_step_size(noise_class: str) -> float:
    """Get optimal step size for noise class."""
    return get_params(noise_class).step_size


def get_filter_length(noise_class: str) -> int:
    """Get optimal filter length for noise class."""
    return get_params(noise_class).filter_length


def update_params(
    noise_class: str,
    step_size: Optional[float] = None,
    filter_length: Optional[int] = None,
    regularization: Optional[float] = None
):
    """
    Update optimal parameters for a noise class.

    Useful for fine-tuning after additional experiments.

    Args:
        noise_class: Class to update
        step_size: New step size (or None to keep current)
        filter_length: New filter length (or None to keep current)
        regularization: New regularization (or None to keep current)
    """
    if noise_class not in OPTIMAL_PARAMS:
        OPTIMAL_PARAMS[noise_class] = FxNLMSParams(
            step_size=step_size or DEFAULT_PARAMS.step_size,
            filter_length=filter_length or DEFAULT_PARAMS.filter_length,
            regularization=regularization or DEFAULT_PARAMS.regularization,
        )
    else:
        current = OPTIMAL_PARAMS[noise_class]
        OPTIMAL_PARAMS[noise_class] = FxNLMSParams(
            step_size=step_size if step_size is not None else current.step_size,
            filter_length=filter_length if filter_length is not None else current.filter_length,
            regularization=regularization if regularization is not None else current.regularization,
        )


def estimate_filter_length(
    rt60: float,
    fs: int = 16000,
    factor: float = 0.5
) -> int:
    """
    Estimate appropriate filter length from room acoustics.

    The filter should be long enough to capture the significant
    portion of the room impulse response.

    Args:
        rt60: Reverberation time T60 in seconds
        fs: Sample rate in Hz
        factor: Portion of RT60 to capture (0.5 = half)

    Returns:
        Recommended filter length (power of 2)
    """
    # Basic calculation
    raw_length = int(rt60 * fs * factor)

    # Round to nearest power of 2
    power = 1
    while power < raw_length:
        power *= 2

    # Clamp to reasonable range
    return max(64, min(power, 1024))


def get_all_params() -> Dict[str, Dict[str, Any]]:
    """
    Get all optimal parameters as nested dictionary.

    Returns:
        Dictionary mapping noise class to parameters
    """
    return {
        noise_class: params.to_dict()
        for noise_class, params in OPTIMAL_PARAMS.items()
    }


def print_params_table():
    """Print a formatted table of all parameters."""
    print("\n" + "=" * 60)
    print("OPTIMAL FxNLMS PARAMETERS BY NOISE CLASS")
    print("=" * 60)
    print(f"{'Class':<12} {'Step Size (μ)':<15} {'Filter Length (L)':<18} {'Reg.':<10}")
    print("-" * 60)

    for noise_class, params in OPTIMAL_PARAMS.items():
        print(f"{noise_class:<12} {params.step_size:<15.4f} {params.filter_length:<18} {params.regularization:<10.0e}")

    print("-" * 60)
    print(f"{'DEFAULT':<12} {DEFAULT_PARAMS.step_size:<15.4f} {DEFAULT_PARAMS.filter_length:<18} {DEFAULT_PARAMS.regularization:<10.0e}")
    print("=" * 60)
