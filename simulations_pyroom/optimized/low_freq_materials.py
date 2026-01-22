"""
Low-Frequency Material Absorption Coefficients

Provides frequency-dependent absorption coefficients optimized for 20-300 Hz ANC target range.

Key Insight:
    Standard pyroomacoustics materials use 500Hz reference values, but ANC
    targets 20-300 Hz where absorption characteristics differ significantly.

Solution:
    Create material profiles with octave band coefficients emphasizing low frequencies.
"""

import numpy as np

try:
    import pyroomacoustics as pra
    HAS_PRA = True
except ImportError:
    HAS_PRA = False


# Octave band center frequencies (Hz)
OCTAVE_BANDS = [125, 250, 500, 1000, 2000, 4000]


# Low-frequency absorption coefficients from acoustic literature
# Format: [125Hz, 250Hz, 500Hz, 1000Hz, 2000Hz, 4000Hz]
MATERIAL_COEFFICIENTS = {
    # Car interior materials
    'glass_window': {
        'description': 'Automotive glass (tempered)',
        'absorption': [0.35, 0.25, 0.18, 0.12, 0.07, 0.04],
        'scattering': 0.05,
        'notes': 'High low-freq absorption due to panel resonance'
    },
    'car_carpet': {
        'description': 'Automotive floor carpet with backing',
        'absorption': [0.08, 0.24, 0.57, 0.69, 0.71, 0.73],
        'scattering': 0.10,
        'notes': 'Low absorption at bass frequencies'
    },
    'car_headliner': {
        'description': 'Fabric headliner with foam backing',
        'absorption': [0.10, 0.20, 0.40, 0.55, 0.60, 0.60],
        'scattering': 0.15,
        'notes': 'Moderate low-freq absorption'
    },
    'dashboard_plastic': {
        'description': 'Hard plastic dashboard',
        'absorption': [0.02, 0.03, 0.04, 0.05, 0.05, 0.05],
        'scattering': 0.05,
        'notes': 'Very reflective at all frequencies'
    },
    'seat_fabric': {
        'description': 'Upholstered car seat',
        'absorption': [0.15, 0.35, 0.55, 0.65, 0.60, 0.55],
        'scattering': 0.20,
        'notes': 'Good mid-high absorption'
    },
    'seat_leather': {
        'description': 'Leather car seat',
        'absorption': [0.10, 0.15, 0.25, 0.35, 0.40, 0.45],
        'scattering': 0.15,
        'notes': 'More reflective than fabric'
    },
    'door_panel': {
        'description': 'Door panel (mixed materials)',
        'absorption': [0.08, 0.12, 0.18, 0.22, 0.25, 0.28],
        'scattering': 0.10,
        'notes': 'Composite panel behavior'
    },
    'metal_body': {
        'description': 'Car body metal (under carpet/panels)',
        'absorption': [0.01, 0.01, 0.02, 0.02, 0.02, 0.03],
        'scattering': 0.02,
        'notes': 'Highly reflective'
    },

    # Room materials (for comparison/testing)
    'acoustic_tile': {
        'description': 'Acoustic ceiling tile',
        'absorption': [0.30, 0.50, 0.70, 0.85, 0.90, 0.85],
        'scattering': 0.30,
        'notes': 'Good broadband absorption'
    },
    'concrete': {
        'description': 'Bare concrete',
        'absorption': [0.01, 0.01, 0.02, 0.02, 0.02, 0.03],
        'scattering': 0.05,
        'notes': 'Highly reflective'
    },
    'drywall': {
        'description': 'Painted drywall/gypsum board',
        'absorption': [0.29, 0.10, 0.05, 0.04, 0.07, 0.09],
        'scattering': 0.05,
        'notes': 'Low-freq absorption from panel resonance'
    },
    'wood_floor': {
        'description': 'Hardwood flooring',
        'absorption': [0.15, 0.11, 0.10, 0.07, 0.06, 0.07],
        'scattering': 0.10,
        'notes': 'Some low-freq absorption'
    },
}


def get_material_absorption(material_name: str) -> dict:
    """
    Get absorption coefficients for a material.

    Args:
        material_name: Name of material (see MATERIAL_COEFFICIENTS keys)

    Returns:
        Dictionary with absorption, scattering, and metadata
    """
    if material_name not in MATERIAL_COEFFICIENTS:
        available = ', '.join(MATERIAL_COEFFICIENTS.keys())
        raise ValueError(f"Unknown material '{material_name}'. Available: {available}")

    return MATERIAL_COEFFICIENTS[material_name].copy()


def get_low_freq_absorption(material_name: str, freq_hz: float = 125.0) -> float:
    """
    Get absorption coefficient at a specific low frequency.

    Interpolates between octave bands if needed.

    Args:
        material_name: Name of material
        freq_hz: Target frequency in Hz (default 125)

    Returns:
        Absorption coefficient (0-1)
    """
    mat = get_material_absorption(material_name)
    coeffs = mat['absorption']

    if freq_hz <= OCTAVE_BANDS[0]:
        return coeffs[0]
    elif freq_hz >= OCTAVE_BANDS[-1]:
        return coeffs[-1]
    else:
        # Log interpolation between octave bands
        log_freq = np.log2(freq_hz)
        log_bands = np.log2(OCTAVE_BANDS)

        return np.interp(log_freq, log_bands, coeffs)


def create_pra_material(material_name: str):
    """
    Create a pyroomacoustics Material object with frequency-dependent absorption.

    Args:
        material_name: Name of material from MATERIAL_COEFFICIENTS

    Returns:
        pra.Material object (or dict if pyroomacoustics not available)
    """
    mat = get_material_absorption(material_name)

    if HAS_PRA:
        return pra.Material(
            energy_absorption=mat['absorption'],
            scattering=mat['scattering']
        )
    else:
        # Return dict for testing without pyroomacoustics
        return {
            'absorption': mat['absorption'],
            'scattering': mat['scattering']
        }


def get_car_interior_materials() -> dict:
    """
    Get a complete set of materials for car interior simulation.

    Returns:
        Dictionary mapping surface type to material name
    """
    return {
        'ceiling': 'car_headliner',
        'floor': 'car_carpet',
        'front': 'dashboard_plastic',  # Dashboard/firewall
        'back': 'seat_fabric',         # Rear seats/trunk
        'left': 'door_panel',
        'right': 'door_panel',
        'windows': 'glass_window',
    }


def calculate_mean_absorption(
    room_dimensions: list,
    materials: dict,
    target_freq: float = 125.0
) -> float:
    """
    Calculate room mean absorption coefficient at target frequency.

    Args:
        room_dimensions: [length, width, height] in meters
        materials: Dict mapping surface to material name
        target_freq: Frequency for absorption calculation (default 125 Hz)

    Returns:
        Area-weighted mean absorption coefficient
    """
    L, W, H = room_dimensions

    # Surface areas
    areas = {
        'ceiling': L * W,
        'floor': L * W,
        'front': W * H,
        'back': W * H,
        'left': L * H,
        'right': L * H,
    }

    total_area = sum(areas.values())
    weighted_absorption = 0.0

    for surface, area in areas.items():
        mat_name = materials.get(surface, 'dashboard_plastic')
        alpha = get_low_freq_absorption(mat_name, target_freq)
        weighted_absorption += alpha * area

    return weighted_absorption / total_area if total_area > 0 else 0.1


def get_material_summary() -> str:
    """
    Get a formatted summary of all available materials.

    Returns:
        Multi-line string with material info
    """
    lines = ["Available Materials for Low-Frequency ANC Simulation", "=" * 55, ""]

    for name, mat in MATERIAL_COEFFICIENTS.items():
        lines.append(f"{name}:")
        lines.append(f"  Description: {mat['description']}")
        lines.append(f"  125 Hz absorption: {mat['absorption'][0]:.2f}")
        lines.append(f"  Scattering: {mat['scattering']:.2f}")
        lines.append(f"  Notes: {mat['notes']}")
        lines.append("")

    return "\n".join(lines)


if __name__ == '__main__':
    print(get_material_summary())

    print("\nCar Interior Mean Absorption at 125 Hz:")
    print("-" * 40)

    # Sedan
    sedan_dims = [4.8, 1.85, 1.5]
    sedan_materials = get_car_interior_materials()
    alpha_sedan = calculate_mean_absorption(sedan_dims, sedan_materials, 125.0)
    print(f"Sedan: {alpha_sedan:.3f}")

    # SUV
    suv_dims = [4.7, 1.9, 1.8]
    alpha_suv = calculate_mean_absorption(suv_dims, sedan_materials, 125.0)
    print(f"SUV: {alpha_suv:.3f}")

    # Compact
    compact_dims = [3.5, 1.8, 1.5]
    alpha_compact = calculate_mean_absorption(compact_dims, sedan_materials, 125.0)
    print(f"Compact: {alpha_compact:.3f}")
