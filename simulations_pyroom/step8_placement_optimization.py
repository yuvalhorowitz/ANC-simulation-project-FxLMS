"""
Step 8: Single Speaker vs 4-Speaker ANC Comparison

Compares ANC performance between:
1. Single speaker at strategic locations (headrest, door, dashboard)
2. 4-speaker stereo system (all speakers working together)

Creates spatial sound heatmaps showing noise reduction across the car interior
for each configuration and scenario.

Key questions:
- Is 4-speaker better than a well-placed single speaker?
- Where is the "quiet zone" for each configuration?
- How does performance vary with noise type?
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pandas as pd
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyroomacoustics as pra
from src.acoustic.path_generator import FIRPath
from src.core.fxlms import FxNLMS
from src.noise.noise_mixer import NoiseMixer


# =============================================================================
# Car Cabin Configuration
# =============================================================================

SEDAN_DIMENSIONS = [4.5, 1.85, 1.2]  # Length, Width, Height in meters

SEDAN_MATERIALS = {
    'ceiling': 0.38,
    'floor': 0.52,
    'east': 0.14,
    'west': 0.14,
    'north': 0.20,
    'south': 0.30,
}

# =============================================================================
# Speaker Configurations to Compare
# =============================================================================

# =============================================================================
# Car Layout Reference (Sedan 4.5m x 1.85m x 1.2m)
# =============================================================================
#
#     x=0 (FRONT)                                      x=4.5 (REAR)
#     ┌──────────────────────────────────────────────────────┐
#     │  Engine   Dashboard    Front Seats    Rear Seats     │
#     │   0-0.5     0.5-1.2      1.2-2.2        2.8-4.0      │  y=0 (Driver side)
#     │                                                       │
#     │                                                       │  y=0.92 (Center)
#     │                                                       │
#     │                                                       │  y=1.85 (Passenger side)
#     └──────────────────────────────────────────────────────┘
#
# Driver head position: approximately [1.8, 0.55, 1.0]
# Passenger head position: approximately [1.8, 1.30, 1.0]

# Key positions
DRIVER_HEAD_POS = [1.8, 0.55, 1.0]
PASSENGER_HEAD_POS = [1.8, 1.30, 1.0]

# Single speaker options (strategic locations)
SINGLE_SPEAKER_CONFIGS = {
    'headrest': {
        'name': 'Headrest Speaker',
        'position': [1.9, 0.55, 1.0],  # In driver headrest
        'description': 'Speaker in driver headrest (closest to ear)',
    },
    'door': {
        'name': 'Door Speaker',
        'position': [1.5, 0.1, 0.5],   # Driver door panel
        'description': 'Driver door panel speaker',
    },
    'dashboard': {
        'name': 'Dashboard Speaker',
        'position': [0.8, 0.35, 0.9],  # Dashboard driver side
        'description': 'Dashboard driver side speaker',
    },
    'rear': {
        'name': 'Rear Shelf Speaker',
        'position': [3.5, 0.55, 0.9],  # Rear shelf
        'description': 'Rear shelf speaker behind driver',
    },
}

# 4-speaker stereo configuration
STEREO_4_SPEAKERS = {
    'door_L': [1.5, 0.1, 0.5],        # Driver door
    'door_R': [1.5, 1.75, 0.5],       # Passenger door
    'dash_L': [0.8, 0.35, 0.9],       # Dashboard driver side
    'dash_R': [0.8, 1.50, 0.9],       # Dashboard passenger side
}

# =============================================================================
# Fixed Microphone Positions
# =============================================================================

# Reference mic - near noise source (firewall)
REF_MIC_POSITION = [0.3, 0.92, 0.5]

# Error mic - at driver's ear
ERROR_MIC_POSITION = DRIVER_HEAD_POS.copy()

# =============================================================================
# Noise Scenarios
# =============================================================================

NOISE_SCENARIOS = {
    'engine': {
        'name': 'Engine Noise',
        'source_position': [0.3, 0.92, 0.4],
        'description': 'Low frequency engine rumble',
        'scenario': 'idle',
    },
    'road': {
        'name': 'Road Noise',
        'source_position': [1.0, 0.92, 0.15],
        'description': 'Tire and road surface noise',
        'scenario': 'city',
    },
    'highway': {
        'name': 'Highway (Combined)',
        'source_position': [0.5, 0.92, 0.5],
        'description': 'Mix of engine, road, and wind',
        'scenario': 'highway',
    },
}

# =============================================================================
# Grid for Spatial Heatmap
# =============================================================================

# Listener height (ear level when seated)
LISTENER_HEIGHT = 1.0

# Grid resolution for heatmap
GRID_RESOLUTION = 0.15  # meters


class ANCSimulator:
    """
    ANC simulator that can work with single or multiple speakers.
    """

    def __init__(
        self,
        speakers: Dict[str, List[float]],
        ref_mic_pos: List[float],
        error_mic_pos: List[float],
        noise_source_pos: List[float],
        room_dims: List[float] = SEDAN_DIMENSIONS,
        fs: int = 16000,
    ):
        self.fs = fs
        self.speakers = speakers
        self.speaker_names = list(speakers.keys())
        self.n_speakers = len(speakers)

        # Create room
        materials = {
            'ceiling': pra.Material(SEDAN_MATERIALS['ceiling']),
            'floor': pra.Material(SEDAN_MATERIALS['floor']),
            'east': pra.Material(SEDAN_MATERIALS['east']),
            'west': pra.Material(SEDAN_MATERIALS['west']),
            'north': pra.Material(SEDAN_MATERIALS['north']),
            'south': pra.Material(SEDAN_MATERIALS['south']),
        }

        self.room = pra.ShoeBox(
            room_dims,
            fs=fs,
            materials=materials,
            max_order=3,
            air_absorption=True
        )

        # Add noise source (source 0)
        self.room.add_source(noise_source_pos)

        # Add speakers (sources 1, 2, ...)
        for name in self.speaker_names:
            self.room.add_source(speakers[name])

        # Add microphones: [0] = reference, [1] = error
        mic_array = np.array([ref_mic_pos, error_mic_pos]).T
        self.room.add_microphone_array(pra.MicrophoneArray(mic_array, fs=fs))

        # Compute RIRs
        self.room.compute_rir()

        # Extract paths
        max_len = 512

        # Primary path: noise -> error mic
        self.H_primary = self.room.rir[1][0][:max_len]

        # Reference path: noise -> reference mic
        self.H_reference = self.room.rir[0][0][:max_len]

        # Secondary paths (speaker -> error mic)
        self.H_secondary = {}
        for i, name in enumerate(self.speaker_names):
            self.H_secondary[name] = self.room.rir[1][i + 1][:max_len]

        # Combined secondary path
        self.H_secondary_combined = np.zeros(max_len)
        for name in self.speaker_names:
            path = self.H_secondary[name]
            self.H_secondary_combined[:len(path)] += path

        # Estimate with modeling error
        self.H_secondary_est = self.H_secondary_combined * (
            1 + 0.05 * np.random.randn(len(self.H_secondary_combined))
        )

        # Create FIR path filters
        self.primary_path = FIRPath(self.H_primary)
        self.reference_path = FIRPath(self.H_reference)
        self.secondary_paths = {
            name: FIRPath(self.H_secondary[name])
            for name in self.speaker_names
        }

    def run(self, noise_signal: np.ndarray, filter_length: int = 256,
            step_size: float = 0.005) -> dict:
        """Run ANC simulation."""
        n_samples = len(noise_signal)

        # Create FxNLMS
        fxlms = FxNLMS(
            filter_length=filter_length,
            step_size=step_size,
            secondary_path_estimate=self.H_secondary_est,
            regularization=1e-4
        )

        # Reset
        self.primary_path.reset()
        self.reference_path.reset()
        for path in self.secondary_paths.values():
            path.reset()

        # Storage
        desired = np.zeros(n_samples)
        error = np.zeros(n_samples)

        for i in range(n_samples):
            sample = noise_signal[i]

            # Reference signal
            x = self.reference_path.filter_sample(sample)

            # Noise at error mic
            d = self.primary_path.filter_sample(sample)
            desired[i] = d

            # Generate anti-noise
            y = fxlms.generate_antinoise(x)

            # Through all secondary paths
            y_at_error = 0.0
            for name in self.speaker_names:
                y_at_error += self.secondary_paths[name].filter_sample(y)

            # Error
            e = d + y_at_error
            error[i] = e

            # Update
            fxlms.filter_reference(x)
            fxlms.update_weights(e)

        # Calculate noise reduction (steady state)
        steady_start = n_samples // 2
        d_power = np.mean(desired[steady_start:]**2)
        e_power = np.mean(error[steady_start:]**2)

        if e_power > 1e-10:
            nr_db = 10 * np.log10(d_power / e_power)
        else:
            nr_db = 60.0

        return {
            'noise_reduction_db': nr_db,
            'desired': desired,
            'error': error,
            'weights': fxlms.get_weights(),
        }


def compute_spatial_heatmap(
    speakers: Dict[str, List[float]],
    noise_source_pos: List[float],
    ref_mic_pos: List[float],
    scenario: str,
    room_dims: List[float] = SEDAN_DIMENSIONS,
    resolution: float = GRID_RESOLUTION,
    fs: int = 16000,
    duration: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute noise reduction at multiple points across the room.

    Returns:
        x_grid, y_grid, nr_grid: Grid coordinates and noise reduction values
    """
    length, width, height = room_dims

    # Create grid (at ear height)
    x_points = np.arange(0.3, length - 0.3, resolution)
    y_points = np.arange(0.2, width - 0.2, resolution)

    nr_grid = np.zeros((len(y_points), len(x_points)))

    # Generate noise once
    noise_gen = NoiseMixer(fs)
    noise_signal = noise_gen.generate_scenario(duration, scenario)

    total_points = len(x_points) * len(y_points)
    count = 0

    for i, y in enumerate(y_points):
        for j, x in enumerate(x_points):
            count += 1

            # Error mic at this grid point
            error_mic_pos = [x, y, LISTENER_HEIGHT]

            # Skip points inside seats and dashboard (approximate exclusion zones)
            # Dashboard: x=0.35-0.95
            # Front seats: x=1.2-2.1, y=0.2-0.8 (driver) or y=1.05-1.65 (passenger)
            # Rear seats: x=2.8-3.8
            in_dashboard = 0.35 < x < 0.95
            in_driver_seat = 1.2 < x < 2.1 and 0.2 < y < 0.8
            in_passenger_seat = 1.2 < x < 2.1 and 1.05 < y < 1.65
            in_rear_seats = 2.8 < x < 3.8

            if in_dashboard or in_driver_seat or in_passenger_seat or in_rear_seats:
                nr_grid[i, j] = np.nan
                continue

            try:
                sim = ANCSimulator(
                    speakers=speakers,
                    ref_mic_pos=ref_mic_pos,
                    error_mic_pos=error_mic_pos,
                    noise_source_pos=noise_source_pos,
                    fs=fs,
                )
                result = sim.run(noise_signal)
                nr_grid[i, j] = result['noise_reduction_db']
            except Exception as e:
                nr_grid[i, j] = np.nan

            if count % 20 == 0:
                print(f"    Progress: {count}/{total_points} points")

    x_grid, y_grid = np.meshgrid(x_points, y_points)
    return x_grid, y_grid, nr_grid


def plot_spatial_heatmap(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    nr_grid: np.ndarray,
    title: str,
    speakers: Dict[str, List[float]],
    noise_pos: List[float],
    ref_mic_pos: List[float],
    save_path: str,
):
    """Plot spatial heatmap with car interior overlay."""

    fig, ax = plt.subplots(figsize=(12, 8))

    length, width, height = SEDAN_DIMENSIONS

    # Plot heatmap
    masked_nr = np.ma.masked_invalid(nr_grid)

    # Color scale: red (negative/bad) to green (positive/good)
    vmin = max(-5, np.nanmin(nr_grid))
    vmax = min(25, np.nanmax(nr_grid))

    im = ax.pcolormesh(x_grid, y_grid, masked_nr,
                       cmap='RdYlGn', vmin=vmin, vmax=vmax, shading='auto')

    # Add contour lines
    try:
        contours = ax.contour(x_grid, y_grid, masked_nr,
                              levels=[0, 5, 10, 15, 20], colors='black',
                              linewidths=0.5, alpha=0.5)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%d dB')
    except:
        pass

    # Draw car interior
    # Outline
    car = plt.Rectangle((0, 0), length, width, fill=False,
                         edgecolor='#333', linewidth=3)
    ax.add_patch(car)

    # Windshield (front of car at x=0)
    windshield = plt.Polygon([[0, 0.15], [0, width-0.15], [0.3, width-0.25], [0.3, 0.25]],
                              fill=True, facecolor='#87CEEB', edgecolor='#333',
                              alpha=0.4, linewidth=2)
    ax.add_patch(windshield)
    ax.text(0.15, width/2, 'FRONT', ha='center', va='center', fontsize=8,
            color='#333', rotation=90, fontweight='bold')

    # Rear window
    rear_window = plt.Polygon([[length, 0.2], [length, width-0.2],
                                [length-0.25, width-0.3], [length-0.25, 0.3]],
                               fill=True, facecolor='#87CEEB', edgecolor='#333',
                               alpha=0.4, linewidth=2)
    ax.add_patch(rear_window)

    # Dashboard (x=0.4 to 1.0)
    dashboard = plt.Rectangle((0.35, 0.2), 0.6, width-0.4, fill=True,
                                facecolor='#555', edgecolor='#333', alpha=0.4)
    ax.add_patch(dashboard)
    ax.text(0.65, width/2, 'Dashboard', ha='center', va='center', fontsize=7, color='white')

    # Front seats (x=1.2 to 2.2) - where driver and passenger sit
    # Driver seat (left side, y=0.2 to 0.8)
    driver_seat = plt.Rectangle((1.2, 0.2), 0.9, 0.6, fill=True,
                                  facecolor='#8B4513', edgecolor='#333', alpha=0.6)
    ax.add_patch(driver_seat)
    ax.text(1.65, 0.5, 'Driver\nSeat', ha='center', va='center', fontsize=8, color='white')

    # Passenger seat (right side, y=1.05 to 1.65)
    passenger_seat = plt.Rectangle((1.2, 1.05), 0.9, 0.6, fill=True,
                                     facecolor='#8B4513', edgecolor='#333', alpha=0.6)
    ax.add_patch(passenger_seat)
    ax.text(1.65, 1.35, 'Passenger\nSeat', ha='center', va='center', fontsize=8, color='white')

    # Center console
    console = plt.Rectangle((1.0, 0.8), 1.2, 0.25, fill=True,
                              facecolor='#444', edgecolor='#333', alpha=0.4)
    ax.add_patch(console)

    # Rear seats (x=2.8 to 4.0)
    rear_seats = plt.Rectangle((2.8, 0.2), 1.0, width-0.4, fill=True,
                                 facecolor='#A0522D', edgecolor='#333', alpha=0.4)
    ax.add_patch(rear_seats)
    ax.text(3.3, width/2, 'Rear Seats', ha='center', va='center', fontsize=8, color='white')

    # Plot speakers
    for name, pos in speakers.items():
        ax.scatter(pos[0], pos[1], s=200, c='#2ecc71', marker='o',
                   edgecolors='white', linewidths=2, zorder=20)
        ax.annotate(name.replace('_', '\n'), (pos[0], pos[1]),
                    textcoords="offset points", xytext=(0, 15),
                    ha='center', fontsize=7, fontweight='bold', color='#27ae60')

    # Plot noise source
    ax.scatter(noise_pos[0], noise_pos[1], s=250, c='#e74c3c', marker='s',
               edgecolors='white', linewidths=2, zorder=20)
    ax.annotate('Noise\nSource', (noise_pos[0], noise_pos[1]),
                textcoords="offset points", xytext=(20, 0),
                ha='left', fontsize=8, color='#c0392b')

    # Plot reference mic
    ax.scatter(ref_mic_pos[0], ref_mic_pos[1], s=150, c='#3498db', marker='^',
               edgecolors='white', linewidths=2, zorder=20)
    ax.annotate('Ref Mic', (ref_mic_pos[0], ref_mic_pos[1]),
                textcoords="offset points", xytext=(0, -20),
                ha='center', fontsize=8, color='#2980b9')

    # Mark driver ear position (head is above/behind seat)
    driver_ear = DRIVER_HEAD_POS
    ax.scatter(driver_ear[0], driver_ear[1], s=150, c='yellow', marker='*',
               edgecolors='black', linewidths=1.5, zorder=25)
    ax.annotate('Driver\nEar', (driver_ear[0], driver_ear[1]),
                textcoords="offset points", xytext=(15, 5),
                ha='left', fontsize=8, color='#333', fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Noise Reduction (dB)', fontsize=11)

    # Labels
    ax.set_xlim(-0.2, length + 0.2)
    ax.set_ylim(-0.2, width + 0.2)
    ax.set_aspect('equal')
    ax.set_xlabel('Length (m) ← Front | Rear →', fontsize=11)
    ax.set_ylabel('Width (m)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add front arrow
    ax.annotate('', xy=(-0.1, 0.92), xytext=(0.3, 0.92),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.text(-0.15, 0.92, 'FRONT', ha='right', fontsize=9, color='gray', rotation=90)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def run_comparison(scenario_name: str, scenario_config: dict, save_dir: str):
    """
    Run comparison between single speaker configs and 4-speaker stereo.
    """
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_config['name']}")
    print(f"{'='*60}")

    noise_pos = scenario_config['source_position']
    scenario = scenario_config['scenario']

    results = {}

    # Test each single speaker configuration
    for spk_key, spk_config in SINGLE_SPEAKER_CONFIGS.items():
        print(f"\n  Testing: {spk_config['name']}")

        speakers = {spk_key: spk_config['position']}

        x_grid, y_grid, nr_grid = compute_spatial_heatmap(
            speakers=speakers,
            noise_source_pos=noise_pos,
            ref_mic_pos=REF_MIC_POSITION,
            scenario=scenario,
            resolution=GRID_RESOLUTION,
            duration=2.0,
        )

        # Get noise reduction at driver ear
        driver_y_idx = np.argmin(np.abs(y_grid[:, 0] - DRIVER_HEAD_POS[1]))
        driver_x_idx = np.argmin(np.abs(x_grid[0, :] - DRIVER_HEAD_POS[0]))
        driver_ear_nr = nr_grid[driver_y_idx, driver_x_idx]

        results[spk_key] = {
            'name': spk_config['name'],
            'type': 'single',
            'nr_at_ear': float(driver_ear_nr) if not np.isnan(driver_ear_nr) else 0,
            'nr_mean': float(np.nanmean(nr_grid)),
            'nr_max': float(np.nanmax(nr_grid)),
            'x_grid': x_grid,
            'y_grid': y_grid,
            'nr_grid': nr_grid,
        }

        # Plot heatmap
        plot_spatial_heatmap(
            x_grid, y_grid, nr_grid,
            title=f"{spk_config['name']} - {scenario_config['name']}\n"
                  f"Reduction at ear: {driver_ear_nr:.1f} dB",
            speakers=speakers,
            noise_pos=noise_pos,
            ref_mic_pos=REF_MIC_POSITION,
            save_path=f"{save_dir}/pyroom_step8_heatmap_{scenario_name}_{spk_key}.png"
        )

    # Test 4-speaker stereo
    print(f"\n  Testing: 4-Speaker Stereo")

    x_grid, y_grid, nr_grid = compute_spatial_heatmap(
        speakers=STEREO_4_SPEAKERS,
        noise_source_pos=noise_pos,
        ref_mic_pos=REF_MIC_POSITION,
        scenario=scenario,
        resolution=GRID_RESOLUTION,
        duration=2.0,
    )

    driver_ear_nr = nr_grid[
        np.argmin(np.abs(y_grid[:, 0] - 0.55)),
        np.argmin(np.abs(x_grid[0, :] - 3.2))
    ]

    results['stereo_4'] = {
        'name': '4-Speaker Stereo',
        'type': '4-speaker',
        'nr_at_ear': float(driver_ear_nr) if not np.isnan(driver_ear_nr) else 0,
        'nr_mean': float(np.nanmean(nr_grid)),
        'nr_max': float(np.nanmax(nr_grid)),
        'x_grid': x_grid,
        'y_grid': y_grid,
        'nr_grid': nr_grid,
    }

    plot_spatial_heatmap(
        x_grid, y_grid, nr_grid,
        title=f"4-Speaker Stereo - {scenario_config['name']}\n"
              f"Reduction at ear: {driver_ear_nr:.1f} dB",
        speakers=STEREO_4_SPEAKERS,
        noise_pos=noise_pos,
        ref_mic_pos=REF_MIC_POSITION,
        save_path=f"{save_dir}/pyroom_step8_heatmap_{scenario_name}_stereo4.png"
    )

    return results


def plot_comparison_summary(all_results: dict, save_dir: str):
    """Create summary comparison chart."""

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    configs = list(SINGLE_SPEAKER_CONFIGS.keys()) + ['stereo_4']
    config_names = [SINGLE_SPEAKER_CONFIGS.get(c, {}).get('name', '4-Speaker Stereo')
                    for c in configs]

    colors = ['#3498db', '#9b59b6', '#e67e22', '#1abc9c', '#2ecc71']

    for idx, (scenario_name, scenario_results) in enumerate(all_results.items()):
        ax = axes[idx]

        nr_at_ear = [scenario_results[c]['nr_at_ear'] for c in configs]

        bars = ax.barh(range(len(configs)), nr_at_ear, color=colors)

        ax.set_yticks(range(len(configs)))
        ax.set_yticklabels(config_names)
        ax.set_xlabel('Noise Reduction at Driver Ear (dB)')
        ax.set_title(f"{NOISE_SCENARIOS[scenario_name]['name']}", fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, nr_at_ear)):
            if not np.isnan(val):
                ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}', va='center', fontsize=9)

        # Highlight best
        if not all(np.isnan(nr_at_ear)):
            best_idx = np.nanargmax(nr_at_ear)
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)

    plt.suptitle('Single Speaker vs 4-Speaker Stereo ANC Comparison\n'
                 '(Gold border = best configuration)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/pyroom_step8_comparison_summary.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir}/pyroom_step8_comparison_summary.png")
    plt.close()


def plot_all_heatmaps_grid(all_results: dict, save_dir: str):
    """Create grid of all heatmaps for easy comparison."""

    scenarios = list(NOISE_SCENARIOS.keys())
    configs = list(SINGLE_SPEAKER_CONFIGS.keys()) + ['stereo_4']

    n_scenarios = len(scenarios)
    n_configs = len(configs)

    fig, axes = plt.subplots(n_scenarios, n_configs, figsize=(4*n_configs, 3.5*n_scenarios))

    for i, scenario_name in enumerate(scenarios):
        for j, config in enumerate(configs):
            ax = axes[i, j] if n_scenarios > 1 else axes[j]

            result = all_results[scenario_name][config]
            nr_grid = result['nr_grid']
            x_grid = result['x_grid']
            y_grid = result['y_grid']

            masked_nr = np.ma.masked_invalid(nr_grid)

            im = ax.pcolormesh(x_grid, y_grid, masked_nr,
                               cmap='RdYlGn', vmin=-5, vmax=20, shading='auto')

            # Car outline
            car = plt.Rectangle((0, 0), SEDAN_DIMENSIONS[0], SEDAN_DIMENSIONS[1],
                                  fill=False, edgecolor='#333', linewidth=1)
            ax.add_patch(car)

            # Front seats (simplified)
            driver_seat = plt.Rectangle((1.2, 0.2), 0.9, 0.6, fill=True,
                                          facecolor='#8B4513', alpha=0.4)
            ax.add_patch(driver_seat)
            pass_seat = plt.Rectangle((1.2, 1.05), 0.9, 0.6, fill=True,
                                        facecolor='#8B4513', alpha=0.4)
            ax.add_patch(pass_seat)

            # Driver ear position
            ax.scatter(DRIVER_HEAD_POS[0], DRIVER_HEAD_POS[1], s=50, c='yellow', marker='*',
                       edgecolors='black', linewidths=0.5, zorder=10)

            ax.set_xlim(-0.1, SEDAN_DIMENSIONS[0] + 0.1)
            ax.set_ylim(-0.1, SEDAN_DIMENSIONS[1] + 0.1)
            ax.set_aspect('equal')

            nr_at_ear = result['nr_at_ear']
            title = f"{result['name']}\n{nr_at_ear:.1f} dB at ear"
            ax.set_title(title, fontsize=9)

            if j == 0:
                ax.set_ylabel(NOISE_SCENARIOS[scenario_name]['name'], fontsize=10)

            ax.set_xticks([])
            ax.set_yticks([])

    # Add colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Noise Reduction (dB)')

    plt.suptitle('Spatial Noise Reduction: Single Speaker vs 4-Speaker Stereo\n'
                 '(Star = Driver Ear Position)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.savefig(f'{save_dir}/pyroom_step8_heatmap_grid.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir}/pyroom_step8_heatmap_grid.png")
    plt.close()


def main():
    """Run the complete comparison study."""
    print("=" * 70)
    print("Step 8: Single Speaker vs 4-Speaker ANC Comparison")
    print("=" * 70)
    print("\nThis study compares:")
    print("  - Single speaker at strategic locations")
    print("  - 4-speaker stereo system")
    print("\nGenerating spatial heatmaps showing 'quiet zones'...")

    save_dir = 'output/plots'
    data_dir = 'output/data'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    all_results = {}

    # Run for each scenario
    for scenario_name, scenario_config in NOISE_SCENARIOS.items():
        results = run_comparison(scenario_name, scenario_config, save_dir)
        all_results[scenario_name] = results

    # Generate summary plots
    print("\n" + "=" * 70)
    print("GENERATING SUMMARY VISUALIZATIONS")
    print("=" * 70)

    plot_comparison_summary(all_results, save_dir)
    plot_all_heatmaps_grid(all_results, save_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for scenario_name, results in all_results.items():
        print(f"\n{NOISE_SCENARIOS[scenario_name]['name']}:")
        print("-" * 40)

        sorted_results = sorted(results.items(),
                                key=lambda x: x[1]['nr_at_ear'] if not np.isnan(x[1]['nr_at_ear']) else -999,
                                reverse=True)

        for rank, (config, data) in enumerate(sorted_results, 1):
            nr = data['nr_at_ear']
            print(f"  {rank}. {data['name']}: {nr:.1f} dB at driver ear")

    # Save results to JSON
    json_results = {}
    for scenario_name, results in all_results.items():
        json_results[scenario_name] = {
            config: {
                'name': data['name'],
                'type': data['type'],
                'nr_at_ear': data['nr_at_ear'],
                'nr_mean': data['nr_mean'],
                'nr_max': data['nr_max'],
            }
            for config, data in results.items()
        }

    with open(f'{data_dir}/pyroom_step8_comparison_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {data_dir}/pyroom_step8_comparison_results.json")

    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    print("\nHeatmaps (one per configuration per scenario):")
    print("  output/plots/pyroom_step8_heatmap_<scenario>_<config>.png")
    print("\nSummary plots:")
    print("  output/plots/pyroom_step8_comparison_summary.png  <- Bar chart comparison")
    print("  output/plots/pyroom_step8_heatmap_grid.png        <- All heatmaps in grid")
    print("\nData:")
    print("  output/data/pyroom_step8_comparison_results.json")

    return all_results


if __name__ == '__main__':
    all_results = main()
