"""
Step 8: Microphone Placement Optimization for Multi-Speaker Car ANC

Goal: Find optimal reference and error microphone locations when using
      ALL 4 car stereo speakers simultaneously for anti-noise generation.

This simulation:
1. Uses 4 speakers working together (front doors + dashboard)
2. Tests multiple reference microphone locations
3. Tests multiple error microphone locations
4. Runs across different noise types and source locations:
   - Engine noise (from firewall)
   - Road noise (from floor/wheels)
   - Wind noise (from windshield)
5. Generates recommendations for optimal mic placement

Key question: Given 4 speakers working together, where should we place
the reference and error microphones for best ANC performance?

Target frequency range: 20-300 Hz
"""

import numpy as np
import matplotlib.pyplot as plt
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
# 4-Speaker Configuration (Fixed)
# =============================================================================
# These 4 speakers work together to produce anti-noise

SPEAKERS_4CH = {
    'front_left': [2.0, 0.1, 0.4],      # Driver door
    'front_right': [2.0, 1.75, 0.4],    # Passenger door
    'dash_left': [0.8, 0.25, 0.9],      # Dashboard driver side
    'dash_right': [0.8, 1.60, 0.9],     # Dashboard passenger side
}

# =============================================================================
# Reference Microphone Candidates
# =============================================================================

REF_MIC_POSITIONS = {
    'firewall_center': [0.3, 0.92, 0.5],
    'firewall_driver': [0.3, 0.45, 0.5],
    'dashboard': [0.9, 0.92, 0.8],
    'a_pillar_left': [0.7, 0.15, 1.0],
    'a_pillar_right': [0.7, 1.70, 1.0],
    'under_driver_seat': [2.5, 0.55, 0.15],
    'floor_front_left': [1.0, 0.15, 0.1],
    'floor_front_right': [1.0, 1.70, 0.1],
}

# =============================================================================
# Error Microphone Candidates
# =============================================================================

ERROR_MIC_POSITIONS = {
    'driver_headrest': [3.2, 0.55, 1.0],
    'driver_ear_left': [3.2, 0.40, 1.0],
    'driver_ear_right': [3.2, 0.70, 1.0],
    'passenger_headrest': [3.2, 1.30, 1.0],
    'center_cabin': [2.5, 0.92, 0.9],
    'rearview_mirror': [1.5, 0.92, 1.1],
}

# =============================================================================
# Noise Source Configurations
# =============================================================================

NOISE_CONFIGS = {
    'engine': {
        'name': 'Engine Noise',
        'description': 'Low frequency engine rumble from firewall',
        'source_position': [0.3, 0.92, 0.4],
        'noise_type': 'engine',
        'dominant_freqs': [30, 60, 90, 120],
    },
    'road_front': {
        'name': 'Road Noise (Front)',
        'description': 'Tire/road noise from front wheel wells',
        'source_position': [0.8, 0.92, 0.15],
        'noise_type': 'road',
        'dominant_freqs': [50, 100, 150, 200],
    },
    'road_rear': {
        'name': 'Road Noise (Rear)',
        'description': 'Tire/road noise from rear wheel wells',
        'source_position': [3.8, 0.92, 0.15],
        'noise_type': 'road',
        'dominant_freqs': [50, 100, 150, 200],
    },
    'wind': {
        'name': 'Wind Noise',
        'description': 'Aerodynamic noise from windshield/A-pillars',
        'source_position': [0.6, 0.92, 1.0],
        'noise_type': 'wind',
        'dominant_freqs': [100, 200, 300],
    },
    'combined': {
        'name': 'Combined (Highway)',
        'description': 'Mix of engine, road, and wind noise',
        'source_position': [0.5, 0.92, 0.5],  # Approximate center of noise sources
        'noise_type': 'highway',
        'dominant_freqs': [50, 100, 150, 200],
    },
}


class MultiSpeakerANC:
    """
    ANC system using multiple speakers simultaneously.

    All 4 speakers receive the same anti-noise signal, which then
    propagates through their individual secondary paths to the error mic.
    """

    def __init__(
        self,
        speakers: Dict[str, List[float]],
        ref_mic_pos: List[float],
        error_mic_pos: List[float],
        noise_source_pos: List[float],
        room_dims: List[float] = SEDAN_DIMENSIONS,
        materials: Dict[str, float] = SEDAN_MATERIALS,
        fs: int = 16000,
    ):
        self.fs = fs
        self.speakers = speakers
        self.ref_mic_pos = ref_mic_pos
        self.error_mic_pos = error_mic_pos
        self.noise_source_pos = noise_source_pos

        # Create room
        pra_materials = {
            'ceiling': pra.Material(materials['ceiling']),
            'floor': pra.Material(materials['floor']),
            'east': pra.Material(materials['east']),
            'west': pra.Material(materials['west']),
            'north': pra.Material(materials['north']),
            'south': pra.Material(materials['south']),
        }

        self.room = pra.ShoeBox(
            room_dims,
            fs=fs,
            materials=pra_materials,
            max_order=3,
            air_absorption=True
        )

        # Add noise source (source 0)
        self.room.add_source(noise_source_pos)

        # Add speakers (sources 1, 2, 3, 4)
        self.speaker_names = list(speakers.keys())
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

        # Secondary paths: each speaker -> error mic
        self.H_secondary = {}
        for i, name in enumerate(self.speaker_names):
            rir = self.room.rir[1][i + 1][:max_len]  # +1 because source 0 is noise
            self.H_secondary[name] = rir

        # Combined secondary path (sum of all speaker paths)
        # This represents what the error mic hears from all speakers
        self.H_secondary_combined = np.zeros(max_len)
        for name in self.speaker_names:
            path = self.H_secondary[name]
            self.H_secondary_combined[:len(path)] += path

        # Estimate of secondary path (with 5% error for realism)
        self.H_secondary_est = self.H_secondary_combined * (1 + 0.05 * np.random.randn(len(self.H_secondary_combined)))

        # Create FIR filters
        self.primary_path = FIRPath(self.H_primary)
        self.reference_path = FIRPath(self.H_reference)
        self.secondary_paths = {name: FIRPath(self.H_secondary[name]) for name in self.speaker_names}

    def run_simulation(
        self,
        noise_signal: np.ndarray,
        filter_length: int = 256,
        step_size: float = 0.005,
    ) -> dict:
        """
        Run ANC simulation with multi-speaker setup.

        The same anti-noise signal is sent to all speakers, each with
        its own acoustic path to the error microphone.
        """
        n_samples = len(noise_signal)

        # Create FxNLMS with combined secondary path estimate
        fxlms = FxNLMS(
            filter_length=filter_length,
            step_size=step_size,
            secondary_path_estimate=self.H_secondary_est,
            regularization=1e-4
        )

        # Reset filters
        self.primary_path.reset()
        self.reference_path.reset()
        for path in self.secondary_paths.values():
            path.reset()

        # Storage
        desired = np.zeros(n_samples)
        error = np.zeros(n_samples)
        mse = np.zeros(n_samples)

        for i in range(n_samples):
            sample = noise_signal[i]

            # Reference signal (noise at reference mic)
            x = self.reference_path.filter_sample(sample)

            # Desired signal (noise at error mic via primary path)
            d = self.primary_path.filter_sample(sample)
            desired[i] = d

            # Generate anti-noise
            y = fxlms.generate_antinoise(x)

            # Anti-noise through all secondary paths (same signal to all speakers)
            y_at_error = 0.0
            for name in self.speaker_names:
                y_at_error += self.secondary_paths[name].filter_sample(y)

            # Error signal
            e = d + y_at_error
            error[i] = e
            mse[i] = e ** 2

            # Update FxLMS
            fxlms.filter_reference(x)
            fxlms.update_weights(e)

        # Calculate noise reduction (steady state - last 50%)
        steady_start = n_samples // 2
        d_power = np.mean(desired[steady_start:]**2)
        e_power = np.mean(error[steady_start:]**2)

        if e_power > 1e-10:
            nr_db = 10 * np.log10(d_power / e_power)
        else:
            nr_db = 60.0

        # Convergence time (time to reach 90% of final reduction)
        window = 500
        if len(mse) > window:
            mse_smooth = np.convolve(mse, np.ones(window)/window, mode='valid')
            final_mse = np.mean(mse_smooth[-window:])
            target_mse = final_mse + 0.1 * (mse_smooth[0] - final_mse)
            conv_idx = np.argmax(mse_smooth < target_mse)
            conv_time = conv_idx / self.fs if conv_idx > 0 else len(noise_signal) / self.fs
        else:
            conv_time = len(noise_signal) / self.fs

        return {
            'noise_reduction_db': nr_db,
            'convergence_time_s': conv_time,
            'final_mse': np.mean(mse[-1000:]) if len(mse) > 1000 else np.mean(mse),
            'desired': desired,
            'error': error,
            'mse': mse,
            'weights': fxlms.get_weights(),
        }


class PlacementOptimizer:
    """
    Tests microphone placements with 4-speaker ANC system.
    """

    def __init__(self, fs: int = 16000, duration: float = 3.0):
        self.fs = fs
        self.duration = duration
        self.noise_gen = NoiseMixer(fs)
        self.results = []

    def generate_noise(self, noise_config: dict) -> np.ndarray:
        """Generate noise signal based on configuration."""
        noise_type = noise_config['noise_type']

        if noise_type == 'engine':
            return self.noise_gen.generate_scenario(self.duration, 'idle')
        elif noise_type == 'road':
            return self.noise_gen.generate_scenario(self.duration, 'highway')
        elif noise_type == 'wind':
            return self.noise_gen.generate_scenario(self.duration, 'highway')
        elif noise_type == 'highway':
            return self.noise_gen.generate_scenario(self.duration, 'highway')
        else:
            return self.noise_gen.generate_scenario(self.duration, 'highway')

    def run_single_test(
        self,
        ref_mic_name: str,
        error_mic_name: str,
        noise_name: str,
        verbose: bool = False,
    ) -> dict:
        """Run a single placement test."""

        ref_mic_pos = REF_MIC_POSITIONS[ref_mic_name]
        error_mic_pos = ERROR_MIC_POSITIONS[error_mic_name]
        noise_config = NOISE_CONFIGS[noise_name]

        try:
            # Create multi-speaker ANC system
            anc = MultiSpeakerANC(
                speakers=SPEAKERS_4CH,
                ref_mic_pos=ref_mic_pos,
                error_mic_pos=error_mic_pos,
                noise_source_pos=noise_config['source_position'],
                fs=self.fs,
            )

            # Generate noise
            noise_signal = self.generate_noise(noise_config)

            # Run simulation
            result = anc.run_simulation(noise_signal)

            return {
                'success': True,
                'ref_mic': ref_mic_name,
                'error_mic': error_mic_name,
                'noise_type': noise_name,
                'noise_reduction_db': result['noise_reduction_db'],
                'convergence_time_s': result['convergence_time_s'],
                'final_mse': result['final_mse'],
            }

        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
            return {
                'success': False,
                'ref_mic': ref_mic_name,
                'error_mic': error_mic_name,
                'noise_type': noise_name,
                'noise_reduction_db': -999,
                'convergence_time_s': -1,
                'final_mse': -1,
                'error': str(e),
            }

    def run_sweep(self, verbose: bool = True) -> pd.DataFrame:
        """
        Run all placement combinations.

        Tests: ref_mics × error_mics × noise_types
        """
        ref_mics = list(REF_MIC_POSITIONS.keys())
        error_mics = list(ERROR_MIC_POSITIONS.keys())
        noise_types = list(NOISE_CONFIGS.keys())

        total = len(ref_mics) * len(error_mics) * len(noise_types)

        print(f"\nMulti-Speaker Placement Optimization")
        print(f"=" * 60)
        print(f"Speakers: {len(SPEAKERS_4CH)} (fixed configuration)")
        print(f"  - {', '.join(SPEAKERS_4CH.keys())}")
        print(f"Reference mic positions: {len(ref_mics)}")
        print(f"Error mic positions: {len(error_mics)}")
        print(f"Noise types: {len(noise_types)}")
        print(f"Total configurations: {total}")
        print(f"=" * 60)

        results = []
        start_time = datetime.now()
        count = 0

        for noise_name in noise_types:
            print(f"\nTesting noise: {NOISE_CONFIGS[noise_name]['name']}")

            for ref_mic in ref_mics:
                for error_mic in error_mics:
                    count += 1

                    if verbose and count % 10 == 0:
                        elapsed = (datetime.now() - start_time).seconds
                        print(f"  [{count}/{total}] {ref_mic} -> {error_mic} (elapsed: {elapsed}s)")

                    result = self.run_single_test(ref_mic, error_mic, noise_name, verbose=False)
                    results.append(result)

        self.results = results
        df = pd.DataFrame(results)

        # Save results
        os.makedirs('output/data', exist_ok=True)
        df.to_csv('output/data/pyroom_step8_sweep_results.csv', index=False)

        print(f"\nSweep complete! Tested {len(df)} configurations")
        print(f"Results saved to: output/data/pyroom_step8_sweep_results.csv")

        return df

    def analyze_results(self, df: pd.DataFrame = None) -> dict:
        """Analyze sweep results."""
        if df is None:
            df = pd.DataFrame(self.results)

        df_valid = df[df['success'] == True].copy()

        if len(df_valid) == 0:
            return {'error': 'No valid results'}

        analysis = {}

        # Best overall
        best_idx = df_valid['noise_reduction_db'].idxmax()
        best = df_valid.loc[best_idx]
        analysis['best_overall'] = {
            'ref_mic': best['ref_mic'],
            'error_mic': best['error_mic'],
            'noise_type': best['noise_type'],
            'noise_reduction_db': float(best['noise_reduction_db']),
        }

        # Best per noise type
        analysis['best_per_noise'] = {}
        for noise_type in df_valid['noise_type'].unique():
            df_noise = df_valid[df_valid['noise_type'] == noise_type]
            best_idx = df_noise['noise_reduction_db'].idxmax()
            best = df_noise.loc[best_idx]
            analysis['best_per_noise'][noise_type] = {
                'ref_mic': best['ref_mic'],
                'error_mic': best['error_mic'],
                'noise_reduction_db': float(best['noise_reduction_db']),
            }

        # Average by ref_mic
        ref_mic_avg = df_valid.groupby('ref_mic')['noise_reduction_db'].mean().sort_values(ascending=False)
        analysis['ref_mic_ranking'] = {k: float(v) for k, v in ref_mic_avg.items()}

        # Average by error_mic
        error_mic_avg = df_valid.groupby('error_mic')['noise_reduction_db'].mean().sort_values(ascending=False)
        analysis['error_mic_ranking'] = {k: float(v) for k, v in error_mic_avg.items()}

        # Average by noise type
        noise_avg = df_valid.groupby('noise_type')['noise_reduction_db'].mean().sort_values(ascending=False)
        analysis['noise_type_ranking'] = {k: float(v) for k, v in noise_avg.items()}

        return analysis


def plot_car_interior_layout(save_dir: str = 'output/plots'):
    """
    Create a top-down car interior visualization showing all mic and speaker positions.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 10))

    # Car dimensions
    length, width, height = SEDAN_DIMENSIONS

    # Draw car outline
    car_outline = plt.Rectangle((0, 0), length, width,
                                  fill=False, edgecolor='#333333', linewidth=3)
    ax.add_patch(car_outline)

    # Draw windshield (front)
    windshield = plt.Polygon([[0, 0.1], [0, width-0.1], [0.4, width-0.3], [0.4, 0.3]],
                              fill=True, facecolor='#87CEEB', edgecolor='#333', alpha=0.5)
    ax.add_patch(windshield)

    # Draw rear window
    rear_window = plt.Polygon([[length, 0.15], [length, width-0.15],
                                [length-0.35, width-0.35], [length-0.35, 0.35]],
                               fill=True, facecolor='#87CEEB', edgecolor='#333', alpha=0.5)
    ax.add_patch(rear_window)

    # Draw seats
    seat_color = '#8B4513'
    seat_alpha = 0.3

    # Driver seat
    driver_seat = plt.Rectangle((2.8, 0.25), 0.8, 0.55,
                                  fill=True, facecolor=seat_color,
                                  edgecolor='#333', alpha=seat_alpha, linewidth=1)
    ax.add_patch(driver_seat)
    ax.text(3.2, 0.52, 'Driver\nSeat', ha='center', va='center', fontsize=8, color='#333')

    # Passenger seat
    passenger_seat = plt.Rectangle((2.8, 1.05), 0.8, 0.55,
                                     fill=True, facecolor=seat_color,
                                     edgecolor='#333', alpha=seat_alpha, linewidth=1)
    ax.add_patch(passenger_seat)
    ax.text(3.2, 1.32, 'Passenger\nSeat', ha='center', va='center', fontsize=8, color='#333')

    # Rear seats
    rear_seat = plt.Rectangle((3.7, 0.2), 0.6, 1.45,
                                fill=True, facecolor=seat_color,
                                edgecolor='#333', alpha=seat_alpha, linewidth=1)
    ax.add_patch(rear_seat)
    ax.text(4.0, 0.92, 'Rear\nSeats', ha='center', va='center', fontsize=8, color='#333')

    # Dashboard area
    dashboard = plt.Rectangle((0.4, 0.15), 0.6, width-0.3,
                                fill=True, facecolor='#696969',
                                edgecolor='#333', alpha=0.3, linewidth=1)
    ax.add_patch(dashboard)
    ax.text(0.7, 0.92, 'Dashboard', ha='center', va='center', fontsize=8, color='#333')

    # Center console
    console = plt.Rectangle((1.5, 0.75), 1.5, 0.34,
                              fill=True, facecolor='#A0A0A0',
                              edgecolor='#333', alpha=0.3, linewidth=1)
    ax.add_patch(console)

    # =========================================================================
    # Plot 4 Speakers (Fixed) - Large green circles
    # =========================================================================
    for name, pos in SPEAKERS_4CH.items():
        ax.scatter(pos[0], pos[1], s=300, c='#2ecc71', marker='o',
                   edgecolors='white', linewidths=2, zorder=10)
        # Add speaker icon text
        ax.annotate(f'🔊\n{name.replace("_", " ").title()}',
                    (pos[0], pos[1]),
                    textcoords="offset points",
                    xytext=(0, -30),
                    ha='center', fontsize=7, color='#27ae60', fontweight='bold')

    # =========================================================================
    # Plot Reference Mic Positions - Blue triangles
    # =========================================================================
    ref_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(REF_MIC_POSITIONS)))
    for i, (name, pos) in enumerate(REF_MIC_POSITIONS.items()):
        ax.scatter(pos[0], pos[1], s=200, c=[ref_colors[i]], marker='^',
                   edgecolors='white', linewidths=1.5, zorder=15)
        # Offset labels to avoid overlap
        offset_y = 15 if i % 2 == 0 else -25
        ax.annotate(name.replace('_', '\n'),
                    (pos[0], pos[1]),
                    textcoords="offset points",
                    xytext=(0, offset_y),
                    ha='center', fontsize=6, color='#2980b9')

    # =========================================================================
    # Plot Error Mic Positions - Purple diamonds
    # =========================================================================
    err_colors = plt.cm.Purples(np.linspace(0.4, 0.9, len(ERROR_MIC_POSITIONS)))
    for i, (name, pos) in enumerate(ERROR_MIC_POSITIONS.items()):
        ax.scatter(pos[0], pos[1], s=200, c=[err_colors[i]], marker='D',
                   edgecolors='white', linewidths=1.5, zorder=15)
        # Offset labels
        offset_y = 20 if i % 2 == 0 else -25
        offset_x = 10 if 'right' in name else -10 if 'left' in name else 0
        ax.annotate(name.replace('_', '\n'),
                    (pos[0], pos[1]),
                    textcoords="offset points",
                    xytext=(offset_x, offset_y),
                    ha='center', fontsize=6, color='#8e44ad')

    # =========================================================================
    # Plot Noise Source Positions - Red squares
    # =========================================================================
    for name, config in NOISE_CONFIGS.items():
        pos = config['source_position']
        ax.scatter(pos[0], pos[1], s=250, c='#e74c3c', marker='s',
                   edgecolors='white', linewidths=1.5, zorder=12, alpha=0.7)
        ax.annotate(f'Noise:\n{name}',
                    (pos[0], pos[1]),
                    textcoords="offset points",
                    xytext=(20, 0),
                    ha='left', fontsize=6, color='#c0392b',
                    arrowprops=dict(arrowstyle='->', color='#c0392b', lw=0.5))

    # =========================================================================
    # Add labels and dimensions
    # =========================================================================
    ax.set_xlim(-0.5, length + 0.5)
    ax.set_ylim(-0.5, width + 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('Length (m) - Front ← → Rear', fontsize=11)
    ax.set_ylabel('Width (m) - Driver ↓ ↑ Passenger', fontsize=11)
    ax.set_title(f'Car Interior Layout - Microphone & Speaker Positions\n'
                 f'Sedan: {length}m × {width}m × {height}m', fontsize=14, fontweight='bold')

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add arrow showing car direction
    ax.annotate('', xy=(0.2, -0.3), xytext=(0.8, -0.3),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.text(0.5, -0.4, 'FRONT', ha='center', fontsize=9, color='gray')

    # =========================================================================
    # Add legend
    # =========================================================================
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71',
               markersize=15, label=f'Speakers (4 fixed)', markeredgecolor='white'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#3498db',
               markersize=12, label=f'Reference Mics ({len(REF_MIC_POSITIONS)} tested)', markeredgecolor='white'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#9b59b6',
               markersize=10, label=f'Error Mics ({len(ERROR_MIC_POSITIONS)} tested)', markeredgecolor='white'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#e74c3c',
               markersize=10, label=f'Noise Sources ({len(NOISE_CONFIGS)} types)', markeredgecolor='white', alpha=0.7),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/pyroom_step8_car_layout.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir}/pyroom_step8_car_layout.png")
    plt.close()


def plot_car_interior_results(df: pd.DataFrame, analysis: dict, save_dir: str = 'output/plots'):
    """
    Create car interior visualization highlighting the best mic positions.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Car dimensions
    length, width, height = SEDAN_DIMENSIONS

    for ax_idx, ax in enumerate(axes):
        # Draw car outline
        car_outline = plt.Rectangle((0, 0), length, width,
                                      fill=False, edgecolor='#333333', linewidth=3)
        ax.add_patch(car_outline)

        # Windshield
        windshield = plt.Polygon([[0, 0.1], [0, width-0.1], [0.4, width-0.3], [0.4, 0.3]],
                                  fill=True, facecolor='#87CEEB', edgecolor='#333', alpha=0.3)
        ax.add_patch(windshield)

        # Seats (simplified)
        driver_seat = plt.Rectangle((2.8, 0.25), 0.8, 0.55,
                                      fill=True, facecolor='#8B4513', alpha=0.2)
        ax.add_patch(driver_seat)
        passenger_seat = plt.Rectangle((2.8, 1.05), 0.8, 0.55,
                                         fill=True, facecolor='#8B4513', alpha=0.2)
        ax.add_patch(passenger_seat)

        # Plot speakers
        for name, pos in SPEAKERS_4CH.items():
            ax.scatter(pos[0], pos[1], s=200, c='#2ecc71', marker='o',
                       edgecolors='white', linewidths=2, zorder=10, alpha=0.7)

        ax.set_xlim(-0.3, length + 0.3)
        ax.set_ylim(-0.3, width + 0.3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')

    # Left plot: Reference Mic Performance
    ax = axes[0]
    ax.set_title('Reference Microphone Performance\n(Color = Avg Noise Reduction)', fontsize=12, fontweight='bold')

    df_valid = df[df['success'] == True]
    ref_avg = df_valid.groupby('ref_mic')['noise_reduction_db'].mean()

    # Normalize colors
    vmin, vmax = ref_avg.min(), ref_avg.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdYlGn

    for name, pos in REF_MIC_POSITIONS.items():
        if name in ref_avg:
            color = cmap(norm(ref_avg[name]))
            size = 300 + (ref_avg[name] - vmin) / (vmax - vmin) * 300
            ax.scatter(pos[0], pos[1], s=size, c=[color], marker='^',
                       edgecolors='black', linewidths=2, zorder=15)
            ax.annotate(f'{name.replace("_", " ")}\n{ref_avg[name]:.1f} dB',
                        (pos[0], pos[1]),
                        textcoords="offset points",
                        xytext=(0, 25),
                        ha='center', fontsize=7, fontweight='bold')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label('Noise Reduction (dB)')
    ax.set_xlabel('Length (m)')
    ax.set_ylabel('Width (m)')

    # Right plot: Error Mic Performance
    ax = axes[1]
    ax.set_title('Error Microphone Performance\n(Color = Avg Noise Reduction)', fontsize=12, fontweight='bold')

    err_avg = df_valid.groupby('error_mic')['noise_reduction_db'].mean()

    vmin, vmax = err_avg.min(), err_avg.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for name, pos in ERROR_MIC_POSITIONS.items():
        if name in err_avg:
            color = cmap(norm(err_avg[name]))
            size = 300 + (err_avg[name] - vmin) / (vmax - vmin) * 300
            ax.scatter(pos[0], pos[1], s=size, c=[color], marker='D',
                       edgecolors='black', linewidths=2, zorder=15)
            ax.annotate(f'{name.replace("_", " ")}\n{err_avg[name]:.1f} dB',
                        (pos[0], pos[1]),
                        textcoords="offset points",
                        xytext=(0, 25),
                        ha='center', fontsize=7, fontweight='bold')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label('Noise Reduction (dB)')
    ax.set_xlabel('Length (m)')
    ax.set_ylabel('Width (m)')

    plt.suptitle('Microphone Position Performance in Car Interior (4-Speaker ANC)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/pyroom_step8_car_results.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir}/pyroom_step8_car_results.png")
    plt.close()


def plot_heatmap_per_noise(df: pd.DataFrame, save_dir: str = 'output/plots'):
    """Create heatmaps for each noise type."""
    os.makedirs(save_dir, exist_ok=True)

    df_valid = df[df['success'] == True]

    for noise_type in df_valid['noise_type'].unique():
        df_noise = df_valid[df_valid['noise_type'] == noise_type]

        pivot = df_noise.pivot_table(
            values='noise_reduction_db',
            index='ref_mic',
            columns='error_mic',
            aggfunc='mean'
        )

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(pivot.index, fontsize=9)

        ax.set_xlabel('Error Microphone')
        ax.set_ylabel('Reference Microphone')

        noise_name = NOISE_CONFIGS[noise_type]['name']
        ax.set_title(f'Noise Reduction (dB) - {noise_name}\n4-Speaker ANC System')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Noise Reduction (dB)')

        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    text_color = 'white' if val < np.nanmean(pivot.values) else 'black'
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                           color=text_color, fontsize=8)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/pyroom_step8_heatmap_{noise_type}.png', dpi=150)
        print(f"Saved: {save_dir}/pyroom_step8_heatmap_{noise_type}.png")
        plt.close()


def plot_mic_rankings(df: pd.DataFrame, save_dir: str = 'output/plots'):
    """Bar charts for microphone rankings."""
    os.makedirs(save_dir, exist_ok=True)

    df_valid = df[df['success'] == True]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Reference mic ranking
    ref_avg = df_valid.groupby('ref_mic')['noise_reduction_db'].mean().sort_values(ascending=True)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(ref_avg)))
    axes[0].barh(range(len(ref_avg)), ref_avg.values, color=colors)
    axes[0].set_yticks(range(len(ref_avg)))
    axes[0].set_yticklabels(ref_avg.index)
    axes[0].set_xlabel('Average Noise Reduction (dB)')
    axes[0].set_title('Reference Microphone Ranking')
    axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    for i, v in enumerate(ref_avg.values):
        axes[0].text(v + 0.2, i, f'{v:.1f}', va='center', fontsize=9)

    # Error mic ranking
    error_avg = df_valid.groupby('error_mic')['noise_reduction_db'].mean().sort_values(ascending=True)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(error_avg)))
    axes[1].barh(range(len(error_avg)), error_avg.values, color=colors)
    axes[1].set_yticks(range(len(error_avg)))
    axes[1].set_yticklabels(error_avg.index)
    axes[1].set_xlabel('Average Noise Reduction (dB)')
    axes[1].set_title('Error Microphone Ranking')
    axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    for i, v in enumerate(error_avg.values):
        axes[1].text(v + 0.2, i, f'{v:.1f}', va='center', fontsize=9)

    plt.suptitle('Microphone Position Rankings (4-Speaker ANC)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/pyroom_step8_mic_rankings.png', dpi=150)
    print(f"Saved: {save_dir}/pyroom_step8_mic_rankings.png")
    plt.close()


def plot_noise_comparison(df: pd.DataFrame, save_dir: str = 'output/plots'):
    """Compare performance across noise types."""
    os.makedirs(save_dir, exist_ok=True)

    df_valid = df[df['success'] == True]

    fig, ax = plt.subplots(figsize=(10, 6))

    noise_stats = df_valid.groupby('noise_type')['noise_reduction_db'].agg(['mean', 'std', 'max'])
    noise_stats = noise_stats.sort_values('mean', ascending=True)

    x = range(len(noise_stats))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(noise_stats)))

    bars = ax.barh(x, noise_stats['mean'], xerr=noise_stats['std'],
                   color=colors, capsize=5, alpha=0.8)

    # Add max markers
    ax.scatter(noise_stats['max'], x, color='red', s=100, marker='*',
              zorder=5, label='Best config')

    ax.set_yticks(x)
    ax.set_yticklabels([NOISE_CONFIGS[n]['name'] for n in noise_stats.index])
    ax.set_xlabel('Noise Reduction (dB)')
    ax.set_title('ANC Performance by Noise Type\n(Mean ± Std, * = Best Configuration)')
    ax.legend()
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/pyroom_step8_noise_comparison.png', dpi=150)
    print(f"Saved: {save_dir}/pyroom_step8_noise_comparison.png")
    plt.close()


def plot_top_configurations(df: pd.DataFrame, n_top: int = 10, save_dir: str = 'output/plots'):
    """Top N best configurations."""
    os.makedirs(save_dir, exist_ok=True)

    df_valid = df[df['success'] == True].copy()
    df_top = df_valid.nlargest(n_top, 'noise_reduction_db')

    fig, ax = plt.subplots(figsize=(12, 6))

    labels = [f"Ref: {row['ref_mic']}\nErr: {row['error_mic']}\n({row['noise_type']})"
              for _, row in df_top.iterrows()]

    colors = plt.cm.viridis(np.linspace(0.8, 0.2, n_top))
    bars = ax.barh(range(n_top), df_top['noise_reduction_db'], color=colors)

    ax.set_yticks(range(n_top))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Noise Reduction (dB)')
    ax.set_title(f'Top {n_top} Microphone Placements (4-Speaker ANC)')
    ax.invert_yaxis()

    for i, (bar, val) in enumerate(zip(bars, df_top['noise_reduction_db'])):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f} dB', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/pyroom_step8_top_configurations.png', dpi=150)
    print(f"Saved: {save_dir}/pyroom_step8_top_configurations.png")
    plt.close()


def main():
    """Run placement optimization study."""
    print("=" * 70)
    print("Step 8: Microphone Placement Optimization (4-Speaker ANC)")
    print("=" * 70)

    os.makedirs('output/plots', exist_ok=True)
    os.makedirs('output/data', exist_ok=True)

    # Run optimization
    optimizer = PlacementOptimizer(fs=16000, duration=3.0)
    df = optimizer.run_sweep(verbose=True)

    # Analyze
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)

    analysis = optimizer.analyze_results(df)

    if 'error' not in analysis:
        print("\n" + "=" * 40)
        print("BEST OVERALL CONFIGURATION")
        print("=" * 40)
        best = analysis['best_overall']
        print(f"  Reference Mic: {best['ref_mic']}")
        print(f"  Error Mic: {best['error_mic']}")
        print(f"  Noise Type: {best['noise_type']}")
        print(f"  Noise Reduction: {best['noise_reduction_db']:.1f} dB")

        print("\n" + "=" * 40)
        print("BEST PER NOISE TYPE")
        print("=" * 40)
        for noise_type, config in analysis['best_per_noise'].items():
            noise_name = NOISE_CONFIGS[noise_type]['name']
            print(f"\n  {noise_name}:")
            print(f"    Ref: {config['ref_mic']}")
            print(f"    Err: {config['error_mic']}")
            print(f"    Reduction: {config['noise_reduction_db']:.1f} dB")

        print("\n" + "=" * 40)
        print("REFERENCE MIC RANKING (best to worst)")
        print("=" * 40)
        for i, (mic, nr) in enumerate(analysis['ref_mic_ranking'].items(), 1):
            print(f"  {i}. {mic}: {nr:.1f} dB avg")

        print("\n" + "=" * 40)
        print("ERROR MIC RANKING (best to worst)")
        print("=" * 40)
        for i, (mic, nr) in enumerate(analysis['error_mic_ranking'].items(), 1):
            print(f"  {i}. {mic}: {nr:.1f} dB avg")

    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    # Car interior layout (shows all positions being tested)
    plot_car_interior_layout()

    # Performance results on car layout
    plot_car_interior_results(df, analysis)

    # Heatmaps and other plots
    plot_heatmap_per_noise(df)
    plot_mic_rankings(df)
    plot_noise_comparison(df)
    plot_top_configurations(df)

    # Save analysis
    with open('output/data/pyroom_step8_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to: output/data/pyroom_step8_analysis.json")

    print("\n" + "=" * 70)
    print("STEP 8 COMPLETE")
    print("=" * 70)
    print("\nSpeakers used (fixed):")
    for name, pos in SPEAKERS_4CH.items():
        print(f"  - {name}: {pos}")
    print("\nOutput files:")
    print("  output/plots/pyroom_step8_car_layout.png      <- CAR INTERIOR WITH ALL POSITIONS")
    print("  output/plots/pyroom_step8_car_results.png     <- PERFORMANCE ON CAR LAYOUT")
    print("  output/data/pyroom_step8_sweep_results.csv")
    print("  output/data/pyroom_step8_analysis.json")
    print("  output/plots/pyroom_step8_heatmap_*.png")
    print("  output/plots/pyroom_step8_mic_rankings.png")
    print("  output/plots/pyroom_step8_noise_comparison.png")
    print("  output/plots/pyroom_step8_top_configurations.png")

    return df, analysis


if __name__ == '__main__':
    df, analysis = main()
