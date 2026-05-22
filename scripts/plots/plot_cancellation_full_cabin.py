"""
Full-Cabin Cancellation Heatmap

Top-down view of the entire car interior at driver ear height. Divides the
cabin into 5×5 cm cells and computes ANC noise reduction (in dB) at every
cell, after training each algorithm with the canonical error mic position.

Compares:
  - SISO (1 speaker, scalar FxLMS)
  - True MIMO (4 speakers, independent filters)

Diverging colormap: blue = cancellation, red = amplification, white = neutral.
This exposes the spatial structure of each algorithm's "zone of quiet" across
the full cabin and reveals where each method amplifies noise.

Output: output/plots/cancellation_heatmap_full_cabin.png
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from scipy.signal import fftconvolve

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pyroomacoustics as pra
from src.acoustic.path_generator import FIRPath
from src.core.fxlms import FxNLMS
from src.core.mimo_fxnlms import MIMOFxNLMS
from src.noise.noise_mixer import NoiseMixer


# ============================================================
# Configuration
# ============================================================

ROOM_DIMS = [4.5, 1.85, 1.2]
ABSORPTION = 0.35
MAX_ORDER = 3
NOISE_POS = [0.5, 0.92, 0.4]
REF_MIC_POS = [1.1, 0.92, 0.8]
ERROR_MIC_POS = [2.5, 0.55, 1.05]

SISO_SPEAKER = [0.8, 0.25, 0.9]
FOUR_SPEAKERS = {
    'door_L':  [2.0, 0.10, 0.4],
    'door_R':  [2.0, 1.75, 0.4],
    'dash_L':  [0.8, 0.25, 0.9],
    'dash_R':  [0.8, 1.60, 0.9],
}

FILTER_LENGTH = 512
STEP_SIZE = 0.003
FS = 16000
DURATION_TRAIN = 5.0
AUDIO_FILE = 'real_noises/realcar1.wav'

# 5×5 cm grid covering the cabin at driver ear height
CELL_SIZE_M = 0.05
EVAL_Z = ERROR_MIC_POS[2]   # driver ear height (1.05 m)
GRID_INSET = 0.025          # 2.5 cm from walls so mics aren't on boundaries

OUTPUT_PATH = Path(__file__).parent.parent.parent / 'output' / 'plots' / 'cancellation_heatmap_full_cabin.png'


def make_grid():
    """5×5 cm grid covering the full cabin floor area at EVAL_Z height."""
    x_centers = np.arange(GRID_INSET, ROOM_DIMS[0] - GRID_INSET + 1e-9, CELL_SIZE_M)
    y_centers = np.arange(GRID_INSET, ROOM_DIMS[1] - GRID_INSET + 1e-9, CELL_SIZE_M)
    positions = []
    for y in y_centers:
        for x in x_centers:
            positions.append([float(x), float(y), float(EVAL_Z)])
    return positions, x_centers, y_centers


def build_room(speaker_positions, eval_positions):
    """Room with noise + speakers + ref + error + grid eval mics."""
    materials = {
        'ceiling': pra.Material(ABSORPTION * 1.1),
        'floor':   pra.Material(ABSORPTION * 1.5),
        'east':    pra.Material(ABSORPTION * 0.5),
        'west':    pra.Material(ABSORPTION * 0.5),
        'north':   pra.Material(ABSORPTION * 0.7),
        'south':   pra.Material(ABSORPTION * 0.9),
    }
    room = pra.ShoeBox(ROOM_DIMS, fs=FS, materials=materials,
                       max_order=MAX_ORDER, air_absorption=True)
    room.add_source(NOISE_POS)
    for s in speaker_positions:
        room.add_source(s)
    all_mics = [REF_MIC_POS, ERROR_MIC_POS] + eval_positions
    room.add_microphone_array(pra.MicrophoneArray(np.array(all_mics).T, fs=FS))
    room.compute_rir()
    return room


def extract_paths(room, num_speakers, num_eval):
    max_len = 512
    H_reference = room.rir[0][0][:max_len]
    H_primary = room.rir[1][0][:max_len]
    H_secondary = [room.rir[1][m + 1][:max_len] for m in range(num_speakers)]
    H_primary_eval = [room.rir[2 + k][0][:max_len] for k in range(num_eval)]
    H_secondary_eval = [
        [room.rir[2 + k][m + 1][:max_len] for k in range(num_eval)]
        for m in range(num_speakers)
    ]
    return H_primary, H_secondary, H_reference, H_primary_eval, H_secondary_eval


def train_siso(noise, H_primary, H_sec, H_ref, H_sec_est):
    fxlms = FxNLMS(filter_length=FILTER_LENGTH, step_size=STEP_SIZE,
                   secondary_path_estimate=H_sec_est, regularization=1e-4)
    primary, secondary, reference = FIRPath(H_primary), FIRPath(H_sec), FIRPath(H_ref)
    n = len(noise); antinoise = np.zeros(n)
    for i in range(n):
        s = noise[i]; x = reference.filter_sample(s); d = primary.filter_sample(s)
        y = fxlms.generate_antinoise(x); antinoise[i] = y
        e = d + secondary.filter_sample(y)
        fxlms.filter_reference(x); fxlms.update_weights(e)
    return antinoise


def train_true_mimo(noise, H_primary, H_sec_list, H_ref, H_sec_est_list):
    M = len(H_sec_list)
    fxlms = MIMOFxNLMS(filter_length=FILTER_LENGTH, step_size=STEP_SIZE,
                       secondary_path_estimates=H_sec_est_list, regularization=1e-4)
    primary, reference = FIRPath(H_primary), FIRPath(H_ref)
    secondaries = [FIRPath(s) for s in H_sec_list]
    n = len(noise); antinoise_per = np.zeros((n, M))
    for i in range(n):
        s = noise[i]; x = reference.filter_sample(s); d = primary.filter_sample(s)
        y_per = fxlms.generate_antinoise(x); antinoise_per[i] = y_per
        e = d + sum(secondaries[m].filter_sample(y_per[m]) for m in range(M))
        fxlms.filter_reference(x); fxlms.update_weights(e)
    return antinoise_per


def evaluate_grid(noise, antinoise, H_primary_eval_list, H_sec_eval_lists):
    """
    For each eval mic k: e_k = d_k + sum_m (anti_noise_m * sec_eval[m][k])
    Returns array of attenuation_k in dB.

    antinoise: 1D (n,) for SISO or 2D (n, M) for MIMO
    H_sec_eval_lists: list[m][k] of secondary path arrays
    """
    if antinoise.ndim == 1:
        antinoise_per_m = [antinoise]
    else:
        antinoise_per_m = [antinoise[:, m] for m in range(antinoise.shape[1])]
    M = len(antinoise_per_m)
    n = len(noise)
    half = n // 2
    num_eval = len(H_primary_eval_list)
    attens = np.zeros(num_eval)

    for k in range(num_eval):
        d_k = fftconvolve(noise, H_primary_eval_list[k], mode='same')
        a_k = np.zeros(n)
        for m in range(M):
            sec = H_sec_eval_lists[m][k]
            a_k += fftconvolve(antinoise_per_m[m], sec, mode='same')
        e_k = d_k + a_k
        d_p = np.mean(d_k[half:] ** 2); e_p = np.mean(e_k[half:] ** 2)
        attens[k] = 10 * np.log10(d_p / e_p) if (e_p > 1e-12 and d_p > 1e-12) else 0.0
    return attens


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print(" Full-Cabin Cancellation Heatmap (5×5 cm grid)")
    print("=" * 70)

    print(f"\nLoading noise ({AUDIO_FILE})...")
    noise = NoiseMixer(FS).load_audio_file(AUDIO_FILE, duration=DURATION_TRAIN)
    print(f"  {len(noise)} samples ({len(noise)/FS:.1f} s)")

    eval_pos, x_centers, y_centers = make_grid()
    nx, ny = len(x_centers), len(y_centers)
    num_eval = len(eval_pos)
    print(f"\n5×5 cm grid: {nx} × {ny} = {num_eval} cells (covers {nx*5} × {ny*5} cm)")
    print(f"Slice height: z = {EVAL_Z:.2f} m (driver ear)")

    # ----- SISO room -----
    print("\nBuilding SISO room (1 speaker)...")
    np.random.seed(42)
    siso_room = build_room([SISO_SPEAKER], eval_pos)
    print("  Computing RIRs...", end='', flush=True)
    H_pri_s, H_sec_s, H_ref_s, H_pri_eval_s, H_sec_eval_s = extract_paths(
        siso_room, num_speakers=1, num_eval=num_eval)
    print(" done")
    H_sec_est_s = H_sec_s[0] * (1 + 0.05 * np.random.randn(len(H_sec_s[0])))

    # ----- 4-speaker room (true MIMO) -----
    print("\nBuilding 4-speaker room...")
    np.random.seed(42)
    multi_room = build_room(list(FOUR_SPEAKERS.values()), eval_pos)
    print("  Computing RIRs...", end='', flush=True)
    H_pri_m, H_sec_m, H_ref_m, H_pri_eval_m, H_sec_eval_m = extract_paths(
        multi_room, num_speakers=4, num_eval=num_eval)
    print(" done")
    H_sec_est_m = [s * (1 + 0.05 * np.random.randn(len(s))) for s in H_sec_m]

    # ----- Train -----
    print("\nTraining SISO scalar FxLMS...")
    siso_antinoise = train_siso(noise, H_pri_s, H_sec_s[0], H_ref_s, H_sec_est_s)

    print("Training True MIMO (4 speakers, independent filters)...")
    mimo_antinoise = train_true_mimo(noise, H_pri_m, H_sec_m, H_ref_m, H_sec_est_m)

    # ----- Evaluate spatial grids -----
    print(f"\nEvaluating SISO at {num_eval} grid cells...")
    siso_atten = evaluate_grid(noise, siso_antinoise, H_pri_eval_s, H_sec_eval_s)

    print(f"Evaluating True MIMO at {num_eval} grid cells...")
    mimo_atten = evaluate_grid(noise, mimo_antinoise, H_pri_eval_m, H_sec_eval_m)

    siso_grid = siso_atten.reshape(ny, nx)
    mimo_grid = mimo_atten.reshape(ny, nx)

    # ----- Plot -----
    print("\nGenerating heatmap...")
    vmax = max(np.max(np.abs(siso_grid)), np.max(np.abs(mimo_grid)))
    vmax = max(vmax, 5)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    extent = [0, ROOM_DIMS[0], 0, ROOM_DIMS[1]]

    titles = ['SISO\n(1 speaker, scalar FxLMS)',
              'True MIMO\n(4 speakers, independent filters)']
    grids = [siso_grid, mimo_grid]
    speaker_sets = [[SISO_SPEAKER], list(FOUR_SPEAKERS.values())]

    for ax, grid, title, spks in zip(axes, grids, titles, speaker_sets):
        im = ax.imshow(grid, origin='lower', extent=extent,
                       cmap='RdBu', vmin=-vmax, vmax=vmax,
                       interpolation='nearest', aspect='equal')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('x (m) — front → rear')
        ax.set_ylabel('y (m) — left ↔ right')

        # Car outline
        ax.add_patch(Rectangle((0, 0), ROOM_DIMS[0], ROOM_DIMS[1],
                               fill=False, edgecolor='black', linewidth=1.5))
        # Driver and passenger seats (visual reference)
        ax.add_patch(Rectangle((2.0, 0.2), 0.6, 0.7, fill=False,
                               edgecolor='gray', linestyle='--', linewidth=0.7, alpha=0.7))
        ax.add_patch(Rectangle((2.0, 1.0), 0.6, 0.7, fill=False,
                               edgecolor='gray', linestyle='--', linewidth=0.7, alpha=0.7))

        # Noise source
        ax.plot(NOISE_POS[0], NOISE_POS[1], 'r*', markersize=14, mec='black', mew=0.5)
        ax.annotate('noise', (NOISE_POS[0], NOISE_POS[1]),
                    textcoords="offset points", xytext=(8, 8), fontsize=8, color='darkred')

        # Reference mic
        ax.plot(REF_MIC_POS[0], REF_MIC_POS[1], 'g^', markersize=10, mec='black', mew=0.5)
        ax.annotate('ref', (REF_MIC_POS[0], REF_MIC_POS[1]),
                    textcoords="offset points", xytext=(8, -12), fontsize=8, color='darkgreen')

        # Error mic (training target)
        ax.plot(ERROR_MIC_POS[0], ERROR_MIC_POS[1], 'kX', markersize=14, mec='white', mew=0.5)
        ax.annotate('error mic\n(training target)', (ERROR_MIC_POS[0], ERROR_MIC_POS[1]),
                    textcoords="offset points", xytext=(10, 5), fontsize=8, color='black')

        # Speakers
        for spk in spks:
            ax.plot(spk[0], spk[1], 'bs', markersize=8, mec='black', mew=0.5)

        # Statistics box
        max_db = np.max(grid); min_db = np.min(grid)
        pct_pos = 100 * np.sum(grid > 0) / grid.size
        pct_neg = 100 * np.sum(grid < 0) / grid.size
        mean_db = np.mean(grid)
        ax.text(0.02, 0.98,
                f"max: {max_db:+.1f} dB\nmin: {min_db:+.1f} dB\n"
                f"mean: {mean_db:+.1f} dB\n"
                f"cancel area: {pct_pos:.0f}%\n"
                f"amplify area: {pct_neg:.0f}%",
                transform=ax.transAxes, fontsize=8.5, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label('Noise reduction (dB)\nblue = cancellation, red = amplification',
                   fontsize=10)

    fig.suptitle(f'Full-Cabin Quiet Zone Map (5×5 cm cells, {nx*ny} cells, '
                 f'trained on {Path(AUDIO_FILE).name})',
                 fontsize=12, fontweight='bold', y=1.02)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved to: {OUTPUT_PATH}")
    plt.close()

    print("\n" + "=" * 70)
    print(" Full-cabin spatial summary")
    print("=" * 70)
    print(f"{'Algorithm':<15} | {'Mean':>7} | {'Max':>7} | {'Min':>7} | "
          f"{'%cancel':>8} | {'%amplify':>9}")
    print("-" * 70)
    for label, grid in zip(['SISO', 'True MIMO'], grids):
        mean_db = np.mean(grid); max_db = np.max(grid); min_db = np.min(grid)
        pct_pos = 100 * np.sum(grid > 0) / grid.size
        pct_neg = 100 * np.sum(grid < 0) / grid.size
        print(f"{label:<15} | {mean_db:>+6.2f} | {max_db:>+6.2f} | "
              f"{min_db:>+6.2f} | {pct_pos:>7.1f}% | {pct_neg:>8.1f}%")


if __name__ == '__main__':
    main()
