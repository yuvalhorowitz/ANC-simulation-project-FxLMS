"""
Full-Cabin Cancellation Heatmap — Four-Way Comparison

Generates a top-down 5×5 cm spatial map of the entire cabin at driver ear
height for four configurations:
  1. SISO  (1 ref, 1 speaker, 1 error mic) — scalar FxLMS
  2. Pseudo-SIMO (1 ref, 4 speakers broadcast, 1 error mic) — current 4-speaker mode
  3. Stage 1 SIMO (1 ref, 4 speakers independent, 1 error mic) — MIMOFxNLMS
  4. Stage 2 SIMO+multi-error (1 ref, 4 speakers indep, 4 error mics) — MIMOFxNLMSMultiError

Output: output/plots/cancellation_heatmap_full_cabin_4way.png
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import fftconvolve

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pyroomacoustics as pra
from src.acoustic.path_generator import FIRPath
from src.core.fxlms import FxNLMS
from src.core.mimo_fxnlms import MIMOFxNLMS
from src.core.mimo_fxnlms_multierror import MIMOFxNLMSMultiError
from src.noise.noise_mixer import NoiseMixer

# Config
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
HEAD_ZONE_OFFSETS = [(0.05, 0.05), (-0.05, 0.05), (0.05, -0.05), (-0.05, -0.05)]
ERROR_MICS_K4 = [
    [ERROR_MIC_POS[0], ERROR_MIC_POS[1] + dy, ERROR_MIC_POS[2] + dz]
    for dy, dz in HEAD_ZONE_OFFSETS
]

FILTER_LENGTH = 512
STEP_SIZE = 0.003
FS = 16000
DURATION_TRAIN = 5.0
AUDIO_FILE = 'real_noises/realcar1.wav'

CELL_SIZE_M = 0.05
EVAL_Z = ERROR_MIC_POS[2]
GRID_INSET = 0.025

OUTPUT_PATH = Path(__file__).parent.parent.parent / 'output' / 'plots' / 'cancellation_heatmap_full_cabin_4way.png'


def make_grid():
    x = np.arange(GRID_INSET, ROOM_DIMS[0] - GRID_INSET + 1e-9, CELL_SIZE_M)
    y = np.arange(GRID_INSET, ROOM_DIMS[1] - GRID_INSET + 1e-9, CELL_SIZE_M)
    pos = [[float(xi), float(yi), float(EVAL_Z)] for yi in y for xi in x]
    return pos, x, y


def build_room(speaker_positions, training_error_mics, eval_positions):
    """
    Layout:
      Source 0: noise. Sources 1..M: speakers.
      Mic 0: reference. Mics 1..K: training error mics. Mics K+1...: eval grid.
    """
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
    all_mics = [REF_MIC_POS] + list(training_error_mics) + list(eval_positions)
    room.add_microphone_array(pra.MicrophoneArray(np.array(all_mics).T, fs=FS))
    room.compute_rir()
    return room


def extract_paths(room, M, K_train, num_eval):
    L = 512
    H_ref = room.rir[0][0][:L]
    # Training: per-error-mic primary and per-(speaker, error_mic) secondary
    H_pri_train = [room.rir[1 + k][0][:L] for k in range(K_train)]
    H_sec_train = [
        [room.rir[1 + k][1 + m][:L] for k in range(K_train)] for m in range(M)
    ]
    # Eval: per-eval-cell primary + per-(speaker, eval_cell) secondary
    H_pri_eval = [room.rir[1 + K_train + e][0][:L] for e in range(num_eval)]
    H_sec_eval = [
        [room.rir[1 + K_train + e][1 + m][:L] for e in range(num_eval)]
        for m in range(M)
    ]
    return H_ref, H_pri_train, H_sec_train, H_pri_eval, H_sec_eval


# ---------- Trainers ----------

def train_siso(noise, H_pri, H_sec, H_ref, H_sec_est):
    fxlms = FxNLMS(filter_length=FILTER_LENGTH, step_size=STEP_SIZE,
                   secondary_path_estimate=H_sec_est, regularization=1e-4)
    pri, sec, ref = FIRPath(H_pri), FIRPath(H_sec), FIRPath(H_ref)
    n = len(noise); a = np.zeros(n)
    for i in range(n):
        s = noise[i]; x = ref.filter_sample(s); d = pri.filter_sample(s)
        y = fxlms.generate_antinoise(x); a[i] = y
        e = d + sec.filter_sample(y)
        fxlms.filter_reference(x); fxlms.update_weights(e)
    return a


def train_pseudo(noise, H_pri, H_sec_list, H_ref, H_sec_est_combined):
    """Pseudo-SIMO: scalar FxLMS with combined sum of secondary paths."""
    fxlms = FxNLMS(filter_length=FILTER_LENGTH, step_size=STEP_SIZE,
                   secondary_path_estimate=H_sec_est_combined, regularization=1e-4)
    pri, ref = FIRPath(H_pri), FIRPath(H_ref)
    secs = [FIRPath(s) for s in H_sec_list]
    n = len(noise); a = np.zeros(n)
    for i in range(n):
        s = noise[i]; x = ref.filter_sample(s); d = pri.filter_sample(s)
        y = fxlms.generate_antinoise(x); a[i] = y
        e = d + sum(sc.filter_sample(y) for sc in secs)
        fxlms.filter_reference(x); fxlms.update_weights(e)
    return a


def train_stage1(noise, H_pri, H_sec_list, H_ref, H_sec_est_list):
    M = len(H_sec_list)
    fxlms = MIMOFxNLMS(filter_length=FILTER_LENGTH, step_size=STEP_SIZE,
                       secondary_path_estimates=H_sec_est_list, regularization=1e-4)
    pri, ref = FIRPath(H_pri), FIRPath(H_ref)
    secs = [FIRPath(s) for s in H_sec_list]
    n = len(noise); a = np.zeros((n, M))
    for i in range(n):
        s = noise[i]; x = ref.filter_sample(s); d = pri.filter_sample(s)
        y_per = fxlms.generate_antinoise(x); a[i] = y_per
        e = d + sum(secs[m].filter_sample(y_per[m]) for m in range(M))
        fxlms.filter_reference(x); fxlms.update_weights(e)
    return a


def train_stage2(noise, H_pri_list, H_sec_lists, H_ref, H_sec_est_lists):
    """
    Stage 2: K error mics, M speakers.
    H_pri_list: list of K primary paths, one per training error mic
    H_sec_lists: list[m][k] of speaker→err_mic paths
    H_sec_est_lists: same shape, with 5% noise
    """
    M = len(H_sec_lists); K = len(H_sec_lists[0])
    fxlms = MIMOFxNLMSMultiError(
        filter_length=FILTER_LENGTH, step_size=STEP_SIZE,
        secondary_path_estimates=H_sec_est_lists, regularization=1e-4
    )
    pris = [FIRPath(p) for p in H_pri_list]
    ref = FIRPath(H_ref)
    secs = [[FIRPath(p) for p in row] for row in H_sec_lists]
    n = len(noise); a = np.zeros((n, M))
    for i in range(n):
        s = noise[i]; x = ref.filter_sample(s)
        d = np.array([pris[k].filter_sample(s) for k in range(K)])
        y_per = fxlms.generate_antinoise(x); a[i] = y_per
        antin = np.zeros(K)
        for m in range(M):
            for k in range(K):
                antin[k] += secs[m][k].filter_sample(y_per[m])
        errors = d + antin
        fxlms.filter_reference(x); fxlms.update_weights(errors)
    return a


# ---------- Evaluator ----------

def evaluate_grid(noise, antinoise, H_pri_eval_list, H_sec_eval_lists):
    if antinoise.ndim == 1:
        antinoise_per_m = [antinoise]
    else:
        antinoise_per_m = [antinoise[:, m] for m in range(antinoise.shape[1])]
    M = len(antinoise_per_m); n = len(noise); half = n // 2
    num_eval = len(H_pri_eval_list)
    attens = np.zeros(num_eval)
    for e in range(num_eval):
        d = fftconvolve(noise, H_pri_eval_list[e], mode='same')
        a = np.zeros(n)
        for m in range(M):
            sec = H_sec_eval_lists[m][e]
            a += fftconvolve(antinoise_per_m[m], sec, mode='same')
        e_sig = d + a
        d_p = np.mean(d[half:] ** 2); e_p = np.mean(e_sig[half:] ** 2)
        attens[e] = 10 * np.log10(d_p / e_p) if (d_p > 1e-12 and e_p > 1e-12) else 0.0
    return attens


# ---------- Main ----------

def main():
    print("=" * 72)
    print(" Full-Cabin 4-Way Cancellation Heatmap")
    print("=" * 72)

    print("\nLoading noise...")
    noise = NoiseMixer(FS).load_audio_file(AUDIO_FILE, duration=DURATION_TRAIN)
    print(f"  {len(noise)} samples")

    eval_pos, x_centers, y_centers = make_grid()
    nx, ny = len(x_centers), len(y_centers)
    num_eval = len(eval_pos)
    print(f"\n5×5 cm grid: {nx} × {ny} = {num_eval} cells")

    # =======================
    # SISO room: 1 speaker, 1 training error mic
    # =======================
    print("\n[1/4] Building SISO room...")
    np.random.seed(42)
    siso_room = build_room([SISO_SPEAKER], [ERROR_MIC_POS], eval_pos)
    print("  RIRs computed")
    H_ref_s, H_pri_train_s, H_sec_train_s, H_pri_eval_s, H_sec_eval_s = extract_paths(
        siso_room, M=1, K_train=1, num_eval=num_eval
    )
    H_sec_est_s = H_sec_train_s[0][0] * (1 + 0.05 * np.random.randn(len(H_sec_train_s[0][0])))

    print("  Training SISO...")
    siso_anti = train_siso(noise, H_pri_train_s[0], H_sec_train_s[0][0], H_ref_s, H_sec_est_s)

    # =======================
    # Pseudo-SIMO and Stage 1 SIMO room: 4 speakers, 1 training error mic
    # =======================
    print("\n[2/4 + 3/4] Building 4-speaker room (1 error mic)...")
    np.random.seed(42)
    spk_list = list(FOUR_SPEAKERS.values())
    multi_room = build_room(spk_list, [ERROR_MIC_POS], eval_pos)
    print("  RIRs computed")
    H_ref_m, H_pri_train_m, H_sec_train_m, H_pri_eval_m, H_sec_eval_m = extract_paths(
        multi_room, M=4, K_train=1, num_eval=num_eval
    )

    # Pseudo: combined estimate
    H_sec_combined = np.zeros(512)
    for m in range(4):
        s = H_sec_train_m[m][0]
        H_sec_combined[:len(s)] += s
    H_sec_est_pseudo = H_sec_combined * (1 + 0.05 * np.random.randn(len(H_sec_combined)))

    # Stage 1: per-speaker estimate (secondary at the single training error mic)
    H_sec_train_stage1 = [H_sec_train_m[m][0] for m in range(4)]
    H_sec_est_stage1 = [s * (1 + 0.05 * np.random.randn(len(s))) for s in H_sec_train_stage1]

    print("  Training Pseudo-SIMO...")
    pseudo_anti = train_pseudo(noise, H_pri_train_m[0], H_sec_train_stage1, H_ref_m, H_sec_est_pseudo)

    print("  Training Stage 1 SIMO...")
    stage1_anti = train_stage1(noise, H_pri_train_m[0], H_sec_train_stage1, H_ref_m, H_sec_est_stage1)

    # =======================
    # Stage 2 SIMO+multi-error: 4 speakers, 4 training error mics
    # =======================
    print("\n[4/4] Building Stage 2 room (4 speakers, 4 training error mics)...")
    np.random.seed(42)
    multi_k4_room = build_room(spk_list, ERROR_MICS_K4, eval_pos)
    print("  RIRs computed")
    H_ref_k4, H_pri_train_k4, H_sec_train_k4, H_pri_eval_k4, H_sec_eval_k4 = extract_paths(
        multi_k4_room, M=4, K_train=4, num_eval=num_eval
    )
    H_sec_est_stage2 = [
        [s * (1 + 0.05 * np.random.randn(len(s))) for s in row] for row in H_sec_train_k4
    ]

    print("  Training Stage 2 SIMO+multi-error...")
    stage2_anti = train_stage2(noise, H_pri_train_k4, H_sec_train_k4, H_ref_k4, H_sec_est_stage2)

    # =======================
    # Evaluate spatial grids
    # =======================
    print("\nEvaluating SISO grid...")
    siso_at = evaluate_grid(noise, siso_anti, H_pri_eval_s, H_sec_eval_s)
    print("Evaluating Pseudo-SIMO grid...")
    H_sec_eval_pseudo = [[
        sum(H_sec_eval_m[m][e] for m in range(4)) for e in range(num_eval)
    ]]
    pseudo_at = evaluate_grid(noise, pseudo_anti, H_pri_eval_m, H_sec_eval_pseudo)
    print("Evaluating Stage 1 grid...")
    stage1_at = evaluate_grid(noise, stage1_anti, H_pri_eval_m, H_sec_eval_m)
    print("Evaluating Stage 2 grid...")
    stage2_at = evaluate_grid(noise, stage2_anti, H_pri_eval_k4, H_sec_eval_k4)

    siso_g = siso_at.reshape(ny, nx)
    pseudo_g = pseudo_at.reshape(ny, nx)
    stage1_g = stage1_at.reshape(ny, nx)
    stage2_g = stage2_at.reshape(ny, nx)

    grids = [siso_g, pseudo_g, stage1_g, stage2_g]
    titles = [
        'SISO\n(1 spk, 1 err mic, scalar FxLMS)',
        'Pseudo-SIMO\n(4 spk broadcast, 1 err mic)',
        'Stage 1 SIMO\n(4 spk indep, 1 err mic)',
        'Stage 2 SIMO+multi-error\n(4 spk indep, 4 err mics)',
    ]
    speakers_per = [[SISO_SPEAKER]] + [list(FOUR_SPEAKERS.values())] * 3
    err_mics_per = [[ERROR_MIC_POS], [ERROR_MIC_POS], [ERROR_MIC_POS], ERROR_MICS_K4]

    # Symmetric color scale
    vmax = max(np.max(np.abs(g)) for g in grids)
    vmax = max(vmax, 5)

    print("\nGenerating 4-panel heatmap...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    extent = [0, ROOM_DIMS[0], 0, ROOM_DIMS[1]]

    for ax, grid, title, spks, errs in zip(axes, grids, titles, speakers_per, err_mics_per):
        im = ax.imshow(grid, origin='lower', extent=extent,
                       cmap='RdBu', vmin=-vmax, vmax=vmax,
                       interpolation='nearest', aspect='equal')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('x (m) — front → rear')
        ax.set_ylabel('y (m) — left ↔ right')

        ax.add_patch(Rectangle((0, 0), ROOM_DIMS[0], ROOM_DIMS[1],
                               fill=False, edgecolor='black', linewidth=1.2))
        ax.add_patch(Rectangle((2.0, 0.2), 0.6, 0.7, fill=False,
                               edgecolor='gray', linestyle='--', linewidth=0.7, alpha=0.6))
        ax.add_patch(Rectangle((2.0, 1.0), 0.6, 0.7, fill=False,
                               edgecolor='gray', linestyle='--', linewidth=0.7, alpha=0.6))

        ax.plot(NOISE_POS[0], NOISE_POS[1], 'r*', markersize=14, mec='black', mew=0.5)
        ax.plot(REF_MIC_POS[0], REF_MIC_POS[1], 'g^', markersize=10, mec='black', mew=0.5)

        for em in errs:
            ax.plot(em[0], em[1], 'kX', markersize=12, mec='white', mew=0.5)
        for spk in spks:
            ax.plot(spk[0], spk[1], 'bs', markersize=8, mec='black', mew=0.5)

        max_db = np.max(grid); min_db = np.min(grid); mean_db = np.mean(grid)
        pct_pos = 100 * np.sum(grid > 0) / grid.size
        pct_neg = 100 * np.sum(grid < 0) / grid.size
        ax.text(0.02, 0.98,
                f"max: {max_db:+.1f} dB\n"
                f"min: {min_db:+.1f} dB\n"
                f"mean: {mean_db:+.1f} dB\n"
                f"cancel: {pct_pos:.0f}%\n"
                f"amplify: {pct_neg:.0f}%",
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label('Noise reduction (dB)\nblue = cancellation, red = amplification',
                   fontsize=10)

    fig.suptitle(f'Full-Cabin Quiet Zone Map — Algorithm Comparison '
                 f'(5×5 cm cells, {nx*ny} cells, {Path(AUDIO_FILE).name})',
                 fontsize=12, fontweight='bold', y=0.99)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=140, bbox_inches='tight', facecolor='white')
    print(f"Saved to: {OUTPUT_PATH}")
    plt.close()

    print("\n" + "=" * 72)
    print(" Full-cabin spatial summary")
    print("=" * 72)
    print(f"{'Algorithm':<35} | {'Mean':>7} | {'Max':>7} | {'Min':>7} | "
          f"{'%cancel':>7} | {'%amplify':>8}")
    print("-" * 90)
    for label, grid in zip(titles, grids):
        short = label.split('\n')[0]
        print(f"{short:<35} | {np.mean(grid):>+6.2f} | {np.max(grid):>+6.2f} | "
              f"{np.min(grid):>+6.2f} | {100*np.mean(grid>0):>6.1f}% | "
              f"{100*np.mean(grid<0):>7.1f}%")


if __name__ == '__main__':
    main()
