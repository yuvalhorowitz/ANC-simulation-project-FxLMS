"""
1×5 Cancellation Heatmap — full cabin only

Top-down 5×5 cm grid spanning the entire cabin at driver-ear height.
5 panels, one per algorithm. Same color scale across all panels for fair
comparison.

Output: output/plots/cancellation_heatmap_1x5_cabin.png

(For the 2-row version with head-zone closeup, see plot_cancellation_2x5.py.)
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
from src.core.mimo_fxnlms_full import MIMOFxNLMSFull
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
FOUR_REF_MICS = {
    'firewall':  [0.3, 0.92, 0.5],
    'floor':     [2.0, 0.55, 0.15],
    'a_pillar':  [0.5, 0.15, 1.0],
    'dashboard': [0.9, 0.92, 0.8],
}
HEAD_ZONE_OFFSETS = [(0.05, 0.05), (-0.05, 0.05), (0.05, -0.05), (-0.05, -0.05)]
ERROR_MICS_K4 = [
    [ERROR_MIC_POS[0], ERROR_MIC_POS[1] + dy, ERROR_MIC_POS[2] + dz]
    for dy, dz in HEAD_ZONE_OFFSETS
]

FILTER_LENGTH = 256
FILTER_LENGTH_STAGE3 = 256
STEP_SIZE = 0.003
STEP_SIZE_STAGE3 = 0.001
FS = 16000
DURATION_TRAIN = 5.0
AUDIO_FILE = 'real_noises/realcar1.wav'

CELL_SIZE_M = 0.05
EVAL_Z = ERROR_MIC_POS[2]
GRID_INSET = 0.025

OUTPUT_PATH = Path(__file__).parent.parent.parent / 'output' / 'plots' / 'cancellation_heatmap_1x5_cabin.png'


def make_grid():
    x = np.arange(GRID_INSET, ROOM_DIMS[0] - GRID_INSET + 1e-9, CELL_SIZE_M)
    y = np.arange(GRID_INSET, ROOM_DIMS[1] - GRID_INSET + 1e-9, CELL_SIZE_M)
    pos = [[float(xi), float(yi), float(EVAL_Z)] for yi in y for xi in x]
    return pos, x, y


def build_room(speaker_positions, training_error_mics, eval_positions,
               num_ref_mics_extra=0):
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
    mics = [REF_MIC_POS]
    if num_ref_mics_extra > 0:
        mics.extend(list(FOUR_REF_MICS.values())[:num_ref_mics_extra])
    mics.extend(list(training_error_mics))
    mics.extend(list(eval_positions))
    room.add_microphone_array(pra.MicrophoneArray(np.array(mics).T, fs=FS))
    room.compute_rir()
    return room


def extract_paths(room, M, K_train, num_eval, num_extra_refs=0):
    L = 512
    base_ref = 1 + num_extra_refs
    H_ref = room.rir[0][0][:L]
    H_extra = [room.rir[1 + n][0][:L] for n in range(num_extra_refs)]
    H_pri_train = [room.rir[base_ref + k][0][:L] for k in range(K_train)]
    H_sec_train = [
        [room.rir[base_ref + k][1 + m][:L] for k in range(K_train)] for m in range(M)
    ]
    H_pri_eval = [room.rir[base_ref + K_train + e][0][:L] for e in range(num_eval)]
    H_sec_eval = [
        [room.rir[base_ref + K_train + e][1 + m][:L] for e in range(num_eval)]
        for m in range(M)
    ]
    return H_ref, H_extra, H_pri_train, H_sec_train, H_pri_eval, H_sec_eval


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


def train_stage3(noise, H_pri_list, H_sec_lists, H_refs, H_sec_est_lists):
    M = len(H_sec_lists); K = len(H_sec_lists[0]); N = len(H_refs)
    fxlms = MIMOFxNLMSFull(
        filter_length=FILTER_LENGTH_STAGE3, step_size=STEP_SIZE_STAGE3,
        num_reference_mics=N, secondary_path_estimates=H_sec_est_lists,
        regularization=1e-4
    )
    pris = [FIRPath(p) for p in H_pri_list]
    refs = [FIRPath(r) for r in H_refs]
    secs = [[FIRPath(p) for p in row] for row in H_sec_lists]
    n = len(noise); a = np.zeros((n, M))
    for i in range(n):
        s = noise[i]
        x_vec = np.array([refs[nn].filter_sample(s) for nn in range(N)])
        d = np.array([pris[k].filter_sample(s) for k in range(K)])
        y_per = fxlms.generate_antinoise(x_vec); a[i] = y_per
        antin = np.zeros(K)
        for m in range(M):
            for k in range(K):
                antin[k] += secs[m][k].filter_sample(y_per[m])
        errors = d + antin
        fxlms.filter_reference(x_vec); fxlms.update_weights(errors)
    return a


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


def main():
    print("=" * 70)
    print(" 1×5 Full-Cabin Cancellation Heatmap")
    print("=" * 70)

    print("\nLoading noise...")
    noise = NoiseMixer(FS).load_audio_file(AUDIO_FILE, duration=DURATION_TRAIN)
    eval_pos, x_centers, y_centers = make_grid()
    nx, ny = len(x_centers), len(y_centers)
    n_eval = len(eval_pos)
    print(f"  Grid: {nx}×{ny} = {n_eval} cells\n")

    # ----- SISO room -----
    print("[1/5] SISO...")
    np.random.seed(42)
    room = build_room([SISO_SPEAKER], [ERROR_MIC_POS], eval_pos)
    H_ref, _, H_pri_train, H_sec_train, H_pri_eval, H_sec_eval = extract_paths(
        room, M=1, K_train=1, num_eval=n_eval, num_extra_refs=0
    )
    H_sec_est = H_sec_train[0][0] * (1 + 0.05 * np.random.randn(len(H_sec_train[0][0])))
    siso_anti = train_siso(noise, H_pri_train[0], H_sec_train[0][0], H_ref, H_sec_est)
    siso_grid = evaluate_grid(noise, siso_anti, H_pri_eval, H_sec_eval).reshape(ny, nx)

    # ----- 4-speaker (Pseudo + Stage 1) -----
    print("[2-3/5] Pseudo-SIMO + Stage 1...")
    np.random.seed(42)
    spk_list = list(FOUR_SPEAKERS.values())
    room2 = build_room(spk_list, [ERROR_MIC_POS], eval_pos)
    H_ref2, _, H_pri_train2, H_sec_train2, H_pri_eval2, H_sec_eval2 = extract_paths(
        room2, M=4, K_train=1, num_eval=n_eval, num_extra_refs=0
    )
    H_sec_combined = np.zeros(512)
    for m in range(4):
        ssec = H_sec_train2[m][0]
        H_sec_combined[:len(ssec)] += ssec
    H_sec_est_pseudo = H_sec_combined * (1 + 0.05 * np.random.randn(len(H_sec_combined)))
    pseudo_anti = train_pseudo(noise, H_pri_train2[0],
                               [H_sec_train2[m][0] for m in range(4)],
                               H_ref2, H_sec_est_pseudo)
    H_sec_train_s1 = [H_sec_train2[m][0] for m in range(4)]
    H_sec_est_s1 = [s * (1 + 0.05 * np.random.randn(len(s))) for s in H_sec_train_s1]
    stage1_anti = train_stage1(noise, H_pri_train2[0], H_sec_train_s1, H_ref2, H_sec_est_s1)
    H_sec_eval_pseudo = [[
        sum(H_sec_eval2[m][e] for m in range(4)) for e in range(n_eval)
    ]]
    pseudo_grid = evaluate_grid(noise, pseudo_anti, H_pri_eval2, H_sec_eval_pseudo).reshape(ny, nx)
    stage1_grid = evaluate_grid(noise, stage1_anti, H_pri_eval2, H_sec_eval2).reshape(ny, nx)

    # ----- 4-speaker + K=4 (Stage 2) -----
    print("[4/5] Stage 2...")
    np.random.seed(42)
    room3 = build_room(spk_list, ERROR_MICS_K4, eval_pos)
    _, _, H_pri_train3, H_sec_train3, H_pri_eval3, H_sec_eval3 = extract_paths(
        room3, M=4, K_train=4, num_eval=n_eval, num_extra_refs=0
    )
    H_sec_est_s2 = [
        [s * (1 + 0.05 * np.random.randn(len(s))) for s in row] for row in H_sec_train3
    ]
    stage2_anti = train_stage2(noise, H_pri_train3, H_sec_train3, H_ref2, H_sec_est_s2)
    stage2_grid = evaluate_grid(noise, stage2_anti, H_pri_eval3, H_sec_eval3).reshape(ny, nx)

    # ----- N=4 ref + 4 spk + K=4 (Stage 3) -----
    print("[5/5] Stage 3...")
    np.random.seed(42)
    room4 = build_room(spk_list, ERROR_MICS_K4, eval_pos, num_ref_mics_extra=4)
    H_ref4, H_extra4, H_pri_train4, H_sec_train4, H_pri_eval4, H_sec_eval4 = \
        extract_paths(room4, M=4, K_train=4, num_eval=n_eval, num_extra_refs=4)
    H_sec_est_s3 = [
        [s * (1 + 0.05 * np.random.randn(len(s))) for s in row] for row in H_sec_train4
    ]
    stage3_anti = train_stage3(noise, H_pri_train4, H_sec_train4, H_extra4, H_sec_est_s3)
    stage3_grid = evaluate_grid(noise, stage3_anti, H_pri_eval4, H_sec_eval4).reshape(ny, nx)

    grids = [siso_grid, pseudo_grid, stage1_grid, stage2_grid, stage3_grid]
    titles = [
        'SISO\n(1 spk, 1 err mic)',
        'Pseudo-SIMO\n(4 spk broadcast)',
        'Stage 1 SIMO\n(4 spk indep)',
        'Stage 2 SIMO+multi-err\n(4 spk + 4 err)',
        'Stage 3 Full MIMO\n(4 ref + 4 spk + 4 err)',
    ]
    speakers_per = [[SISO_SPEAKER]] + [list(FOUR_SPEAKERS.values())] * 4
    err_mics_per = [[ERROR_MIC_POS], [ERROR_MIC_POS], [ERROR_MIC_POS],
                    ERROR_MICS_K4, ERROR_MICS_K4]

    vmax = max(np.max(np.abs(g)) for g in grids)
    vmax = max(vmax, 5)

    print(f"\nGenerating 1×5 heatmap (vmax={vmax:.1f} dB)...")
    fig = plt.figure(figsize=(22, 5.5))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.04])

    extent = [0, ROOM_DIMS[0], 0, ROOM_DIMS[1]]
    for col in range(5):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(grids[col], origin='lower', extent=extent,
                       cmap='RdBu', vmin=-vmax, vmax=vmax,
                       interpolation='nearest', aspect='equal')
        ax.set_title(titles[col], fontsize=11, fontweight='bold')
        ax.set_xlabel('x (m) — front → rear', fontsize=9)
        if col == 0:
            ax.set_ylabel('y (m)', fontsize=10)

        # Car outline + seats
        ax.add_patch(Rectangle((0, 0), ROOM_DIMS[0], ROOM_DIMS[1],
                               fill=False, edgecolor='black', linewidth=1.2))
        ax.add_patch(Rectangle((2.0, 0.2), 0.6, 0.7, fill=False,
                               edgecolor='gray', linestyle='--', linewidth=0.6, alpha=0.7))
        ax.add_patch(Rectangle((2.0, 1.0), 0.6, 0.7, fill=False,
                               edgecolor='gray', linestyle='--', linewidth=0.6, alpha=0.7))

        ax.plot(NOISE_POS[0], NOISE_POS[1], 'r*', markersize=12, mec='black', mew=0.5)
        ax.plot(REF_MIC_POS[0], REF_MIC_POS[1], 'g^', markersize=9, mec='black', mew=0.5)
        for em in err_mics_per[col]:
            ax.plot(em[0], em[1], 'kX', markersize=11, mec='white', mew=0.5)
        for spk in speakers_per[col]:
            ax.plot(spk[0], spk[1], 'bs', markersize=7, mec='black', mew=0.5)

        # Stats overlay
        g = grids[col]
        ax.text(0.01, 0.98,
                f"max: {np.max(g):+.1f} dB\nmin: {np.min(g):+.1f} dB\n"
                f"mean: {np.mean(g):+.1f} dB\n"
                f"%cancel: {100*np.mean(g>0):.0f}%",
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    cax = fig.add_subplot(gs[:, 5])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Noise reduction (dB)\nblue = cancellation, red = amplification',
                   fontsize=10)

    fig.suptitle(f'Full-Cabin Cancellation Patterns — Algorithm Comparison '
                 f'({nx}×{ny}={nx*ny} cells, 5×5 cm grid, '
                 f'trained on {Path(AUDIO_FILE).name})',
                 fontsize=12, fontweight='bold', y=1.02)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=140, bbox_inches='tight', facecolor='white')
    print(f"Saved: {OUTPUT_PATH}")
    plt.close()

    # Summary
    print("\n" + "=" * 80)
    print(" Full-cabin spatial summary")
    print("=" * 80)
    print(f"{'Algorithm':<28} | {'Mean':>7} | {'Max':>7} | {'Min':>7} | "
          f"{'%cancel':>7} | {'%amplify':>8}")
    print("-" * 80)
    for label, g in zip(['SISO', 'Pseudo-SIMO', 'Stage 1 SIMO',
                         'Stage 2 SIMO+multi-err', 'Stage 3 Full MIMO'], grids):
        print(f"{label:<28} | {np.mean(g):>+6.2f} | {np.max(g):>+6.2f} | "
              f"{np.min(g):>+6.2f} | {100*np.mean(g>0):>6.1f}% | "
              f"{100*np.mean(g<0):>7.1f}%")


if __name__ == '__main__':
    main()
