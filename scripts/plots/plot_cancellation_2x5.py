"""
2×5 Cancellation Heatmap — Most Comprehensive Comparison

Top row: full cabin (5×5 cm grid, 90×37 = 3330 cells) at driver ear height
Bottom row: head zone (1 cm grid, 30×30 cm closeup, 30×30 = 900 cells)

5 columns: SISO / Pseudo-SIMO / Stage 1 / Stage 2 / Stage 3

Trains each model with the canonical error mic, then evaluates the cancellation
field at every grid point. Shows the trade-off between deep cancellation at
the target point (bottom) and broader amplification spillage (top).

Output: output/plots/cancellation_heatmap_2x5.png
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
HEAD_ZONE_OFFSETS_TRAIN = [(0.05, 0.05), (-0.05, 0.05), (0.05, -0.05), (-0.05, -0.05)]
ERROR_MICS_K4 = [
    [ERROR_MIC_POS[0], ERROR_MIC_POS[1] + dy, ERROR_MIC_POS[2] + dz]
    for dy, dz in HEAD_ZONE_OFFSETS_TRAIN
]

FILTER_LENGTH = 256
FILTER_LENGTH_STAGE3 = 256
STEP_SIZE = 0.003
STEP_SIZE_STAGE3 = 0.001
FS = 16000
DURATION_TRAIN = 5.0
AUDIO_FILE = 'real_noises/realcar1.wav'

# Two grid resolutions
FULL_CELL_M = 0.05      # 5 cm full cabin
FULL_INSET = 0.025
EVAL_Z = ERROR_MIC_POS[2]

ZOOM_HALF_M = 0.15      # ±15 cm head zone
ZOOM_N = 11             # 11×11 close-up

OUTPUT_PATH = Path(__file__).parent.parent.parent / 'output' / 'plots' / 'cancellation_heatmap_2x5.png'


def make_full_cabin_grid():
    x = np.arange(FULL_INSET, ROOM_DIMS[0] - FULL_INSET + 1e-9, FULL_CELL_M)
    y = np.arange(FULL_INSET, ROOM_DIMS[1] - FULL_INSET + 1e-9, FULL_CELL_M)
    pos = [[float(xi), float(yi), float(EVAL_Z)] for yi in y for xi in x]
    return pos, x, y


def make_zoom_grid():
    cx, cy, cz = ERROR_MIC_POS
    y_offs = np.linspace(-ZOOM_HALF_M, ZOOM_HALF_M, ZOOM_N)
    z_offs = np.linspace(-ZOOM_HALF_M, ZOOM_HALF_M, ZOOM_N)
    pos = []
    for z_off in z_offs:
        for y_off in y_offs:
            p = [cx, cy + y_off, cz + z_off]
            p[1] = max(0.05, min(ROOM_DIMS[1] - 0.05, p[1]))
            p[2] = max(0.05, min(ROOM_DIMS[2] - 0.05, p[2]))
            pos.append(p)
    return pos, y_offs, z_offs


def build_room(speaker_positions, training_error_mics, eval_positions,
               num_ref_mics_extra=0):
    """
    Layout:
      Source 0: noise. Sources 1..M: speakers.
      Mics: 0 = ref, [1..N] = extra ref mics (Stage 3 only),
            [next K_train] = training error mics, [rest] = eval grid.
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
    H_extra_refs = [room.rir[1 + n][0][:L] for n in range(num_extra_refs)]

    H_pri_train = [room.rir[base_ref + k][0][:L] for k in range(K_train)]
    H_sec_train = [
        [room.rir[base_ref + k][1 + m][:L] for k in range(K_train)] for m in range(M)
    ]
    H_pri_eval = [room.rir[base_ref + K_train + e][0][:L] for e in range(num_eval)]
    H_sec_eval = [
        [room.rir[base_ref + K_train + e][1 + m][:L] for e in range(num_eval)]
        for m in range(M)
    ]
    return H_ref, H_extra_refs, H_pri_train, H_sec_train, H_pri_eval, H_sec_eval


# ---------- Trainers (same logic as scripts/plots/plot_cancellation_full_cabin_4way.py) ----------

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
    """N reference signals fed independently."""
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
    print("=" * 75)
    print(" 2×5 Cancellation Heatmap — Comprehensive Algorithm Comparison")
    print("=" * 75)

    print("\nLoading noise...")
    noise = NoiseMixer(FS).load_audio_file(AUDIO_FILE, duration=DURATION_TRAIN)
    print(f"  {len(noise)} samples ({len(noise)/FS:.1f} s)")

    full_pos, x_centers, y_centers = make_full_cabin_grid()
    nx, ny = len(x_centers), len(y_centers)
    n_full = len(full_pos)

    zoom_pos, y_offs, z_offs = make_zoom_grid()
    n_zoom = len(zoom_pos)
    eval_positions = full_pos + zoom_pos
    n_eval = n_full + n_zoom
    print(f"\nFull cabin grid: {nx}×{ny} = {n_full} cells (5 cm)")
    print(f"Zoom head zone: {ZOOM_N}×{ZOOM_N} = {n_zoom} cells (3 cm spacing)")
    print(f"Total evaluations per algorithm: {n_eval}")

    # ----- SISO room -----
    print("\n[1/5] Building SISO room...")
    np.random.seed(42)
    room = build_room([SISO_SPEAKER], [ERROR_MIC_POS], eval_positions)
    H_ref, _, H_pri_train, H_sec_train, H_pri_eval, H_sec_eval = extract_paths(
        room, M=1, K_train=1, num_eval=n_eval, num_extra_refs=0
    )
    H_sec_est = H_sec_train[0][0] * (1 + 0.05 * np.random.randn(len(H_sec_train[0][0])))
    print("  Training SISO...")
    siso_anti = train_siso(noise, H_pri_train[0], H_sec_train[0][0], H_ref, H_sec_est)

    # SISO grid eval
    print("  Evaluating grid...")
    siso_at = evaluate_grid(noise, siso_anti, H_pri_eval, H_sec_eval)

    # ----- 4-Speaker room (used for Pseudo / Stage 1) -----
    print("\n[2/5 + 3/5] Building 4-speaker room (1 error mic)...")
    np.random.seed(42)
    spk_list = list(FOUR_SPEAKERS.values())
    room2 = build_room(spk_list, [ERROR_MIC_POS], eval_positions)
    H_ref2, _, H_pri_train2, H_sec_train2, H_pri_eval2, H_sec_eval2 = extract_paths(
        room2, M=4, K_train=1, num_eval=n_eval, num_extra_refs=0
    )

    # Pseudo combined estimate
    H_sec_combined = np.zeros(512)
    for m in range(4):
        ssec = H_sec_train2[m][0]
        H_sec_combined[:len(ssec)] += ssec
    H_sec_est_pseudo = H_sec_combined * (1 + 0.05 * np.random.randn(len(H_sec_combined)))

    print("  Training Pseudo-SIMO...")
    pseudo_anti = train_pseudo(noise, H_pri_train2[0],
                               [H_sec_train2[m][0] for m in range(4)],
                               H_ref2, H_sec_est_pseudo)

    H_sec_train_stage1 = [H_sec_train2[m][0] for m in range(4)]
    H_sec_est_stage1 = [s * (1 + 0.05 * np.random.randn(len(s))) for s in H_sec_train_stage1]
    print("  Training Stage 1 SIMO...")
    stage1_anti = train_stage1(noise, H_pri_train2[0], H_sec_train_stage1,
                               H_ref2, H_sec_est_stage1)

    print("  Evaluating Pseudo + Stage 1 grids...")
    H_sec_eval_pseudo = [[
        sum(H_sec_eval2[m][e] for m in range(4)) for e in range(n_eval)
    ]]
    pseudo_at = evaluate_grid(noise, pseudo_anti, H_pri_eval2, H_sec_eval_pseudo)
    stage1_at = evaluate_grid(noise, stage1_anti, H_pri_eval2, H_sec_eval2)

    # ----- 4-speaker + K=4 error mics room (Stage 2) -----
    print("\n[4/5] Building Stage 2 room (4 speakers, 4 error mics)...")
    np.random.seed(42)
    room3 = build_room(spk_list, ERROR_MICS_K4, eval_positions)
    H_ref3, _, H_pri_train3, H_sec_train3, H_pri_eval3, H_sec_eval3 = extract_paths(
        room3, M=4, K_train=4, num_eval=n_eval, num_extra_refs=0
    )
    H_sec_est_s2 = [
        [s * (1 + 0.05 * np.random.randn(len(s))) for s in row] for row in H_sec_train3
    ]
    print("  Training Stage 2...")
    stage2_anti = train_stage2(noise, H_pri_train3, H_sec_train3, H_ref3, H_sec_est_s2)

    print("  Evaluating grid...")
    stage2_at = evaluate_grid(noise, stage2_anti, H_pri_eval3, H_sec_eval3)

    # ----- N=4 ref mics + 4 speakers + K=4 error mics room (Stage 3) -----
    print("\n[5/5] Building Stage 3 room (4 ref mics + 4 speakers + 4 error mics)...")
    np.random.seed(42)
    room4 = build_room(spk_list, ERROR_MICS_K4, eval_positions, num_ref_mics_extra=4)
    H_ref4, H_extra_refs4, H_pri_train4, H_sec_train4, H_pri_eval4, H_sec_eval4 = \
        extract_paths(room4, M=4, K_train=4, num_eval=n_eval, num_extra_refs=4)
    # Stage 3 uses ALL 4 ref mics — H_ref4 (mic 0) is the original "ref",
    # and H_extra_refs4 are the 4 firewall/floor/a_pillar/dashboard mics.
    # We use the 4 extras as the N=4 references (matching playground default).
    all_refs_s3 = H_extra_refs4
    H_sec_est_s3 = [
        [s * (1 + 0.05 * np.random.randn(len(s))) for s in row] for row in H_sec_train4
    ]
    print("  Training Stage 3...")
    stage3_anti = train_stage3(noise, H_pri_train4, H_sec_train4, all_refs_s3, H_sec_est_s3)

    print("  Evaluating grid...")
    stage3_at = evaluate_grid(noise, stage3_anti, H_pri_eval4, H_sec_eval4)

    # ----- Reshape for plotting -----
    def split_full_zoom(at):
        return at[:n_full].reshape(ny, nx), at[n_full:].reshape(ZOOM_N, ZOOM_N)

    siso_full, siso_zoom = split_full_zoom(siso_at)
    pseudo_full, pseudo_zoom = split_full_zoom(pseudo_at)
    stage1_full, stage1_zoom = split_full_zoom(stage1_at)
    stage2_full, stage2_zoom = split_full_zoom(stage2_at)
    stage3_full, stage3_zoom = split_full_zoom(stage3_at)

    full_grids = [siso_full, pseudo_full, stage1_full, stage2_full, stage3_full]
    zoom_grids = [siso_zoom, pseudo_zoom, stage1_zoom, stage2_zoom, stage3_zoom]
    titles = [
        'SISO\n(1 spk, 1 err mic)',
        'Pseudo-SIMO\n(4 spk broadcast)',
        'Stage 1 SIMO\n(4 spk indep)',
        'Stage 2 SIMO+multi-err\n(4 spk + 4 err mics)',
        'Stage 3 Full MIMO\n(4 ref + 4 spk + 4 err)',
    ]
    speakers_per = [[SISO_SPEAKER]] + [list(FOUR_SPEAKERS.values())] * 4
    err_mics_per = [[ERROR_MIC_POS], [ERROR_MIC_POS], [ERROR_MIC_POS],
                    ERROR_MICS_K4, ERROR_MICS_K4]

    # Symmetric color scale across all panels
    vmax_full = max(np.max(np.abs(g)) for g in full_grids)
    vmax_zoom = max(np.max(np.abs(g)) for g in zoom_grids)
    vmax_full = max(vmax_full, 5)
    vmax_zoom = max(vmax_zoom, 5)
    # Use a single shared color scale for fairness
    vmax = max(vmax_full, vmax_zoom)

    print(f"\nGenerating 2×5 heatmap (vmax={vmax:.1f} dB)...")
    fig = plt.figure(figsize=(22, 9))
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 1, 0.05],
                          height_ratios=[2.4, 1])

    # Top row: full cabin
    full_extent = [0, ROOM_DIMS[0], 0, ROOM_DIMS[1]]
    for col in range(5):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(full_grids[col], origin='lower', extent=full_extent,
                       cmap='RdBu', vmin=-vmax, vmax=vmax,
                       interpolation='nearest', aspect='equal')
        ax.set_title(titles[col], fontsize=10, fontweight='bold')
        ax.set_xlabel('x (m) — front → rear', fontsize=8)
        if col == 0:
            ax.set_ylabel('Full cabin\ny (m)', fontsize=10, fontweight='bold')

        ax.add_patch(Rectangle((0, 0), ROOM_DIMS[0], ROOM_DIMS[1],
                               fill=False, edgecolor='black', linewidth=1))
        ax.add_patch(Rectangle((2.0, 0.2), 0.6, 0.7, fill=False,
                               edgecolor='gray', linestyle='--', linewidth=0.5, alpha=0.6))
        ax.add_patch(Rectangle((2.0, 1.0), 0.6, 0.7, fill=False,
                               edgecolor='gray', linestyle='--', linewidth=0.5, alpha=0.6))

        ax.plot(NOISE_POS[0], NOISE_POS[1], 'r*', markersize=10, mec='black', mew=0.5)
        ax.plot(REF_MIC_POS[0], REF_MIC_POS[1], 'g^', markersize=8, mec='black', mew=0.5)
        for em in err_mics_per[col]:
            ax.plot(em[0], em[1], 'kX', markersize=10, mec='white', mew=0.5)
        for spk in speakers_per[col]:
            ax.plot(spk[0], spk[1], 'bs', markersize=6, mec='black', mew=0.5)

        # Stats overlay
        g = full_grids[col]
        max_db = np.max(g); min_db = np.min(g); mean_db = np.mean(g)
        pct_pos = 100 * np.mean(g > 0)
        ax.text(0.01, 0.98,
                f"max: {max_db:+.1f} dB\nmin: {min_db:+.1f} dB\n"
                f"mean: {mean_db:+.1f} dB\n%cancel: {pct_pos:.0f}%",
                transform=ax.transAxes, fontsize=7, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # Bottom row: head-zone closeup
    zoom_extent = [y_offs[0]*100, y_offs[-1]*100, z_offs[0]*100, z_offs[-1]*100]
    for col in range(5):
        ax = fig.add_subplot(gs[1, col])
        im_z = ax.imshow(zoom_grids[col], origin='lower', extent=zoom_extent,
                         cmap='RdBu', vmin=-vmax, vmax=vmax,
                         interpolation='bilinear', aspect='equal')
        ax.set_xlabel('Y offset (cm)', fontsize=8)
        if col == 0:
            ax.set_ylabel('Head-zone closeup\nZ offset (cm)', fontsize=10, fontweight='bold')
        ax.plot(0, 0, 'kX', markersize=12, mec='white', mew=0.5)

        # Stats
        g = zoom_grids[col]
        ax.text(0.02, 0.98,
                f"max: {np.max(g):+.1f} dB\nmin: {np.min(g):+.1f} dB\n"
                f"mean: {np.mean(g):+.1f} dB",
                transform=ax.transAxes, fontsize=7, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # Shared colorbar in last column
    cax = fig.add_subplot(gs[:, 5])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Noise reduction (dB)\nblue = cancellation, red = amplification',
                   fontsize=10)

    fig.suptitle(f'Cancellation Patterns — Algorithm Comparison '
                 f'(trained on {Path(AUDIO_FILE).name}, '
                 f'top: full cabin {nx}×{ny} cells; bottom: head zone {ZOOM_N}×{ZOOM_N} cells)',
                 fontsize=12, fontweight='bold', y=0.995)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=140, bbox_inches='tight', facecolor='white')
    print(f"Saved: {OUTPUT_PATH}")
    plt.close()

    # Summary
    print("\n" + "=" * 92)
    print(" Summary")
    print("=" * 92)
    print(f"{'Algorithm':<28} | {'FULL CABIN':^25} | {'HEAD ZONE':^25}")
    print(f"{'':<28} | {'mean':>7} {'max':>7} {'min':>7} | "
          f"{'mean':>7} {'max':>7} {'min':>7}")
    print("-" * 92)
    for label, full, zoom in zip(
        ['SISO', 'Pseudo-SIMO', 'Stage 1 SIMO', 'Stage 2 multi-err', 'Stage 3 Full MIMO'],
        full_grids, zoom_grids
    ):
        print(f"{label:<28} | {np.mean(full):>+6.2f} {np.max(full):>+6.2f} "
              f"{np.min(full):>+6.2f} | {np.mean(zoom):>+6.2f} {np.max(zoom):>+6.2f} "
              f"{np.min(zoom):>+6.2f}")


if __name__ == '__main__':
    main()
