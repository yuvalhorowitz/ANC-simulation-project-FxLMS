"""
Generate cancellation heatmaps for ALL 5 algorithms on la_varying.wav:
  - One combined 1×5 PNG (overview)
  - 5 individual PNGs (one per algorithm) each with explanation text

Output: output/plots/heatmaps_la_varying/
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
DURATION_TRAIN = 10.0  # 10 seconds for stable convergence
AUDIO_FILE = 'real_noises/la_varying.wav'

CELL_SIZE_M = 0.05
EVAL_Z = ERROR_MIC_POS[2]
GRID_INSET = 0.025

OUTPUT_DIR = Path(__file__).parent.parent.parent / 'output' / 'plots' / 'heatmaps_la_varying'


def make_grid():
    x = np.arange(GRID_INSET, ROOM_DIMS[0] - GRID_INSET + 1e-9, CELL_SIZE_M)
    y = np.arange(GRID_INSET, ROOM_DIMS[1] - GRID_INSET + 1e-9, CELL_SIZE_M)
    pos = [[float(xi), float(yi), float(EVAL_Z)] for yi in y for xi in x]
    return pos, x, y


def build_room(speaker_positions, training_error_mics, eval_positions, num_ref_mics_extra=0):
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


# Trainers
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


# Explanations (the take-aways for each algorithm based on what its heatmap shows)
EXPLANATIONS = {
    'SISO': (
        "What you're seeing\n"
        "• A single dashboard speaker tries to cancel noise at one error mic.\n"
        "• Peak cancellation is shallow (≤+1 dB) — the speaker is far from the listener.\n"
        "• Strong red blob near the speaker: anti-noise leaks outward and ADDS energy where it isn't wanted.\n"
        "\n"
        "Main lesson\n"
        "• A single-speaker / single-mic ANC system is fundamentally limited:\n"
        "  not enough acoustic actuators to shape the cancellation field.\n"
        "• Most of the cabin is amplified, not cancelled (waterbed effect)."
    ),
    'Pseudo-SIMO': (
        "What you're seeing\n"
        "• 4 speakers all emit the SAME anti-noise signal (broadcast).\n"
        "• 4× the acoustic energy → deeper cancellation than SISO (peak ~+5 dB).\n"
        "• Soft blue glow across cabin; less aggressive red — the energy spreads.\n"
        "\n"
        "Main lesson\n"
        "• Adding speakers helps even without independent control.\n"
        "• But it can't sculpt the sound field — every speaker says the same thing.\n"
        "• This is what most consumer '4-speaker mode' systems actually do."
    ),
    'Stage 1 SIMO': (
        "What you're seeing\n"
        "• 4 speakers, each with its OWN adaptive filter.\n"
        "• Deeper peak (+6–7 dB) — the filters can constructively combine at the target.\n"
        "• But each speaker pumps a different signal, so red zones are stronger:\n"
        "  the cabin becomes more 'sculpted' — deep wells next to high peaks.\n"
        "\n"
        "Main lesson\n"
        "• Independent control = sharper cancellation at the trained point.\n"
        "• The cost: more aggressive amplification elsewhere (waterbed redistribution)."
    ),
    'Stage 2 SIMO+multi-err': (
        "What you're seeing\n"
        "• 4 speakers + 4 error mics around the head zone (a 2×2 grid ±5 cm).\n"
        "• Cancellation is FORCED to be similar at all 4 head-zone points.\n"
        "• Peak similar to Stage 1, but the quiet zone is WIDER and more uniform.\n"
        "\n"
        "Main lesson\n"
        "• Multiple error mics widen the 'zone of quiet' (good for a moving head).\n"
        "• Constraining cancellation at K points means deeper amplification elsewhere\n"
        "  — the waterbed effect is REDISTRIBUTED, not eliminated."
    ),
    'Stage 3 Full MIMO': (
        "What you're seeing\n"
        "• 4 reference mics (independent inputs) + 4 speakers + 4 error mics.\n"
        "• N×M = 16 independent filters give the system maximum degrees of freedom.\n"
        "• Peak cancellation jumps to +10–12 dB — best of all configurations.\n"
        "• Cancellation extends further into the cabin; less concentrated.\n"
        "\n"
        "Main lesson\n"
        "• Full MIMO = more sensing + more control = best result.\n"
        "• Cost: 4× more weights, slower convergence, smaller step size required.\n"
        "• This is the architecture commercial automotive ANC systems use."
    ),
}


def draw_one_panel(ax, grid, title, speakers, error_mics, vmax, extent,
                    show_axis_labels=True, ref_mics=None):
    """Draw a single cabin heatmap panel.

    ref_mics: list of [x, y, z] positions to plot as green triangles.
              If None, defaults to a single triangle at REF_MIC_POS.
    """
    im = ax.imshow(grid, origin='lower', extent=extent,
                   cmap='RdBu', vmin=-vmax, vmax=vmax,
                   interpolation='nearest', aspect='equal')
    ax.set_title(title, fontsize=11, fontweight='bold')
    if show_axis_labels:
        ax.set_xlabel('x (m) — front → rear', fontsize=9)
        ax.set_ylabel('y (m)', fontsize=9)

    # Car outline + seats
    ax.add_patch(Rectangle((0, 0), ROOM_DIMS[0], ROOM_DIMS[1],
                           fill=False, edgecolor='black', linewidth=1.2))
    ax.add_patch(Rectangle((2.0, 0.2), 0.6, 0.7, fill=False,
                           edgecolor='gray', linestyle='--', linewidth=0.6, alpha=0.7))
    ax.add_patch(Rectangle((2.0, 1.0), 0.6, 0.7, fill=False,
                           edgecolor='gray', linestyle='--', linewidth=0.6, alpha=0.7))

    ax.plot(NOISE_POS[0], NOISE_POS[1], 'r*', markersize=12, mec='black', mew=0.5, label='noise')

    # Reference mic(s) — varies per algorithm
    rms = ref_mics if ref_mics is not None else [REF_MIC_POS]
    for rm in rms:
        ax.plot(rm[0], rm[1], 'g^', markersize=9, mec='black', mew=0.5)

    for em in error_mics:
        ax.plot(em[0], em[1], 'kX', markersize=11, mec='white', mew=0.5)
    for spk in speakers:
        ax.plot(spk[0], spk[1], 'bs', markersize=7, mec='black', mew=0.5)

    # Stats overlay
    ax.text(0.01, 0.98,
            f"max: {np.max(grid):+.1f} dB\nmin: {np.min(grid):+.1f} dB\n"
            f"mean: {np.mean(grid):+.1f} dB\n"
            f"%cancel: {100*np.mean(grid>0):.0f}%",
            transform=ax.transAxes, fontsize=8.5, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    return im


def save_individual_panel(grid, label, speakers, error_mics, vmax_global,
                          extent, audio_name, ref_mics=None):
    """One algorithm → one PNG with figure + explanation text."""
    explanation = EXPLANATIONS[label]

    fig = plt.figure(figsize=(15, 7.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0], wspace=0.10)

    ax_map = fig.add_subplot(gs[0, 0])
    # Per-panel vmax (so individual contrast looks good)
    vmax_local = max(np.max(np.abs(grid)), 3)
    im = draw_one_panel(
        ax_map, grid,
        f"{label}\n(trained on {audio_name})",
        speakers, error_mics, vmax_local, extent, ref_mics=ref_mics,
    )

    cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.02)
    cbar.set_label('Noise reduction (dB)\nblue = cancellation, red = amplification',
                   fontsize=9)

    # Explanation panel (no axes, just text)
    ax_text = fig.add_subplot(gs[0, 1])
    ax_text.set_axis_off()
    ax_text.text(0.02, 0.98, explanation,
                 transform=ax_text.transAxes,
                 va='top', ha='left',
                 fontsize=10.5, family='sans-serif',
                 linespacing=1.4,
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa',
                           edgecolor='#dee2e6'))

    # Quick legend at bottom
    legend_str = (
        "🔴 noise source   "
        "🟢 reference mic   "
        "🔵 speaker(s)   "
        "✕ error mic(s)"
    )
    fig.text(0.5, 0.02, legend_str, ha='center', va='bottom',
             fontsize=9, style='italic', color='#444')

    fig.suptitle(label, fontsize=14, fontweight='bold', y=0.99)

    safe_name = label.replace(' ', '_').replace('+', 'p').replace('/', '-')
    out_path = OUTPUT_DIR / f"heatmap_{safe_name}.png"
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved individual: {out_path.name}")
    return out_path


def save_combined_1x5(grids, labels, speakers_per, err_mics_per, extent,
                      audio_name, ref_mics_per=None):
    """Combined 1×5 PNG (wide layout)."""
    vmax = max(np.max(np.abs(g)) for g in grids)
    vmax = max(vmax, 5)

    fig = plt.figure(figsize=(22, 5.5))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.04])

    im = None
    for col in range(5):
        ax = fig.add_subplot(gs[0, col])
        rms = ref_mics_per[col] if ref_mics_per else None
        im = draw_one_panel(ax, grids[col], labels[col],
                            speakers_per[col], err_mics_per[col],
                            vmax, extent, show_axis_labels=(col == 0),
                            ref_mics=rms)

    cax = fig.add_subplot(gs[:, 5])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Noise reduction (dB)\nblue = cancellation, red = amplification',
                   fontsize=10)

    fig.suptitle(f'Full-Cabin Cancellation Patterns — Algorithm Comparison '
                 f'(trained on {audio_name})',
                 fontsize=12, fontweight='bold', y=1.02)

    out_path = OUTPUT_DIR / 'heatmap_1x5_combined.png'
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved combined: {out_path.name}")


def save_combined_221(grids, labels, speakers_per, err_mics_per, extent,
                       audio_name, ref_mics_per=None):
    """Combined 2-2-1 square-ish PNG: baselines / SIMO / Full MIMO."""
    vmax = max(np.max(np.abs(g)) for g in grids)
    vmax = max(vmax, 5)

    fig = plt.figure(figsize=(15, 14))
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 0.04],
                          hspace=0.30, wspace=0.18)

    rms_for = lambda i: (ref_mics_per[i] if ref_mics_per else None)

    im = None
    # Row 0: SISO, Pseudo-SIMO
    ax = fig.add_subplot(gs[0, 0])
    im = draw_one_panel(ax, grids[0], labels[0], speakers_per[0],
                        err_mics_per[0], vmax, extent, show_axis_labels=True,
                        ref_mics=rms_for(0))
    ax = fig.add_subplot(gs[0, 1])
    im = draw_one_panel(ax, grids[1], labels[1], speakers_per[1],
                        err_mics_per[1], vmax, extent, show_axis_labels=False,
                        ref_mics=rms_for(1))

    # Row 1: Stage 1, Stage 2
    ax = fig.add_subplot(gs[1, 0])
    im = draw_one_panel(ax, grids[2], labels[2], speakers_per[2],
                        err_mics_per[2], vmax, extent, show_axis_labels=True,
                        ref_mics=rms_for(2))
    ax = fig.add_subplot(gs[1, 1])
    im = draw_one_panel(ax, grids[3], labels[3], speakers_per[3],
                        err_mics_per[3], vmax, extent, show_axis_labels=False,
                        ref_mics=rms_for(3))

    # Row 2: Stage 3 — centered (span both columns)
    ax = fig.add_subplot(gs[2, :2])
    im = draw_one_panel(ax, grids[4], labels[4], speakers_per[4],
                        err_mics_per[4], vmax, extent, show_axis_labels=True,
                        ref_mics=rms_for(4))

    # Single shared colorbar on the right
    cax = fig.add_subplot(gs[:, 2])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Noise reduction (dB)\nblue = cancellation, red = amplification',
                   fontsize=10)

    fig.suptitle(f'Full-Cabin Cancellation Patterns — Algorithm Comparison '
                 f'(trained on {audio_name})\n'
                 f'top: baselines · middle: Stage 1/2 SIMO · bottom: Stage 3 Full MIMO',
                 fontsize=12, fontweight='bold', y=0.995)

    out_path = OUTPUT_DIR / 'heatmap_2_2_1_combined.png'
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved combined: {out_path.name}")


def main():
    print("=" * 70)
    print(f" Heatmaps for {AUDIO_FILE} (combined 1×5 + 5 individual)")
    print("=" * 70)

    print("\nLoading noise...")
    noise = NoiseMixer(FS).load_audio_file(AUDIO_FILE, duration=DURATION_TRAIN)
    eval_pos, x_centers, y_centers = make_grid()
    nx, ny = len(x_centers), len(y_centers)
    n_eval = len(eval_pos)
    print(f"  Grid: {nx}×{ny} = {n_eval} cells\n")

    # --- SISO ---
    print("[1/5] SISO...")
    np.random.seed(42)
    room = build_room([SISO_SPEAKER], [ERROR_MIC_POS], eval_pos)
    H_ref, _, H_pri_train, H_sec_train, H_pri_eval, H_sec_eval = extract_paths(
        room, M=1, K_train=1, num_eval=n_eval
    )
    H_sec_est = H_sec_train[0][0] * (1 + 0.05 * np.random.randn(len(H_sec_train[0][0])))
    siso_anti = train_siso(noise, H_pri_train[0], H_sec_train[0][0], H_ref, H_sec_est)
    siso_grid = evaluate_grid(noise, siso_anti, H_pri_eval, H_sec_eval).reshape(ny, nx)

    # --- Pseudo + Stage 1 ---
    print("[2-3/5] Pseudo-SIMO + Stage 1...")
    np.random.seed(42)
    spk_list = list(FOUR_SPEAKERS.values())
    room2 = build_room(spk_list, [ERROR_MIC_POS], eval_pos)
    H_ref2, _, H_pri_train2, H_sec_train2, H_pri_eval2, H_sec_eval2 = extract_paths(
        room2, M=4, K_train=1, num_eval=n_eval
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

    # --- Stage 2 ---
    print("[4/5] Stage 2...")
    np.random.seed(42)
    room3 = build_room(spk_list, ERROR_MICS_K4, eval_pos)
    _, _, H_pri_train3, H_sec_train3, H_pri_eval3, H_sec_eval3 = extract_paths(
        room3, M=4, K_train=4, num_eval=n_eval
    )
    H_sec_est_s2 = [
        [s * (1 + 0.05 * np.random.randn(len(s))) for s in row] for row in H_sec_train3
    ]
    stage2_anti = train_stage2(noise, H_pri_train3, H_sec_train3, H_ref2, H_sec_est_s2)
    stage2_grid = evaluate_grid(noise, stage2_anti, H_pri_eval3, H_sec_eval3).reshape(ny, nx)

    # --- Stage 3 ---
    print("[5/5] Stage 3...")
    np.random.seed(42)
    room4 = build_room(spk_list, ERROR_MICS_K4, eval_pos, num_ref_mics_extra=4)
    _, H_extra4, H_pri_train4, H_sec_train4, H_pri_eval4, H_sec_eval4 = \
        extract_paths(room4, M=4, K_train=4, num_eval=n_eval, num_extra_refs=4)
    H_sec_est_s3 = [
        [s * (1 + 0.05 * np.random.randn(len(s))) for s in row] for row in H_sec_train4
    ]
    stage3_anti = train_stage3(noise, H_pri_train4, H_sec_train4, H_extra4, H_sec_est_s3)
    stage3_grid = evaluate_grid(noise, stage3_anti, H_pri_eval4, H_sec_eval4).reshape(ny, nx)

    grids = [siso_grid, pseudo_grid, stage1_grid, stage2_grid, stage3_grid]
    labels = ['SISO', 'Pseudo-SIMO', 'Stage 1 SIMO',
              'Stage 2 SIMO+multi-err', 'Stage 3 Full MIMO']
    speakers_per = [[SISO_SPEAKER]] + [list(FOUR_SPEAKERS.values())] * 4
    err_mics_per = [[ERROR_MIC_POS], [ERROR_MIC_POS], [ERROR_MIC_POS],
                    ERROR_MICS_K4, ERROR_MICS_K4]
    # Reference mics: SISO/Pseudo/Stage1/Stage2 use the single legacy ref mic;
    # Stage 3 uses the 4 mics from FOUR_REF_MICS (firewall, floor, a-pillar, dashboard).
    ref_mics_per = [
        [REF_MIC_POS],                          # SISO
        [REF_MIC_POS],                          # Pseudo-SIMO
        [REF_MIC_POS],                          # Stage 1 SIMO
        [REF_MIC_POS],                          # Stage 2 SIMO+multi-err
        list(FOUR_REF_MICS.values()),           # Stage 3: N=4 ref mics
    ]

    extent = [0, ROOM_DIMS[0], 0, ROOM_DIMS[1]]
    audio_name = Path(AUDIO_FILE).name

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nGenerating combined 1×5 heatmap...")
    save_combined_1x5(grids, [
        f'{l}\n({c})' for l, c in zip(labels, [
            '1 spk, 1 err mic', '4 spk broadcast', '4 spk indep',
            '4 spk + 4 err', '4 ref + 4 spk + 4 err'
        ])
    ], speakers_per, err_mics_per, extent, audio_name,
       ref_mics_per=ref_mics_per)

    print("\nGenerating combined 2-2-1 square heatmap...")
    save_combined_221(grids, [
        f'{l}\n({c})' for l, c in zip(labels, [
            '1 spk, 1 err mic', '4 spk broadcast', '4 spk indep',
            '4 spk + 4 err', '4 ref + 4 spk + 4 err'
        ])
    ], speakers_per, err_mics_per, extent, audio_name,
       ref_mics_per=ref_mics_per)

    print("\nGenerating 5 individual heatmaps...")
    for grid, label, spks, ems, rms in zip(grids, labels, speakers_per,
                                            err_mics_per, ref_mics_per):
        save_individual_panel(grid, label, spks, ems, None, extent, audio_name,
                              ref_mics=rms)

    # Summary
    print("\n" + "=" * 80)
    print(" Full-cabin spatial summary (la_varying.wav, 10s)")
    print("=" * 80)
    print(f"{'Algorithm':<28} | {'Mean':>7} | {'Max':>7} | {'Min':>7} | "
          f"{'%cancel':>7} | {'%amplify':>8}")
    print("-" * 80)
    for label, g in zip(labels, grids):
        print(f"{label:<28} | {np.mean(g):>+6.2f} | {np.max(g):>+6.2f} | "
              f"{np.min(g):>+6.2f} | {100*np.mean(g>0):>6.1f}% | "
              f"{100*np.mean(g<0):>7.1f}%")


if __name__ == '__main__':
    main()
