"""
Cancellation Heatmap Comparison

After training each algorithm with the canonical error mic position, evaluates
noise reduction (in dB) at every point on a 30×30 cm spatial grid centered on
the original error mic. Uses a diverging colormap so:
  - Blue = positive dB (cancellation)
  - White = 0 dB (no effect)
  - Red = negative dB (amplification — the waterbed effect)

Compares:
  - SISO baseline (PlaygroundSimulation)
  - Pseudo-MIMO (MultiSpeakerSimulation, 4 speakers broadcast)
  - True MIMO Stage 1 (MIMOSimulation, 4 speakers independent)

This visualization exposes the spatial behavior of each algorithm: how the
"zone of quiet" is shaped, and where each method *amplifies* noise instead
of cancelling it.

Output: output/plots/cancellation_heatmap_stage1.png
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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
ERROR_MIC_POS = [2.5, 0.55, 1.05]   # center of the head zone

SISO_SPEAKER = [0.8, 0.25, 0.9]
FOUR_SPEAKERS = {
    'door_L': [2.0, 0.1, 0.4],
    'door_R': [2.0, 1.75, 0.4],
    'dash_L': [0.8, 0.25, 0.9],
    'dash_R': [0.8, 1.60, 0.9],
}

FILTER_LENGTH = 512
STEP_SIZE = 0.003
FS = 16000
DURATION_TRAIN = 10.0   # seconds to train each filter
AUDIO_FILE = 'real_noises/realcar1.wav'

# Spatial grid: 30 cm × 30 cm, 11×11 points centered on error mic (in y-z plane)
GRID_HALF_M = 0.15        # ±15 cm
GRID_N = 11               # 11×11 = 121 evaluation points
DURATION_EVAL = 1.0       # seconds at each grid point (with frozen filter)

OUTPUT_PATH = Path(__file__).parent.parent.parent / 'output' / 'plots' / 'cancellation_heatmap_stage1.png'


def make_room(extra_mics):
    """
    Build a room with sources (noise + speakers) and microphones (reference,
    error, then a list of evaluation mics for the spatial grid).
    """
    materials = {
        'ceiling': pra.Material(ABSORPTION * 1.1),
        'floor': pra.Material(ABSORPTION * 1.5),
        'east': pra.Material(ABSORPTION * 0.5),
        'west': pra.Material(ABSORPTION * 0.5),
        'north': pra.Material(ABSORPTION * 0.7),
        'south': pra.Material(ABSORPTION * 0.9),
    }
    room = pra.ShoeBox(ROOM_DIMS, fs=FS, materials=materials,
                       max_order=MAX_ORDER, air_absorption=True)
    return room, materials


def grid_positions():
    """Return list of (y, z) eval mic positions on the spatial grid."""
    y_offsets = np.linspace(-GRID_HALF_M, GRID_HALF_M, GRID_N)
    z_offsets = np.linspace(-GRID_HALF_M, GRID_HALF_M, GRID_N)
    positions = []
    for z_off in z_offsets:
        for y_off in y_offsets:
            pos = [ERROR_MIC_POS[0],
                   ERROR_MIC_POS[1] + y_off,
                   ERROR_MIC_POS[2] + z_off]
            # Clamp inside room
            pos[1] = max(0.05, min(ROOM_DIMS[1] - 0.05, pos[1]))
            pos[2] = max(0.05, min(ROOM_DIMS[2] - 0.05, pos[2]))
            positions.append(pos)
    return positions, y_offsets, z_offsets


# ============================================================
# Build a single room with all mics so all RIRs come from the same physics
# ============================================================

def build_full_room(speaker_positions: list):
    """
    Build the room with: noise source, speaker(s), reference mic, error mic,
    then GRID_N×GRID_N evaluation mics.
    Returns (room, eval_mic_offsets) — RIRs are computed.
    """
    room, _ = make_room(None)
    room.add_source(NOISE_POS)
    for spk in speaker_positions:
        room.add_source(spk)

    # Mic 0 = reference, Mic 1 = error, Mic 2..N+1 = eval grid
    eval_positions, y_offs, z_offs = grid_positions()
    all_mics = [REF_MIC_POS, ERROR_MIC_POS] + eval_positions
    mic_array = np.array(all_mics).T
    room.add_microphone_array(pra.MicrophoneArray(mic_array, fs=FS))
    room.compute_rir()

    return room, eval_positions, y_offs, z_offs


def extract_paths(room, num_speakers, num_eval):
    """
    Extract acoustic paths from room.rir. Returns:
      H_primary: noise → error mic
      H_secondary: list of speaker → error mic, length num_speakers
      H_reference: noise → reference mic
      H_primary_eval: list of noise → eval mic k, length num_eval
      H_secondary_eval: list of lists, [m][k] = speaker m → eval mic k
    """
    max_len = 512

    # Source 0 = noise; sources 1..num_speakers = speakers
    # Mic 0 = ref, mic 1 = error, mics 2..num_eval+1 = eval grid

    H_reference = room.rir[0][0][:max_len]
    H_primary = room.rir[1][0][:max_len]
    H_secondary = [room.rir[1][m + 1][:max_len] for m in range(num_speakers)]

    H_primary_eval = [room.rir[2 + k][0][:max_len] for k in range(num_eval)]
    H_secondary_eval = [
        [room.rir[2 + k][m + 1][:max_len] for k in range(num_eval)]
        for m in range(num_speakers)
    ]

    return H_primary, H_secondary, H_reference, H_primary_eval, H_secondary_eval


# ============================================================
# Train filters and capture per-sample anti-noise streams
# ============================================================

def train_siso(noise, H_primary, H_secondary_single, H_reference, H_secondary_est_single):
    """Run scalar FxNLMS to train. Returns (anti_noise_signal, weights)."""
    fxlms = FxNLMS(
        filter_length=FILTER_LENGTH,
        step_size=STEP_SIZE,
        secondary_path_estimate=H_secondary_est_single,
        regularization=1e-4,
    )
    primary = FIRPath(H_primary)
    secondary = FIRPath(H_secondary_single)
    reference = FIRPath(H_reference)

    n = len(noise)
    antinoise = np.zeros(n)
    for i in range(n):
        sample = noise[i]
        x = reference.filter_sample(sample)
        d = primary.filter_sample(sample)
        y = fxlms.generate_antinoise(x)
        antinoise[i] = y
        y_at_err = secondary.filter_sample(y)
        e = d + y_at_err
        fxlms.filter_reference(x)
        fxlms.update_weights(e)

    return antinoise, fxlms


def train_pseudo_mimo(noise, H_primary, H_secondary_list, H_reference, H_secondary_est_combined):
    """Pseudo-MIMO: scalar FxLMS with combined secondary path estimate."""
    fxlms = FxNLMS(
        filter_length=FILTER_LENGTH,
        step_size=STEP_SIZE,
        secondary_path_estimate=H_secondary_est_combined,
        regularization=1e-4,
    )
    primary = FIRPath(H_primary)
    secondaries = [FIRPath(s) for s in H_secondary_list]
    reference = FIRPath(H_reference)

    n = len(noise)
    antinoise = np.zeros(n)  # broadcast — same to all speakers
    for i in range(n):
        sample = noise[i]
        x = reference.filter_sample(sample)
        d = primary.filter_sample(sample)
        y = fxlms.generate_antinoise(x)
        antinoise[i] = y
        y_at_err = sum(s.filter_sample(y) for s in secondaries)
        e = d + y_at_err
        fxlms.filter_reference(x)
        fxlms.update_weights(e)

    return antinoise, fxlms


def train_true_mimo(noise, H_primary, H_secondary_list, H_reference, H_secondary_est_list):
    """True MIMO: per-speaker independent filters."""
    M = len(H_secondary_list)
    fxlms = MIMOFxNLMS(
        filter_length=FILTER_LENGTH,
        step_size=STEP_SIZE,
        secondary_path_estimates=H_secondary_est_list,
        regularization=1e-4,
    )
    primary = FIRPath(H_primary)
    secondaries = [FIRPath(s) for s in H_secondary_list]
    reference = FIRPath(H_reference)

    n = len(noise)
    antinoise_per_speaker = np.zeros((n, M))
    for i in range(n):
        sample = noise[i]
        x = reference.filter_sample(sample)
        d = primary.filter_sample(sample)
        y_per = fxlms.generate_antinoise(x)
        antinoise_per_speaker[i] = y_per
        y_at_err = sum(secondaries[m].filter_sample(y_per[m]) for m in range(M))
        e = d + y_at_err
        fxlms.filter_reference(x)
        fxlms.update_weights(e)

    return antinoise_per_speaker, fxlms


# ============================================================
# Evaluate dB attenuation at each grid point using captured anti-noise
# ============================================================

def evaluate_grid(noise, antinoise, H_primary_eval_list, H_secondary_eval_lists, num_eval):
    """
    For each evaluation grid point k:
      d_k(n) = noise convolved with H_primary_eval[k]
      a_k(n) = sum_m anti_noise_m convolved with H_secondary_eval[m][k]
      e_k(n) = d_k(n) + a_k(n)
      attenuation_k = 10*log10(power(d_k) / power(e_k))

    `antinoise` is either a 1D array (SISO/pseudo) or 2D (n, M) for MIMO.
    `H_secondary_eval_lists` is structured as: list[m][k]
    """
    if antinoise.ndim == 1:
        # SISO/pseudo: same anti-noise emitted by all speakers
        # H_secondary_eval_lists may have multiple speakers; if so, sum.
        antinoise_per_m = [antinoise]
    else:
        antinoise_per_m = [antinoise[:, m] for m in range(antinoise.shape[1])]

    M = len(antinoise_per_m)
    n_samples = len(noise)
    attenuations = np.zeros(num_eval)

    # For evaluation we use only the second half (steady state)
    half = n_samples // 2

    for k in range(num_eval):
        # Noise at eval mic k (primary path)
        d_k = np.convolve(noise, H_primary_eval_list[k], mode='same')

        # Anti-noise at eval mic k = sum of (anti_m convolved with sec_eval[m][k])
        a_k = np.zeros(n_samples)
        for m in range(M):
            sec_path = H_secondary_eval_lists[m][k] if M > 1 else H_secondary_eval_lists[0][k]
            a_k += np.convolve(antinoise_per_m[m], sec_path, mode='same')

        e_k = d_k + a_k

        d_power = np.mean(d_k[half:] ** 2)
        e_power = np.mean(e_k[half:] ** 2)

        if e_power > 1e-10 and d_power > 1e-10:
            attenuations[k] = 10 * np.log10(d_power / e_power)
        else:
            attenuations[k] = 0.0

    return attenuations


# ============================================================
# Main pipeline
# ============================================================

def main():
    print("=" * 70)
    print(" Cancellation Heatmap — SISO vs Pseudo-MIMO vs True MIMO")
    print("=" * 70)

    # Generate noise (same noise for all three to ensure fair comparison)
    print("\nLoading noise...")
    noise_gen = NoiseMixer(FS)
    noise = noise_gen.load_audio_file(AUDIO_FILE, duration=DURATION_TRAIN)
    n_samples = len(noise)
    print(f"  {len(noise)} samples ({len(noise)/FS:.1f}s)")

    # ----- Build room for SISO -----
    print("\nBuilding SISO room (1 speaker)...")
    np.random.seed(42)
    siso_room, eval_pos, y_offs, z_offs = build_full_room([SISO_SPEAKER])
    num_eval = len(eval_pos)

    H_pri_siso, H_sec_siso_list, H_ref_siso, H_pri_eval_siso, H_sec_eval_siso = extract_paths(
        siso_room, num_speakers=1, num_eval=num_eval
    )
    H_sec_est_siso = H_sec_siso_list[0] * (1 + 0.05 * np.random.randn(len(H_sec_siso_list[0])))

    print(f"  Evaluation grid: {GRID_N}×{GRID_N} = {num_eval} points over ±{GRID_HALF_M*100:.0f} cm")

    # ----- Build room for 4-speaker (used by both pseudo and true MIMO) -----
    print("\nBuilding 4-speaker room...")
    np.random.seed(42)
    spk_list = list(FOUR_SPEAKERS.values())
    multi_room, _, _, _ = build_full_room(spk_list)

    H_pri_m, H_sec_m_list, H_ref_m, H_pri_eval_m, H_sec_eval_m = extract_paths(
        multi_room, num_speakers=4, num_eval=num_eval
    )

    # Pseudo: combined secondary path estimate
    H_sec_combined = np.zeros(512)
    for s in H_sec_m_list:
        H_sec_combined[:len(s)] += s
    H_sec_est_pseudo = H_sec_combined * (1 + 0.05 * np.random.randn(len(H_sec_combined)))

    # True MIMO: per-speaker estimates
    H_sec_est_true = [
        s * (1 + 0.05 * np.random.randn(len(s))) for s in H_sec_m_list
    ]

    # ----- Train SISO -----
    print("\nTraining SISO scalar FxLMS...")
    siso_antinoise, _ = train_siso(noise, H_pri_siso, H_sec_siso_list[0], H_ref_siso, H_sec_est_siso)

    # ----- Train Pseudo-MIMO -----
    print("Training Pseudo-MIMO (broadcast)...")
    pseudo_antinoise, _ = train_pseudo_mimo(noise, H_pri_m, H_sec_m_list, H_ref_m, H_sec_est_pseudo)

    # ----- Train True MIMO -----
    print("Training True MIMO (independent)...")
    mimo_antinoise_per, _ = train_true_mimo(noise, H_pri_m, H_sec_m_list, H_ref_m, H_sec_est_true)

    # ----- Evaluate spatial attenuation grids -----
    print("\nEvaluating spatial grids...")

    # SISO has 1 speaker; we structure H_sec_eval_siso as [[paths_for_speaker_0]]
    # H_sec_eval_siso is structured as list[m][k]
    print("  SISO grid...")
    siso_atten = evaluate_grid(
        noise, siso_antinoise,
        H_pri_eval_siso,
        H_sec_eval_siso,  # only 1 speaker
        num_eval
    )

    print("  Pseudo-MIMO grid...")
    # Pseudo-MIMO: same anti-noise broadcast to all speakers; antinoise is 1D
    # H_sec_eval_m structured as list[m][k] for 4 speakers
    # We need to sum across all speakers when evaluating
    # For pseudo-MIMO, antinoise is broadcast — we treat it as if each speaker emits the same signal
    # So H_sec_eval_lists effectively becomes [[sum_paths_for_each_k]]
    H_sec_eval_pseudo_combined = [[
        sum(H_sec_eval_m[m][k] for m in range(4)) for k in range(num_eval)
    ]]
    pseudo_atten = evaluate_grid(
        noise, pseudo_antinoise,
        H_pri_eval_m,
        H_sec_eval_pseudo_combined,
        num_eval
    )

    print("  True MIMO grid...")
    mimo_atten = evaluate_grid(
        noise, mimo_antinoise_per,
        H_pri_eval_m,
        H_sec_eval_m,  # per-speaker list[m][k]
        num_eval
    )

    # ----- Plot heatmaps -----
    print("\nGenerating heatmaps...")
    siso_grid = siso_atten.reshape(GRID_N, GRID_N)
    pseudo_grid = pseudo_atten.reshape(GRID_N, GRID_N)
    mimo_grid = mimo_atten.reshape(GRID_N, GRID_N)

    grids = [siso_grid, pseudo_grid, mimo_grid]
    titles = ['SISO\n(1 speaker, scalar FxLMS)',
              'Pseudo-MIMO\n(4 speakers broadcast)',
              'True MIMO\n(4 speakers independent)']

    # Symmetric color scale around 0 so amplification (red) is visible
    vmax = max(np.max(np.abs(g)) for g in grids)
    vmax = max(vmax, 5)  # at least ±5 dB scale

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    extent = [y_offs[0] * 100, y_offs[-1] * 100, z_offs[0] * 100, z_offs[-1] * 100]

    for ax, grid, title in zip(axes, grids, titles):
        im = ax.imshow(grid, origin='lower', extent=extent,
                       cmap='RdBu', vmin=-vmax, vmax=vmax,
                       interpolation='bilinear', aspect='equal')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Y offset from error mic (cm)')
        ax.set_ylabel('Z offset from error mic (cm)')
        # Mark the original error mic position (0,0) center
        ax.plot(0, 0, 'k+', markersize=14, mew=2)
        ax.text(0, -0.5, 'error\nmic', ha='center', va='top', fontsize=8, color='black')

        # Annotate min/max
        max_db = np.max(grid)
        min_db = np.min(grid)
        ax.text(0.02, 0.98, f"max: {max_db:+.1f} dB\nmin: {min_db:+.1f} dB",
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label('Noise reduction (dB)\nblue = cancellation, red = amplification',
                   fontsize=10)

    fig.suptitle('Spatial Cancellation Patterns — 30×30 cm Head Zone\n'
                 f'(trained on {AUDIO_FILE.split("/")[-1]}, frozen filter, 11×11 eval grid)',
                 fontsize=12, fontweight='bold', y=1.03)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved heatmap to: {OUTPUT_PATH}")
    plt.close()

    # ----- Print summary stats -----
    print("\nSpatial summary (over the 30×30 cm head zone):")
    print(f"{'Algorithm':<20} | {'Mean dB':>8} | {'Max dB':>8} | {'Min dB':>8} | "
          f"{'%Negative':>9}")
    print("-" * 70)
    for label, grid in zip(['SISO', 'Pseudo-MIMO', 'True MIMO'], grids):
        mean_db = np.mean(grid)
        max_db = np.max(grid)
        min_db = np.min(grid)
        pct_neg = 100 * np.sum(grid < 0) / grid.size
        print(f"{label:<20} | {mean_db:>+7.2f} | {max_db:>+7.2f} | "
              f"{min_db:>+7.2f} | {pct_neg:>7.1f}%")


if __name__ == '__main__':
    main()
