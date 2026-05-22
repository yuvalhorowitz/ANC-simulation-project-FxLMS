"""
Stage 2 MIMO Evaluation — Four-Way Comparison Across All Recordings

For each of the 13 real audio recordings, runs:
  1. SISO baseline
  2. Pseudo-SIMO (4-speaker broadcast)
  3. Stage 1 SIMO (4 speakers, independent filters, 1 error mic)
  4. Stage 2 SIMO+multi-error (4 speakers indep, 4 error mics around head zone)

For Stage 2 we measure noise reduction averaged across the K=4 error mics.
For Stage 1 we evaluate the same 4 head-zone points (with the K=1 trained
filter) to make a direct head-zone-coverage comparison.

Output: output/data/mimo/stage2_comparison.json
"""

import sys
import json
import warnings
from pathlib import Path
import numpy as np
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

REAL_AUDIO_FILES = [
    ('Real Car 1', 'real_noises/realcar1.wav', 30.0),
    ('Real Car 2', 'real_noises/realcar2.wav', 13.0),
    ('Real Car 3', 'real_noises/realcar3.wav', 18.0),
    ('Real Car 4', 'real_noises/realcar4.wav', 14.0),
    ('Real Car 5', 'real_noises/realcar5.wav', 25.0),
    ('LA City Start', 'real_noises/la_city_start.wav', 20.0),
    ('LA Stop & Go', 'real_noises/la_city_stop_go.wav', 20.0),
    ('LA Quiet Cruise', 'real_noises/la_quiet_cruise.wav', 20.0),
    ('LA Idle', 'real_noises/la_idle.wav', 20.0),
    ('LA Varying', 'real_noises/la_varying.wav', 20.0),
    ('LA Medium Cruise', 'real_noises/la_medium_cruise.wav', 20.0),
    ('LA Loud Low', 'real_noises/la_loud_low.wav', 20.0),
    ('LA Late Drive', 'real_noises/la_late_drive.wav', 20.0),
]

OUTPUT_PATH = Path(__file__).parent.parent.parent / 'output' / 'data' / 'mimo' / 'stage2_comparison.json'


def build_room(speaker_positions, error_mic_positions, eval_mic_positions):
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
    all_mics = [REF_MIC_POS] + list(error_mic_positions) + list(eval_mic_positions)
    room.add_microphone_array(pra.MicrophoneArray(np.array(all_mics).T, fs=FS))
    room.compute_rir()
    return room


def extract_paths(room, M, K_train, num_eval):
    L = 512
    H_ref = room.rir[0][0][:L]
    H_pri_train = [room.rir[1 + k][0][:L] for k in range(K_train)]
    H_sec_train = [
        [room.rir[1 + k][1 + m][:L] for k in range(K_train)] for m in range(M)
    ]
    H_pri_eval = [room.rir[1 + K_train + e][0][:L] for e in range(num_eval)]
    H_sec_eval = [
        [room.rir[1 + K_train + e][1 + m][:L] for e in range(num_eval)]
        for m in range(M)
    ]
    return H_ref, H_pri_train, H_sec_train, H_pri_eval, H_sec_eval


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


def evaluate_at_points(noise, antinoise, H_pri_eval_list, H_sec_eval_lists):
    """Returns array of dB attenuation, one per evaluation mic."""
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
    print("=" * 100)
    print(" Stage 2 MIMO Evaluation — 4-Way Comparison Across All Recordings")
    print(" Metric: noise reduction averaged across K=4 head-zone error mics")
    print("=" * 100)
    print(f"\n{'Recording':<18} | {'SISO':>7} | {'Pseudo':>8} | {'Stage1':>8} | "
          f"{'Stage2':>8} | {'Δ S2-Pseudo':>11} | {'Δ S2-S1':>8}")
    print("-" * 100)

    results = []

    for name, audio, duration in REAL_AUDIO_FILES:
        try:
            noise = NoiseMixer(FS).load_audio_file(audio, duration=duration)

            # SISO room (1 speaker, 1 error, K=4 eval = head zone)
            np.random.seed(42)
            siso_room = build_room([SISO_SPEAKER], [ERROR_MIC_POS], ERROR_MICS_K4)
            H_ref_s, H_pri_train_s, H_sec_train_s, H_pri_eval_s, H_sec_eval_s = extract_paths(
                siso_room, M=1, K_train=1, num_eval=4
            )
            H_sec_est_s = H_sec_train_s[0][0] * (1 + 0.05 * np.random.randn(len(H_sec_train_s[0][0])))
            siso_anti = train_siso(noise, H_pri_train_s[0], H_sec_train_s[0][0], H_ref_s, H_sec_est_s)
            siso_db_per = evaluate_at_points(noise, siso_anti, H_pri_eval_s, H_sec_eval_s)
            siso_mean = float(np.mean(siso_db_per))

            # 4-speaker room (1 error, K=4 eval)
            np.random.seed(42)
            spk_list = list(FOUR_SPEAKERS.values())
            multi_room = build_room(spk_list, [ERROR_MIC_POS], ERROR_MICS_K4)
            H_ref_m, H_pri_train_m, H_sec_train_m, H_pri_eval_m, H_sec_eval_m = extract_paths(
                multi_room, M=4, K_train=1, num_eval=4
            )

            # Pseudo
            H_sec_combined = np.zeros(512)
            for m in range(4):
                ssec = H_sec_train_m[m][0]
                H_sec_combined[:len(ssec)] += ssec
            H_sec_est_pseudo = H_sec_combined * (1 + 0.05 * np.random.randn(len(H_sec_combined)))
            pseudo_anti = train_pseudo(noise, H_pri_train_m[0],
                                       [H_sec_train_m[m][0] for m in range(4)],
                                       H_ref_m, H_sec_est_pseudo)
            pseudo_db_per = evaluate_at_points(noise, pseudo_anti, H_pri_eval_m, H_sec_eval_m)
            pseudo_mean = float(np.mean(pseudo_db_per))

            # Stage 1
            H_sec_train_s1 = [H_sec_train_m[m][0] for m in range(4)]
            H_sec_est_s1 = [s * (1 + 0.05 * np.random.randn(len(s))) for s in H_sec_train_s1]
            stage1_anti = train_stage1(noise, H_pri_train_m[0], H_sec_train_s1,
                                       H_ref_m, H_sec_est_s1)
            stage1_db_per = evaluate_at_points(noise, stage1_anti, H_pri_eval_m, H_sec_eval_m)
            stage1_mean = float(np.mean(stage1_db_per))

            # Stage 2
            np.random.seed(42)
            multi_k4_room = build_room(spk_list, ERROR_MICS_K4, ERROR_MICS_K4)
            H_ref_k4, H_pri_train_k4, H_sec_train_k4, H_pri_eval_k4, H_sec_eval_k4 = extract_paths(
                multi_k4_room, M=4, K_train=4, num_eval=4
            )
            H_sec_est_s2 = [
                [s * (1 + 0.05 * np.random.randn(len(s))) for s in row] for row in H_sec_train_k4
            ]
            stage2_anti = train_stage2(noise, H_pri_train_k4, H_sec_train_k4,
                                       H_ref_k4, H_sec_est_s2)
            stage2_db_per = evaluate_at_points(noise, stage2_anti, H_pri_eval_k4, H_sec_eval_k4)
            stage2_mean = float(np.mean(stage2_db_per))

            d_s2_pseudo = stage2_mean - pseudo_mean
            d_s2_s1 = stage2_mean - stage1_mean

            print(f"{name:<18} | {siso_mean:>+5.2f}dB | {pseudo_mean:>+6.2f}dB | "
                  f"{stage1_mean:>+6.2f}dB | {stage2_mean:>+6.2f}dB | "
                  f"{d_s2_pseudo:>+9.2f}dB | {d_s2_s1:>+6.2f}dB", flush=True)

            results.append({
                'recording': name,
                'siso_mean_head_zone_db': siso_mean,
                'pseudo_mean_head_zone_db': pseudo_mean,
                'stage1_mean_head_zone_db': stage1_mean,
                'stage2_mean_head_zone_db': stage2_mean,
                'siso_per_mic_db': siso_db_per.tolist(),
                'pseudo_per_mic_db': pseudo_db_per.tolist(),
                'stage1_per_mic_db': stage1_db_per.tolist(),
                'stage2_per_mic_db': stage2_db_per.tolist(),
                'delta_s2_vs_pseudo': d_s2_pseudo,
                'delta_s2_vs_s1': d_s2_s1,
            })
        except Exception as ex:
            print(f"{name:<18} | ERROR: {ex}", flush=True)

    # Summary
    if results:
        s_arr = [r['siso_mean_head_zone_db'] for r in results]
        p_arr = [r['pseudo_mean_head_zone_db'] for r in results]
        s1_arr = [r['stage1_mean_head_zone_db'] for r in results]
        s2_arr = [r['stage2_mean_head_zone_db'] for r in results]
        d_pseudo = [r['delta_s2_vs_pseudo'] for r in results]
        d_s1 = [r['delta_s2_vs_s1'] for r in results]

        print("-" * 100)
        print(f"{'MEAN':<18} | {np.mean(s_arr):>+5.2f}dB | {np.mean(p_arr):>+6.2f}dB | "
              f"{np.mean(s1_arr):>+6.2f}dB | {np.mean(s2_arr):>+6.2f}dB | "
              f"{np.mean(d_pseudo):>+9.2f}dB | {np.mean(d_s1):>+6.2f}dB")

        wins_pseudo = sum(1 for d in d_pseudo if d > 0)
        wins_s1 = sum(1 for d in d_s1 if d > 0)
        print(f"\nStage 2 wins vs Pseudo-SIMO: {wins_pseudo}/{len(d_pseudo)}")
        print(f"Stage 2 wins vs Stage 1:     {wins_s1}/{len(d_s1)}")

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, 'w') as f:
            json.dump({
                'metric': 'noise reduction averaged across K=4 head-zone error mics',
                'results': results,
                'summary': {
                    'mean_siso_db': float(np.mean(s_arr)),
                    'mean_pseudo_db': float(np.mean(p_arr)),
                    'mean_stage1_db': float(np.mean(s1_arr)),
                    'mean_stage2_db': float(np.mean(s2_arr)),
                    'mean_delta_s2_vs_pseudo': float(np.mean(d_pseudo)),
                    'mean_delta_s2_vs_s1': float(np.mean(d_s1)),
                    'wins_s2_vs_pseudo': wins_pseudo,
                    'wins_s2_vs_s1': wins_s1,
                    'total_stable': len(results),
                },
            }, f, indent=2)
        print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
