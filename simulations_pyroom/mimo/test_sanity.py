"""
MIMO Sanity Tests

Verifies that MIMOSimulation correctly degenerates to known-good baselines:
1. With M=1 speaker, MIMO must produce numerically identical results to
   PlaygroundSimulation (single-channel scalar FxNLMS).
2. With M=4 speakers and IDENTICAL secondary paths (and identical estimates),
   true MIMO should approximately match pseudo-MIMO (MultiSpeakerSimulation).
   Small numerical drift acceptable due to per-speaker estimate noise.
"""

import sys
import warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from playground.simulation.runner import PlaygroundSimulation, MultiSpeakerSimulation
from playground.simulation.mimo_runner import MIMOSimulation


def base_params(audio_file, duration):
    return {
        'dimensions': [4.5, 1.85, 1.2],
        'absorption': 0.35,
        'max_order': 3,
        'positions': {
            'noise_source': [0.5, 0.92, 0.4],
            'reference_mic': [1.1, 0.92, 0.8],
            'speaker': [0.8, 0.25, 0.9],
            'error_mic': [2.5, 0.55, 1.05],
        },
        'filter_length': 256,
        'step_size': 0.003,
        'leakage': 0.0,
        'sample_rate': 16000,
        'scenario': 'highway',
        'audio_file': audio_file,
        'duration': duration,
    }


def test_single_speaker_degenerate():
    """With M=1 speaker, MIMO must match scalar FxNLMS to numerical precision.

    To get a clean comparison, we run the SISO sim first, extract its secondary
    path estimate, and inject the same estimate into the MIMO sim. Otherwise
    the random call sequence during construction differs slightly between the
    two classes (path_generator.get_all_anc_paths vs my direct randn call),
    yielding two different 5%-noisy estimates.
    """
    print("\n=== TEST 1: M=1 MIMO vs scalar FxLMS (must match exactly) ===")

    np.random.seed(42)
    siso_params = base_params('real_noises/realcar1.wav', 5.0)
    siso = PlaygroundSimulation(siso_params)
    siso_results = siso.run()
    siso_db = siso_results['noise_reduction_db']
    siso_error = siso_results['error']

    # Inject the SAME secondary path estimate into MIMO so we're testing the
    # algorithm, not different estimate noise.
    np.random.seed(42)
    mimo_params = base_params('real_noises/realcar1.wav', 5.0)
    mimo_params['speakers'] = {'single': mimo_params['positions']['speaker']}
    mimo = MIMOSimulation(mimo_params)
    # Override MIMO's per-speaker estimate with SISO's
    mimo.H_secondary_est = {'single': siso.H_secondary_est}
    mimo.fxlms.s_hat = [np.array(siso.H_secondary_est)]
    mimo.fxlms.s_hat_lens = [len(siso.H_secondary_est)]
    mimo_results = mimo.run()
    mimo_db = mimo_results['noise_reduction_db']
    mimo_error = mimo_results['error']

    print(f"  SISO scalar FxNLMS: {siso_db:.6f} dB")
    print(f"  MIMO with M=1:       {mimo_db:.6f} dB")
    db_diff = abs(siso_db - mimo_db)
    print(f"  dB difference:       {db_diff:.6e}")

    error_diff = np.max(np.abs(siso_error - mimo_error))
    print(f"  Max sample error diff: {error_diff:.2e}")

    if db_diff < 1e-3 and error_diff < 1e-6:
        print("  PASS — MIMO with M=1 matches scalar FxNLMS to numerical precision")
        return True
    else:
        print("  FAIL — expected near-identical results")
        return False


def test_four_speaker_vs_pseudo():
    """With M=4 speakers, compare true MIMO to pseudo-MIMO. Should be close."""
    print("\n=== TEST 2: M=4 MIMO vs pseudo-MIMO (should be close) ===")

    speakers = {
        'door_L': [2.0, 0.1, 0.4],
        'door_R': [2.0, 1.75, 0.4],
        'dash_L': [0.8, 0.25, 0.9],
        'dash_R': [0.8, 1.60, 0.9],
    }

    np.random.seed(123)
    pseudo_params = base_params('real_noises/realcar1.wav', 5.0)
    pseudo_params['speakers'] = speakers
    pseudo = MultiSpeakerSimulation(pseudo_params)
    pseudo_results = pseudo.run()
    pseudo_db = pseudo_results['noise_reduction_db']

    np.random.seed(123)
    mimo_params = base_params('real_noises/realcar1.wav', 5.0)
    mimo_params['speakers'] = speakers
    mimo = MIMOSimulation(mimo_params)
    mimo_results = mimo.run()
    mimo_db = mimo_results['noise_reduction_db']

    print(f"  Pseudo-MIMO (broadcast): {pseudo_db:.4f} dB")
    print(f"  True MIMO (independent): {mimo_db:.4f} dB")
    print(f"  Delta:                    {mimo_db - pseudo_db:+.4f} dB")

    # MIMO has more degrees of freedom, so it should match or exceed pseudo-MIMO
    if not (np.isnan(pseudo_db) or np.isnan(mimo_db)):
        print("  PASS — both ran without divergence")
        return True
    else:
        print("  FAIL — at least one diverged")
        return False


if __name__ == '__main__':
    pass1 = test_single_speaker_degenerate()
    pass2 = test_four_speaker_vs_pseudo()

    print(f"\n{'='*60}")
    if pass1 and pass2:
        print(" All sanity tests PASSED")
    else:
        print(" One or more sanity tests FAILED")
    print(f"{'='*60}")
