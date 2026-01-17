"""
Step 6: FxLMS Algorithm with Realistic Acoustic Paths

Goal: Implement FxLMS-based ANC using pyroomacoustics for realistic
      room impulse responses.

This simulation demonstrates:
1. FxLMS adaptive algorithm with pyroomacoustics-generated RIRs
2. The algorithm learns to compensate for secondary path delay
3. Convergence behavior with realistic acoustic paths
4. Effect of room acoustics on adaptation performance

Runs 3 different configurations:
- Config A: Reverberant room (challenging - lots of reflections)
- Config B: Typical room (moderate absorption)
- Config C: Damped room (easier - less reverb)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyroomacoustics as pra
from src.acoustic.room_builder import RoomBuilder
from src.acoustic.path_generator import AcousticPathGenerator, FIRPath
from src.core.fxlms import FxLMS, FxNLMS
from src.utils.audio import save_wav, save_comparison_wav
from configurations import STEP6_CONFIGS, generate_noise_signal, print_config_summary


class PyroomANCSystem:
    """
    Complete ANC system using pyroomacoustics for acoustic modeling.
    Uses FxNLMS for stable convergence across different room conditions.
    """

    def __init__(
        self,
        room: pra.ShoeBox,
        path_gen: AcousticPathGenerator,
        filter_length: int,
        step_size: float,
        secondary_path_error: float = 0.05
    ):
        self.room = room
        self.fs = room.fs
        self.path_gen = path_gen

        # Get acoustic paths
        paths = path_gen.get_all_anc_paths(modeling_error=secondary_path_error)

        self.H_primary = paths['primary']
        self.H_secondary = paths['secondary']
        self.H_secondary_est = paths['secondary_estimate']
        self.H_reference = paths['reference']

        # Truncate paths for efficiency
        max_path_len = 512
        self.H_primary = self.H_primary[:max_path_len]
        self.H_secondary = self.H_secondary[:max_path_len]
        self.H_secondary_est = self.H_secondary_est[:max_path_len]
        self.H_reference = self.H_reference[:max_path_len]

        # Create FIR filters for paths
        self.primary_path = FIRPath(self.H_primary)
        self.secondary_path = FIRPath(self.H_secondary)
        self.reference_path = FIRPath(self.H_reference)

        # Create normalized FxLMS for stability
        self.fxlms = FxNLMS(
            filter_length=filter_length,
            step_size=step_size,
            secondary_path_estimate=self.H_secondary_est,
            regularization=1e-4  # Small regularization for better adaptation
        )

        # Results storage
        self.results = {
            'reference': [],
            'desired': [],
            'antinoise': [],
            'error': [],
            'mse': []
        }

    def process_sample(self, noise_sample: float) -> float:
        """Process one sample through the ANC system."""
        # Reference signal (noise through reference path)
        x = self.reference_path.filter_sample(noise_sample)
        self.results['reference'].append(x)

        # Noise at error mic (through primary path) = d(n)
        d = self.primary_path.filter_sample(noise_sample)
        self.results['desired'].append(d)

        # Generate anti-noise through adaptive filter
        y = self.fxlms.generate_antinoise(x)
        self.results['antinoise'].append(y)

        # Anti-noise through secondary path
        y_at_error = self.secondary_path.filter_sample(y)

        # Error signal: e(n) = d(n) + y'(n)
        e = d + y_at_error
        self.results['error'].append(e)
        self.results['mse'].append(e ** 2)

        # Update FxLMS weights
        self.fxlms.filter_reference(x)
        self.fxlms.update_weights(e)

        return e

    def run(self, noise_signal: np.ndarray, verbose: bool = True) -> dict:
        """Run complete simulation."""
        n_samples = len(noise_signal)

        for i in range(n_samples):
            self.process_sample(noise_signal[i])

            if verbose and (i + 1) % (n_samples // 10) == 0:
                progress = (i + 1) / n_samples * 100
                mse = np.mean(self.results['mse'][-1000:]) if len(self.results['mse']) > 1000 else np.mean(self.results['mse'])
                print(f"  Progress: {progress:.0f}% | MSE: {mse:.6f}")

        return {k: np.array(v) for k, v in self.results.items()}

    def get_noise_reduction_db(self, window: int = None) -> float:
        """Calculate noise reduction in dB."""
        if window is None:
            window = len(self.results['desired']) // 2

        d_power = np.mean(np.array(self.results['desired'][-window:])**2)
        e_power = np.mean(np.array(self.results['error'][-window:])**2)

        if e_power < 1e-10:
            return 60.0

        return 10 * np.log10(d_power / e_power)

    def reset(self):
        """Reset system state."""
        self.fxlms.reset()
        self.primary_path.reset()
        self.secondary_path.reset()
        self.reference_path.reset()
        self.results = {k: [] for k in self.results}


def run_fxlms_simulation(config: dict, fs: int = 16000) -> dict:
    """
    Run FxLMS simulation for a given configuration.
    """
    room_cfg = config['room']
    pos = config['positions']
    fxlms_cfg = config['fxlms']

    duration = fxlms_cfg['duration']
    n_samples = int(duration * fs)

    # Create room
    room = RoomBuilder.simple_room(
        room_cfg['dimensions'],
        fs,
        absorption=room_cfg['absorption'],
        max_order=room_cfg['max_order']
    )

    # Add sources and microphones
    room.add_source(pos['noise_source'])   # Source 0: Noise
    room.add_source(pos['speaker'])        # Source 1: Speaker
    mic_array = np.array([pos['reference_mic'], pos['error_mic']]).T
    room.add_microphone_array(pra.MicrophoneArray(mic_array, fs=fs))
    room.compute_rir()

    # Create path generator
    path_gen = AcousticPathGenerator(room)

    # Create ANC system
    anc = PyroomANCSystem(
        room=room,
        path_gen=path_gen,
        filter_length=fxlms_cfg['filter_length'],
        step_size=fxlms_cfg['step_size'],
        secondary_path_error=0.05
    )

    print(f"  Primary path: {len(anc.H_primary)} taps")
    print(f"  Secondary path: {len(anc.H_secondary)} taps")
    print(f"  Filter length: {fxlms_cfg['filter_length']}")
    print(f"  Step size: {fxlms_cfg['step_size']}")

    # Generate noise
    noise_signal = generate_noise_signal(config['noise'], duration, fs)

    # Run simulation
    print("\nRunning FxNLMS simulation...")
    results = anc.run(noise_signal)

    # Calculate metrics
    reduction_db = anc.get_noise_reduction_db()

    return {
        'config_name': config['name'],
        'results': results,
        'reduction_db': reduction_db,
        'weights': anc.fxlms.get_weights(),
        'fs': fs,
        'duration': duration,
    }


def plot_fxlms_results(sim_results: dict, config: dict, save_path: str):
    """Generate FxLMS result plots."""
    results = sim_results['results']
    fs = sim_results['fs']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Time domain (last 100ms)
    show_samples = int(0.1 * fs)
    t_show = np.arange(show_samples) / fs * 1000

    axes[0, 0].plot(t_show, results['desired'][-show_samples:], 'r-', linewidth=1, alpha=0.7, label='Noise')
    axes[0, 0].plot(t_show, results['error'][-show_samples:], 'g-', linewidth=1, alpha=0.7, label='With ANC')
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title(f"{config['name']}: Before vs After ANC")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # MSE convergence
    window = 200
    mse = results['mse']
    if len(mse) > window:
        mse_smooth = np.convolve(mse, np.ones(window)/window, mode='valid')
        t_mse = np.arange(len(mse_smooth)) / fs
        axes[0, 1].semilogy(t_mse, mse_smooth)
    else:
        axes[0, 1].semilogy(np.arange(len(mse)) / fs, mse)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].set_title(f'Convergence ({sim_results["reduction_db"]:.1f} dB reduction)')
    axes[0, 1].grid(True, alpha=0.3)

    # Frequency spectrum
    from scipy import signal as scipy_signal
    steady_start = len(results['desired']) // 2
    f_d, psd_d = scipy_signal.welch(results['desired'][steady_start:], fs, nperseg=1024)
    f_e, psd_e = scipy_signal.welch(results['error'][steady_start:], fs, nperseg=1024)

    axes[1, 0].semilogy(f_d, psd_d, 'r-', linewidth=2, label='Noise')
    axes[1, 0].semilogy(f_e, psd_e, 'g-', linewidth=2, label='With ANC')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('PSD')
    axes[1, 0].set_title('Spectrum (20-300 Hz target)')
    axes[1, 0].set_xlim(0, 350)
    axes[1, 0].axvline(x=20, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(x=300, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Filter weights
    axes[1, 1].plot(sim_results['weights'], 'b-', linewidth=1)
    axes[1, 1].set_xlabel('Tap Index')
    axes[1, 1].set_ylabel('Weight')
    axes[1, 1].set_title('Learned Filter Coefficients')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    """
    Run FxLMS simulations for all configurations.
    """
    print("=" * 70)
    print("Step 6: FxLMS Algorithm with Realistic Acoustic Paths")
    print("=" * 70)
    print()

    # Ensure output directories exist
    os.makedirs('output/plots', exist_ok=True)
    os.makedirs('output/audio', exist_ok=True)

    fs = 16000
    all_results = {}

    # Run all configurations
    for config_key, config in STEP6_CONFIGS.items():
        print_config_summary(config, f"Step 6 - {config_key}")

        sim_results = run_fxlms_simulation(config, fs)
        all_results[config_key] = sim_results

        print(f"\nNoise Reduction: {sim_results['reduction_db']:.1f} dB")

        # Generate plots
        plot_fxlms_results(sim_results, config, f'output/plots/pyroom_step6_{config_key}.png')
        print(f"Saved: output/plots/pyroom_step6_{config_key}.png")

        # Save audio
        results = sim_results['results']
        save_comparison_wav(
            f'pyroom_step6_{config_key}',
            results['desired'],
            results['error'],
            fs,
            'output/audio'
        )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - FxLMS Performance")
    print("=" * 70)
    print()
    print(f"{'Configuration':<25} {'Absorption':>12} {'Reflections':>12} {'Reduction':>12}")
    print("-" * 65)

    for config_key, sim_results in all_results.items():
        config = STEP6_CONFIGS[config_key]
        room = config['room']
        print(f"{config['name']:<25} {room['absorption']:>10.1%} {room['max_order']:>10} {sim_results['reduction_db']:>10.1f} dB")

    print()
    print("=" * 70)
    print("KEY OBSERVATIONS - FxLMS SUCCESS")
    print("=" * 70)
    print()
    print("1. FxLMS ADAPTS TO ROOM ACOUSTICS:")
    print("   - Algorithm learns optimal filter for each room")
    print("   - Compensates for reflections and reverb")
    print()
    print("2. ROOM CONDITIONS AFFECT PERFORMANCE:")
    print("   - Reverberant rooms: slower convergence, may need longer filters")
    print("   - Damped rooms: faster convergence, shorter filters sufficient")
    print()
    print("3. NORMALIZED STEP SIZE (FxNLMS):")
    print("   - Provides stable convergence across signal levels")
    print("   - More robust than standard FxLMS")
    print()
    print("4. PRACTICAL CONSIDERATIONS:")
    print("   - Filter length must capture room impulse response")
    print("   - Step size trades off speed vs stability")
    print()
    print("NEXT: Apply to car interior scenario in Step 7")
    print()

    return all_results


if __name__ == '__main__':
    main()
