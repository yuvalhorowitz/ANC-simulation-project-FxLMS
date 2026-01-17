"""
Step 5: ANC with Latency Problem

Goal: Demonstrate WHY simple inversion fails when there's
      processing delay in the secondary path.

This simulation demonstrates:
1. The secondary path includes processing delay (ADC, computation, DAC)
2. If we just invert the reference signal, the anti-noise arrives LATE
3. Late anti-noise can actually AMPLIFY the noise!
4. This motivates the need for FxLMS adaptive algorithm

Runs 3 different configurations:
- Config A: Headphone scenario (tight timing ~2ms budget)
- Config B: Desktop system (medium timing ~6ms budget)
- Config C: Large room (relaxed timing ~15ms budget)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyroomacoustics as pra
from src.acoustic.room_builder import RoomBuilder, calculate_distance
from src.utils.audio import save_wav
from configurations import STEP5_CONFIGS, generate_noise_signal, print_config_summary


def run_latency_test(config: dict, fs: int = 16000, duration: float = 2.0) -> dict:
    """
    Run latency test for a given configuration.

    Args:
        config: Configuration dictionary
        fs: Sampling frequency
        duration: Simulation duration in seconds

    Returns:
        Results dictionary with signals and metrics for different latencies
    """
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs

    # Extract configuration
    room_cfg = config['room']
    pos = config['positions']

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

    # Extract RIRs
    rir_reference = room.rir[0][0]   # Noise -> Reference mic
    rir_primary = room.rir[1][0]     # Noise -> Error mic
    rir_secondary = room.rir[1][1]   # Speaker -> Error mic

    # Analyze path delays
    ref_delay = np.argmax(np.abs(rir_reference))
    primary_delay = np.argmax(np.abs(rir_primary))
    secondary_delay = np.argmax(np.abs(rir_secondary))

    # Time budget: time between noise at ref mic and noise at error mic,
    # minus time for anti-noise to travel from speaker to error mic
    time_budget_samples = primary_delay - ref_delay - secondary_delay
    time_budget_ms = time_budget_samples / fs * 1000

    # Generate noise signal
    noise_signal = generate_noise_signal(config['noise'], duration, fs)

    # Reference signal at ref mic
    reference = np.convolve(noise_signal, rir_reference, mode='full')[:n_samples]

    # Noise at error mic (without ANC)
    noise_at_error = np.convolve(noise_signal, rir_primary, mode='full')[:n_samples]
    noise_power = np.mean(noise_at_error[fs:] ** 2)  # Skip first second

    # Test different processing latencies
    test_latencies_ms = config.get('test_latencies_ms', [0, 0.5, 1.0, 2.0, 5.0])
    latency_results = {}

    for latency_ms in test_latencies_ms:
        latency_samples = int(latency_ms * fs / 1000)

        # Naive approach: delayed inverted reference
        naive_antinoise = np.zeros(n_samples)
        if latency_samples > 0 and latency_samples < n_samples:
            naive_antinoise[latency_samples:] = -reference[:-latency_samples]
        elif latency_samples == 0:
            naive_antinoise = -reference.copy()

        # Through secondary path
        naive_at_error = np.convolve(naive_antinoise, rir_secondary, mode='full')[:n_samples]

        # Total
        total = noise_at_error + naive_at_error

        # Calculate reduction
        total_power = np.mean(total[fs:] ** 2)
        if total_power > 1e-10:
            reduction = 10 * np.log10(noise_power / total_power)
        else:
            reduction = 60.0

        latency_results[latency_ms] = {
            'total': total,
            'reduction_db': reduction,
            'latency_samples': latency_samples,
        }

    return {
        'config_name': config['name'],
        'noise_at_error': noise_at_error,
        'reference': reference,
        'latency_results': latency_results,
        'time_budget_ms': time_budget_ms,
        'time_budget_samples': time_budget_samples,
        'ref_delay': ref_delay,
        'primary_delay': primary_delay,
        'secondary_delay': secondary_delay,
        't': t,
    }


def plot_latency_comparison(results: dict, config: dict, fs: int, save_path: str):
    """Plot latency comparison for a configuration."""
    latency_results = results['latency_results']
    latencies = list(latency_results.keys())
    reductions = [latency_results[l]['reduction_db'] for l in latencies]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Bar chart of reduction vs latency
    colors = ['green' if r >= 0 else 'red' for r in reductions]
    axes[0].bar(range(len(latencies)), reductions, color=colors)
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=2)
    axes[0].set_xticks(range(len(latencies)))
    axes[0].set_xticklabels([f'{l:.1f} ms' for l in latencies])
    axes[0].set_xlabel('Processing Latency')
    axes[0].set_ylabel('Noise Reduction (dB)')
    axes[0].set_title(f"{config['name']}: Naive ANC vs Processing Latency")
    axes[0].grid(True, alpha=0.3, axis='y')

    # Add time budget line
    time_budget = results['time_budget_ms']
    if time_budget > 0 and time_budget < max(latencies):
        budget_idx = np.interp(time_budget, latencies, range(len(latencies)))
        axes[0].axvline(x=budget_idx, color='orange', linestyle='--', linewidth=2,
                       label=f'Time budget ({time_budget:.1f} ms)')
        axes[0].legend()

    # Time domain comparison: 0ms vs worst case
    show_samples = int(0.03 * fs)
    start_idx = int(1.5 * fs)
    t_window = np.arange(show_samples) / fs * 1000

    axes[1].plot(t_window, results['noise_at_error'][start_idx:start_idx+show_samples],
                'r-', linewidth=2, label='Noise (no ANC)')

    # Show result with moderate latency
    mid_latency = latencies[len(latencies)//2]
    mid_result = latency_results[mid_latency]['total']
    axes[1].plot(t_window, mid_result[start_idx:start_idx+show_samples],
                'b-', linewidth=2, alpha=0.7, label=f'With {mid_latency}ms latency')

    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Phase Misalignment Due to Latency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    """
    Run latency problem simulations for all configurations.
    """
    print("=" * 70)
    print("Step 5: ANC with Latency Problem")
    print("=" * 70)
    print()

    # Ensure output directories exist
    os.makedirs('output/plots', exist_ok=True)
    os.makedirs('output/audio', exist_ok=True)

    fs = 16000
    duration = 2.0
    all_results = {}

    # Run all configurations
    for config_key, config in STEP5_CONFIGS.items():
        print_config_summary(config, f"Step 5 - {config_key}")

        results = run_latency_test(config, fs, duration)
        all_results[config_key] = results

        print(f"\nPath Analysis:")
        print(f"  Noise -> Reference mic: {results['ref_delay']} samples ({results['ref_delay']/fs*1000:.2f} ms)")
        print(f"  Noise -> Error mic: {results['primary_delay']} samples ({results['primary_delay']/fs*1000:.2f} ms)")
        print(f"  Speaker -> Error mic: {results['secondary_delay']} samples ({results['secondary_delay']/fs*1000:.2f} ms)")
        print(f"  Time budget: {results['time_budget_ms']:.2f} ms")

        print(f"\nLatency Test Results:")
        for latency_ms, lat_result in results['latency_results'].items():
            red = lat_result['reduction_db']
            status = "REDUCTION" if red >= 0 else "AMPLIFICATION!"
            print(f"  {latency_ms:.1f} ms: {red:+.1f} dB ({status})")

        # Generate plots
        plot_latency_comparison(results, config, fs, f'output/plots/pyroom_step5_{config_key}.png')
        print(f"\nSaved: output/plots/pyroom_step5_{config_key}.png")

        # Save audio for worst and best case
        latency_results = results['latency_results']
        latencies = list(latency_results.keys())

        save_wav(f'output/audio/pyroom_step5_{config_key}_no_anc.wav',
                results['noise_at_error'], fs)
        save_wav(f'output/audio/pyroom_step5_{config_key}_0ms.wav',
                latency_results[0]['total'], fs)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - Latency Problem")
    print("=" * 70)
    print()
    print(f"{'Configuration':<30} {'Time Budget':>12} {'0ms Latency':>15} {'Max Tested':>15}")
    print("-" * 75)

    for config_key, results in all_results.items():
        config = STEP5_CONFIGS[config_key]
        latency_results = results['latency_results']
        latencies = list(latency_results.keys())

        red_0ms = latency_results[0]['reduction_db']
        red_max = latency_results[max(latencies)]['reduction_db']

        print(f"{config['name']:<30} {results['time_budget_ms']:>10.1f} ms {red_0ms:>+13.1f} dB {red_max:>+13.1f} dB")

    print()
    print("=" * 70)
    print("KEY OBSERVATIONS - THE LATENCY PROBLEM")
    print("=" * 70)
    print()
    print("1. NAIVE INVERSION FAILS WITH LATENCY:")
    print("   - Simply inverting the reference signal doesn't work")
    print("   - The anti-noise arrives TOO LATE to cancel the noise")
    print()
    print("2. PHASE MISALIGNMENT:")
    print("   - Late anti-noise is out of phase with the noise")
    print("   - Instead of cancelling, it can AMPLIFY the noise!")
    print()
    print("3. TIME BUDGET MATTERS:")
    print("   - Shorter acoustic paths = tighter timing constraints")
    print("   - Headphones: very tight (<2ms)")
    print("   - Large rooms: more relaxed (>10ms)")
    print()
    print("4. THE SOLUTION - ADAPTIVE FILTERING:")
    print("   - FxLMS learns to PREDICT what anti-noise is needed")
    print("   - It compensates for the secondary path delay")
    print()
    print("NEXT: Implement FxLMS adaptive algorithm in Step 6")
    print()

    return all_results


if __name__ == '__main__':
    main()
