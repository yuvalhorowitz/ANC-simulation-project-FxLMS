"""
Plot: Noise reduction (dB) vs speaker distance from driver's ear.

Shows how ANC performance degrades as the cancellation speaker moves
further from the error microphone (driver's ear position).

Requires running simulations — takes ~5 minutes for 4 samples x 12 positions.

Output: output/plots/speaker_distance_vs_reduction.png
"""

import sys, warnings, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from playground.simulation.runner import PlaygroundSimulation
import matplotlib.pyplot as plt

noise_src = [0.5, 0.92, 0.4]
error_mic = [2.5, 0.55, 1.05]
fixed_ref = [0.7, 0.92, 0.5]  # near noise (optimal)

base_params = {
    'dimensions': [4.5, 1.85, 1.2],
    'absorption': 0.35,
    'max_order': 3,
    'filter_length': 512,
    'step_size': 0.003,
    'leakage': 0.0,
    'sample_rate': 16000,
    'scenario': 'highway',
}

# Speaker positions at varying distances from error mic
speaker_configs = [
    [2.4, 0.55, 1.05],   # ~0.10m
    [2.3, 0.55, 1.1],    # ~0.21m
    [2.2, 0.55, 1.0],    # ~0.31m
    [2.0, 0.55, 1.0],    # ~0.50m
    [2.0, 0.05, 0.9],    # ~0.72m (door panel)
    [1.8, 0.55, 1.0],    # ~0.70m
    [1.5, 0.55, 1.0],    # ~1.00m (dash close)
    [1.2, 0.55, 1.0],    # ~1.30m
    [1.0, 0.55, 0.9],    # ~1.51m
    [0.8, 0.25, 0.9],    # ~1.73m (default)
    [0.5, 0.50, 0.9],    # ~2.01m
    [0.3, 0.92, 0.9],    # ~2.24m
]

# Compute distances and sort
speaker_data = []
for pos in speaker_configs:
    dist = np.sqrt(sum((a-b)**2 for a, b in zip(pos, error_mic)))
    speaker_data.append((pos, dist))
speaker_data.sort(key=lambda x: x[1])

test_samples = [
    ('LA Loud Low-Freq', 'real_noises/la_loud_low.wav', 20.0),
    ('Real Car 4', 'real_noises/realcar4.wav', 14.0),
    ('LA City Start', 'real_noises/la_city_start.wav', 20.0),
    ('Real Car 1', 'real_noises/realcar1.wav', 30.0),
]

# Run simulations
results = {name: [] for name, _, _ in test_samples}
distances = []

for spk_pos, dist in speaker_data:
    distances.append(dist)
    for sample_name, audio_path, duration in test_samples:
        params = base_params.copy()
        params['audio_file'] = audio_path
        params['duration'] = duration
        params['positions'] = {
            'noise_source': noise_src,
            'reference_mic': fixed_ref,
            'speaker': spk_pos,
            'error_mic': error_mic,
        }

        try:
            sim = PlaygroundSimulation(params)
            res = sim.run()
            if np.any(np.isnan(res['mse'])):
                results[sample_name].append(np.nan)
            else:
                results[sample_name].append(res['noise_reduction_db'])
        except:
            results[sample_name].append(np.nan)

    print(f"  dist={dist:.2f}m done", flush=True)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

colors = ['#e63946', '#457b9d', '#2a9d8f', '#e9c46a']
markers = ['o', 's', '^', 'D']

for i, (sample_name, _, _) in enumerate(test_samples):
    dbs = results[sample_name]
    valid = [(d, db) for d, db in zip(distances, dbs) if not np.isnan(db)]
    if valid:
        d_vals, db_vals = zip(*valid)
        ax.plot(d_vals, db_vals, marker=markers[i], color=colors[i],
                linewidth=2, markersize=8, label=sample_name, alpha=0.85)

ax.set_xlabel('Distance: Speaker → Driver Ear (m)', fontsize=12)
ax.set_ylabel('Noise Reduction (dB)', fontsize=12)
ax.set_title('ANC Performance vs Speaker Distance from Driver', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

ax.annotate('Closer speaker = shorter secondary path\n= better phase alignment at error mic',
           xy=(0.3, max(max(r) for r in results.values() if r) * 0.4),
           fontsize=9, style='italic', color='gray')

ax.set_xlim(0, 2.4)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('output/plots/speaker_distance_vs_reduction.png', dpi=150, bbox_inches='tight')
print("\nSaved to output/plots/speaker_distance_vs_reduction.png")
plt.close()
