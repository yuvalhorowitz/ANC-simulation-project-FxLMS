"""
Plot: Car interior layout showing ANC performance vs component positions.

Generates a top-down view of the car with reference mic and speaker positions
color-coded by noise reduction performance (green=good, red=bad).

Output: output/plots/position_optimization.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Room dimensions: [4.5, 1.85, 1.2] (length x width x height)
noise_src = [0.5, 0.92, 0.4]
error_mic = [2.5, 0.55, 1.05]

# Reference mic positions tested (dB from LA Loud Low-Freq sample)
ref_positions = {
    'Co-located\n(5.9 dB)': ([0.5, 0.92], 5.93),
    'Very near\n(9.4 dB)': ([0.6, 0.92], 9.44),
    'Near\n(7.9 dB)': ([0.7, 0.92], 7.91),
    'Front cabin\n(3.4 dB)': ([1.0, 0.92], 3.36),
    'Default\n(4.3 dB)': ([1.1, 0.92], 4.30),
    'Mid cabin\n(4.2 dB)': ([1.5, 0.92], 4.18),
    'Near error\n(2.2 dB)': ([2.0, 0.55], 2.22),
}

# Speaker positions tested (dB from LA Loud Low-Freq sample)
spk_positions = {
    'Far front\n(4.1 dB)': ([0.5, 0.50], 4.13),
    'Default\n(2.5 dB)': ([0.8, 0.25], 2.53),
    'Mid dash\n(4.6 dB)': ([1.2, 0.55], 4.58),
    'Dash close\n(7.7 dB)': ([1.5, 0.55], 7.74),
    'Door panel\n(6.8 dB)': ([2.0, 0.05], 6.83),
    'Headrest\n(7.7 dB)': ([2.3, 0.55], 7.69),
    'Ceiling\n(6.8 dB)': ([2.2, 0.55], 6.79),
}

# ===== LEFT PLOT: Reference Mic positions =====
ax = axes[0]
ax.set_xlim(-0.3, 4.8)
ax.set_ylim(-0.3, 2.15)
ax.set_aspect('equal')
ax.set_title('Reference Mic Position vs Performance\n(Top-Down View, Speaker fixed at dash_close)', fontsize=11, fontweight='bold')
ax.set_xlabel('Length (m) — Front → Rear')
ax.set_ylabel('Width (m)')

car = plt.Rectangle((0, 0), 4.5, 1.85, fill=False, edgecolor='black', linewidth=2)
ax.add_patch(car)

ax.add_patch(plt.Rectangle((2.0, 0.2), 0.6, 0.7, fill=True, facecolor='#d4a574', alpha=0.3, edgecolor='gray'))
ax.text(2.3, 0.55, 'Driver\nSeat', ha='center', va='center', fontsize=7, color='gray')
ax.add_patch(plt.Rectangle((2.0, 1.0), 0.6, 0.7, fill=True, facecolor='#d4a574', alpha=0.3, edgecolor='gray'))
ax.text(2.3, 1.35, 'Passenger\nSeat', ha='center', va='center', fontsize=7, color='gray')
ax.add_patch(plt.Rectangle((0.0, 0.0), 0.9, 1.85, fill=True, facecolor='#c0c0c0', alpha=0.15, edgecolor='gray', linestyle='--'))
ax.text(0.45, 1.75, 'Dashboard/Engine', ha='center', va='center', fontsize=7, color='gray')

ax.scatter(*noise_src[:2], c='red', s=200, marker='*', zorder=10)
ax.annotate('NOISE\nSOURCE', noise_src[:2], textcoords="offset points", xytext=(0, 15),
           ha='center', fontsize=8, color='red', fontweight='bold')

ax.scatter(*error_mic[:2], c='black', s=150, marker='X', zorder=10)
ax.annotate('ERROR MIC\n(Driver ear)', error_mic[:2], textcoords="offset points", xytext=(15, 10),
           ha='center', fontsize=8, color='black', fontweight='bold')

ax.scatter(1.5, 0.55, c='blue', s=100, marker='s', zorder=8, alpha=0.5)
ax.annotate('Speaker\n(fixed)', (1.5, 0.55), textcoords="offset points", xytext=(0, -20),
           ha='center', fontsize=7, color='blue', alpha=0.7)

max_db = max(v[1] for v in ref_positions.values())
min_db = min(v[1] for v in ref_positions.values())

for label, (pos, db) in ref_positions.items():
    norm = (db - min_db) / (max_db - min_db) if max_db > min_db else 0.5
    color = plt.cm.RdYlGn(norm)
    size = 80 + norm * 120
    ax.scatter(*pos, c=[color], s=size, marker='o', zorder=9, edgecolors='black', linewidth=0.5)
    ax.annotate(label, pos, textcoords="offset points", xytext=(0, -18),
               ha='center', fontsize=7, color='darkgreen' if norm > 0.6 else 'darkred')

ax.annotate('', xy=(0.55, 0.45), xytext=(1.8, 0.45),
           arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.text(1.1, 0.35, 'Closer to noise = Better', ha='center', fontsize=8, color='green', style='italic')

# ===== RIGHT PLOT: Speaker positions =====
ax = axes[1]
ax.set_xlim(-0.3, 4.8)
ax.set_ylim(-0.3, 2.15)
ax.set_aspect('equal')
ax.set_title('Speaker Position vs Performance\n(Top-Down View, Ref mic fixed near noise)', fontsize=11, fontweight='bold')
ax.set_xlabel('Length (m) — Front → Rear')
ax.set_ylabel('Width (m)')

car = plt.Rectangle((0, 0), 4.5, 1.85, fill=False, edgecolor='black', linewidth=2)
ax.add_patch(car)

ax.add_patch(plt.Rectangle((2.0, 0.2), 0.6, 0.7, fill=True, facecolor='#d4a574', alpha=0.3, edgecolor='gray'))
ax.text(2.3, 0.55, 'Driver\nSeat', ha='center', va='center', fontsize=7, color='gray')
ax.add_patch(plt.Rectangle((2.0, 1.0), 0.6, 0.7, fill=True, facecolor='#d4a574', alpha=0.3, edgecolor='gray'))
ax.text(2.3, 1.35, 'Passenger\nSeat', ha='center', va='center', fontsize=7, color='gray')
ax.add_patch(plt.Rectangle((0.0, 0.0), 0.9, 1.85, fill=True, facecolor='#c0c0c0', alpha=0.15, edgecolor='gray', linestyle='--'))
ax.text(0.45, 1.75, 'Dashboard/Engine', ha='center', va='center', fontsize=7, color='gray')

ax.scatter(*noise_src[:2], c='red', s=200, marker='*', zorder=10)
ax.annotate('NOISE\nSOURCE', noise_src[:2], textcoords="offset points", xytext=(0, 15),
           ha='center', fontsize=8, color='red', fontweight='bold')

ax.scatter(*error_mic[:2], c='black', s=150, marker='X', zorder=10)
ax.annotate('ERROR MIC\n(Driver ear)', error_mic[:2], textcoords="offset points", xytext=(15, 10),
           ha='center', fontsize=8, color='black', fontweight='bold')

ax.scatter(0.7, 0.92, c='green', s=100, marker='^', zorder=8, alpha=0.5)
ax.annotate('Ref mic\n(fixed)', (0.7, 0.92), textcoords="offset points", xytext=(15, 5),
           ha='center', fontsize=7, color='green', alpha=0.7)

max_db = max(v[1] for v in spk_positions.values())
min_db = min(v[1] for v in spk_positions.values())

for label, (pos, db) in spk_positions.items():
    norm = (db - min_db) / (max_db - min_db) if max_db > min_db else 0.5
    color = plt.cm.RdYlGn(norm)
    size = 80 + norm * 120
    ax.scatter(*pos, c=[color], s=size, marker='s', zorder=9, edgecolors='black', linewidth=0.5)
    offset_y = -18 if pos[1] > 0.3 else 12
    ax.annotate(label, pos, textcoords="offset points", xytext=(0, offset_y),
               ha='center', fontsize=7, color='darkgreen' if norm > 0.6 else 'darkred')

ax.annotate('', xy=(2.45, 0.15), xytext=(1.0, 0.15),
           arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax.text(1.7, 0.05, 'Closer to ear = Better', ha='center', fontsize=8, color='blue', style='italic')

plt.tight_layout()
plt.savefig('output/plots/position_optimization.png', dpi=150, bbox_inches='tight')
print("Saved to output/plots/position_optimization.png")
plt.close()
