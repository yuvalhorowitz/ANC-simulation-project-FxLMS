"""
ML Journey Visualization

Shows the four ML attempts we made on this ANC project, their performance
relative to a well-tuned fixed-parameter baseline, and the methodological
evolution between them.

Output: output/plots/ml_journey.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ============================================================
# Data: four ML attempts
# ============================================================

attempts = [
    {
        'short': 'Phase 1A',
        'long': 'Phase 1A\n(Single-Channel, Synthetic)',
        'mean_db': 0.37,
        'win_rate': 25,
        'data': 'Synthetic\n4 scenarios',
        'mode': 'Static (t=0)',
        'classifier': 'Binary\n(IDLE / Non-IDLE)',
        'verdict': 'Passed criteria',
        'passed': True,
        'why': 'IDLE benefits from\nlarger μ (=0.015)',
    },
    {
        'short': 'Phase 1B',
        'long': 'Phase 1B\n(Multi-Channel, Synthetic)',
        'mean_db': 0.061,
        'win_rate': 62,
        'data': 'Synthetic\n4 scenarios',
        'mode': 'Static (t=0)',
        'classifier': 'Binary\n(IDLE / Non-IDLE)',
        'verdict': 'Failed (insufficient gain)',
        'passed': False,
        'why': 'Acoustic summing\nflattens optimal μ',
    },
    {
        'short': 'Phase 2 v1',
        'long': 'Phase 2 v1\n(Real Audio, Isolated Labels)',
        'mean_db': -0.12,
        'win_rate': 0,
        'data': 'Real audio\n13 recordings',
        'mode': 'Dynamic (every 0.5s)',
        'classifier': '5-class\n(0.001–0.01)',
        'verdict': 'Failed (worse than baseline)',
        'passed': False,
        'why': 'Per-segment labels\nignore filter state',
    },
    {
        'short': 'Phase 2 v2',
        'long': 'Phase 2 v2\n(Real Audio, Rolling Sim Labels)',
        'mean_db': -0.065,
        'win_rate': 25,
        'data': 'Real audio\n13 recordings',
        'mode': 'Dynamic (every 0.5s)',
        'classifier': '5-class\n(0.001–0.01)',
        'verdict': 'Failed (recordings too stationary)',
        'passed': False,
        'why': 'Better labeling but\nrecordings stationary',
    },
]

# ============================================================
# Figure setup
# ============================================================

fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(2, 1, height_ratios=[1.4, 1.2], hspace=0.35)
ax_bars = fig.add_subplot(gs[0])
ax_table = fig.add_subplot(gs[1])

# ============================================================
# Panel A: Bar chart
# ============================================================

x = np.arange(len(attempts))
mean_dbs = [a['mean_db'] for a in attempts]
colors = ['#2a9d8f' if a['passed'] else '#e63946' for a in attempts]

bars = ax_bars.bar(x, mean_dbs, color=colors, edgecolor='black', linewidth=1.2, width=0.6)

# Reference lines
ax_bars.axhline(0, color='black', linewidth=1, alpha=0.6)
ax_bars.axhline(0.30, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax_bars.text(len(attempts) - 0.5, 0.32, '+0.30 dB target', fontsize=9, color='gray',
             style='italic', ha='right')
ax_bars.text(len(attempts) - 0.5, -0.02, 'baseline (fixed best μ)', fontsize=9,
             color='black', alpha=0.6, style='italic', ha='right', va='top')

# Value labels above/below bars
for i, (bar, db) in enumerate(zip(bars, mean_dbs)):
    height = bar.get_height()
    if height >= 0:
        ax_bars.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                     f'{db:+.3f} dB', ha='center', va='bottom',
                     fontsize=10, fontweight='bold')
    else:
        ax_bars.text(bar.get_x() + bar.get_width() / 2, height - 0.02,
                     f'{db:+.3f} dB', ha='center', va='top',
                     fontsize=10, fontweight='bold')
    # "Why" annotation
    ax_bars.text(bar.get_x() + bar.get_width() / 2, -0.55, attempts[i]['why'],
                 ha='center', va='top', fontsize=8, color='#444', style='italic')

ax_bars.set_xticks(x)
ax_bars.set_xticklabels([a['long'] for a in attempts], fontsize=9)
ax_bars.set_ylabel('Mean Noise Reduction Improvement (dB)\nvs. Best Fixed Step Size',
                   fontsize=11)
ax_bars.set_title('ML Implementation Journey — Performance vs. Fixed Baseline',
                  fontsize=13, fontweight='bold', pad=15)
ax_bars.set_ylim(-0.65, 0.55)
ax_bars.grid(True, axis='y', alpha=0.3)
ax_bars.spines['top'].set_visible(False)
ax_bars.spines['right'].set_visible(False)

# Legend
green_patch = mpatches.Patch(color='#2a9d8f', label='Passed criteria')
red_patch = mpatches.Patch(color='#e63946', label='Failed criteria')
ax_bars.legend(handles=[green_patch, red_patch], loc='upper right', fontsize=9,
               framealpha=0.9)

# ============================================================
# Panel B: Configuration table
# ============================================================

ax_table.axis('off')

col_headers = ['Attempt', 'Training Data', 'Adaptation', 'Classifier',
               'Mean dB', 'Win Rate', 'Verdict']
col_widths = [0.10, 0.13, 0.13, 0.14, 0.10, 0.10, 0.30]

# Compute column positions (cumulative)
col_positions = np.cumsum([0] + col_widths)

# Header row
header_y = 0.92
for i, header in enumerate(col_headers):
    x_pos = col_positions[i] + col_widths[i] / 2
    ax_table.text(x_pos, header_y, header, ha='center', va='center',
                  fontsize=10, fontweight='bold',
                  transform=ax_table.transAxes)

# Header underline
ax_table.plot([0.005, 0.995], [header_y - 0.08, header_y - 0.08], color='black',
              linewidth=1.5, transform=ax_table.transAxes)

# Rows
row_height = 0.18
for row_idx, attempt in enumerate(attempts):
    y = header_y - 0.18 - row_idx * row_height
    bg_color = '#f0f9f4' if attempt['passed'] else '#fdecec'

    # Row background
    rect = mpatches.Rectangle((0.005, y - row_height / 2 + 0.01),
                              0.99, row_height - 0.02,
                              facecolor=bg_color, edgecolor='none',
                              transform=ax_table.transAxes)
    ax_table.add_patch(rect)

    # Cell values
    cells = [
        attempt['short'],
        attempt['data'],
        attempt['mode'],
        attempt['classifier'],
        f"{attempt['mean_db']:+.3f}",
        f"{attempt['win_rate']}%",
        attempt['verdict'],
    ]

    for i, val in enumerate(cells):
        x_pos = col_positions[i] + col_widths[i] / 2
        weight = 'bold' if i in [0, 4] else 'normal'
        size = 9 if i == 6 else 9
        ax_table.text(x_pos, y, val, ha='center', va='center',
                      fontsize=size, fontweight=weight,
                      transform=ax_table.transAxes)

ax_table.set_title('Methodology Evolution Across Attempts',
                   fontsize=12, fontweight='bold', loc='left', y=1.02)

plt.savefig('output/plots/ml_journey.png', dpi=150, bbox_inches='tight',
            facecolor='white')
print('Saved to output/plots/ml_journey.png')
plt.close()
