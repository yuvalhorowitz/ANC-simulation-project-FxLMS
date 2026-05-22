"""
Render the scenario comparison table as a clean PNG.

Reads from output/data/mimo/scenario_comparison.json and produces a publication-
quality table image with model rows × scenario columns, color-coding the best
algorithm per scenario.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


JSON_PATH = Path(__file__).parent.parent.parent / 'output' / 'data' / 'mimo' / 'scenario_comparison.json'
OUTPUT_PATH = Path(__file__).parent.parent.parent / 'output' / 'plots' / 'scenario_comparison_table.png'


def main():
    with open(JSON_PATH) as f:
        data = json.load(f)

    scenarios = ['IDLE', 'CRUISING', 'ACCELERATION']
    algorithms = list(data['results'].keys())

    # Find best NR per scenario for highlighting
    best_per_scenario = {}
    for s in scenarios:
        best_nr = max((data['results'][a][s]['nr_db'] for a in algorithms
                       if data['results'][a][s]['stable']), default=None)
        best_per_scenario[s] = best_nr

    fig = plt.figure(figsize=(13, 6.0))
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    # ----- Title block -----
    fig.suptitle('Scenario Performance Comparison',
                 fontsize=18, fontweight='bold', y=0.97)
    ax.text(0.5, 1.02, '5 ANC algorithms × 3 driving scenarios',
            ha='center', va='bottom', fontsize=11, style='italic',
            color='#444', transform=ax.transAxes)

    # ----- Layout calculations -----
    n_rows = len(algorithms) + 2  # 2 header rows
    n_cols = 1 + 2 * len(scenarios)  # algorithm name + (NR, Conv) per scenario

    # Column boundaries (relative)
    col_widths = [0.30] + [0.115] * (2 * len(scenarios))
    col_x = np.cumsum([0.0] + col_widths)

    row_heights = [0.075, 0.06] + [0.085] * len(algorithms)
    row_y = np.cumsum([0.0] + row_heights)
    total_h = row_y[-1]

    # Convert to axes coordinates (we'll use ax.transAxes)
    def cell_rect(row, col_start, col_end, **kwargs):
        x0 = col_x[col_start]
        x1 = col_x[col_end]
        y_top = total_h - row_y[row]
        y_bot = total_h - row_y[row + 1]
        # Normalize so the table fills [0..1] vertically
        y_top_n = y_top / total_h
        y_bot_n = y_bot / total_h
        rect = mpatches.Rectangle(
            (x0, y_bot_n),
            x1 - x0,
            y_top_n - y_bot_n,
            transform=ax.transAxes,
            clip_on=False,
            **kwargs,
        )
        ax.add_patch(rect)
        return (x0 + x1) / 2, (y_top_n + y_bot_n) / 2

    def cell_text(row, col_start, col_end, text, **kwargs):
        cx, cy = cell_rect(row, col_start, col_end, fill=False,
                           edgecolor='none')
        # Allow overriding ha/va via kwargs but provide sensible defaults
        kwargs.setdefault('ha', 'center')
        kwargs.setdefault('va', 'center')
        ax.text(cx, cy, text, transform=ax.transAxes, clip_on=False, **kwargs)

    # ----- Header row 1: scenario names -----
    scenario_colors = {
        'IDLE': '#fff3cd',
        'CRUISING': '#cfe2ff',
        'ACCELERATION': '#f8d7da',
    }

    cell_rect(0, 0, 1, facecolor='#e9ecef', edgecolor='#495057', linewidth=1.2)
    cell_text(0, 0, 1, 'Algorithm', fontsize=12, fontweight='bold')

    for i, scen in enumerate(scenarios):
        c0 = 1 + 2 * i
        c1 = 1 + 2 * (i + 1)
        cell_rect(0, c0, c1, facecolor=scenario_colors[scen],
                  edgecolor='#495057', linewidth=1.2)
        cell_text(0, c0, c1, scen, fontsize=12, fontweight='bold')

    # ----- Header row 2: column subheaders -----
    cell_rect(1, 0, 1, facecolor='#f8f9fa', edgecolor='#495057', linewidth=0.8)
    for i, scen in enumerate(scenarios):
        c0 = 1 + 2 * i
        cell_rect(1, c0, c0 + 1, facecolor='#f8f9fa',
                  edgecolor='#495057', linewidth=0.8)
        cell_rect(1, c0 + 1, c0 + 2, facecolor='#f8f9fa',
                  edgecolor='#495057', linewidth=0.8)
        cell_text(1, c0, c0 + 1, 'NR (dB)', fontsize=10, fontweight='600',
                  color='#222')
        cell_text(1, c0 + 1, c0 + 2, 'Conv (s)', fontsize=10, fontweight='600',
                  color='#222')

    # ----- Data rows -----
    # Highlight Stage 3 Full MIMO row
    highlight_alg = 'Stage 3 Full MIMO'

    for r, alg in enumerate(algorithms):
        row_idx = r + 2
        row_color = '#fff8e1' if alg == highlight_alg else (
            '#ffffff' if r % 2 == 0 else '#f8f9fa'
        )

        # Algorithm name cell
        cell_rect(row_idx, 0, 1, facecolor=row_color,
                  edgecolor='#dee2e6', linewidth=0.6)
        weight = 'bold' if alg == highlight_alg else 'normal'
        cell_text(row_idx, 0, 1, alg, fontsize=11, fontweight=weight)

        # Scenario data cells
        for i, scen in enumerate(scenarios):
            cell = data['results'][alg][scen]
            c0 = 1 + 2 * i
            nr = cell['nr_db']
            conv = cell['conv_s']

            # Highlight best NR per scenario
            is_best = (best_per_scenario[scen] is not None
                       and abs(nr - best_per_scenario[scen]) < 1e-6)
            nr_bg = '#a8e6a3' if is_best else row_color

            cell_rect(row_idx, c0, c0 + 1, facecolor=nr_bg,
                      edgecolor='#dee2e6', linewidth=0.6)
            cell_rect(row_idx, c0 + 1, c0 + 2, facecolor=row_color,
                      edgecolor='#dee2e6', linewidth=0.6)

            nr_str = f'{nr:+.2f}' if cell['stable'] else 'DIV'
            conv_str = f'{conv:.2f}' if cell['stable'] else '—'
            text_weight = 'bold' if (is_best or alg == highlight_alg) else 'normal'
            cell_text(row_idx, c0, c0 + 1, nr_str, fontsize=11,
                      fontweight=text_weight,
                      color='#1a5f1a' if is_best else '#222')
            cell_text(row_idx, c0 + 1, c0 + 2, conv_str, fontsize=10,
                      color='#666', fontweight=text_weight)

    # ----- Configuration footer -----
    cfg = data['config']
    config_text = (
        f"Configuration: filter length {cfg['filter_length_default']} taps "
        f"(Stage 3 = {cfg['filter_length_stage3']}); "
        f"step size {cfg['step_size_default']} (Stage 3 = {cfg['step_size_stage3']}); "
        f"head-zone radius ±{cfg['head_zone_radius_m']*100:.0f} cm; "
        f"error mic at {cfg['error_mic_pos']}.\n"
        f"Recordings: la_idle, la_medium_cruise, la_varying (20s each, LA Downtown driving)."
    )
    fig.text(0.5, 0.04, config_text, ha='center', va='top',
             fontsize=8.5, style='italic', color='#555', wrap=True)

    # Legend for highlights
    legend_elements = [
        mpatches.Patch(facecolor='#a8e6a3', edgecolor='#dee2e6',
                       label='Best NR per scenario'),
        mpatches.Patch(facecolor='#fff8e1', edgecolor='#dee2e6',
                       label='Stage 3 Full MIMO (highlighted)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right',
              bbox_to_anchor=(1.0, -0.01), fontsize=9, ncol=2,
              frameon=False)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=180, bbox_inches='tight', facecolor='white')
    print(f"Saved: {OUTPUT_PATH}")
    plt.close()


if __name__ == '__main__':
    main()
