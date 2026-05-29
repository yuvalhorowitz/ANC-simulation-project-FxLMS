"""
Cabin layout figure — top-down + side view of the simulated car cabin.

Used in the Simulation chapter (Chapter 3) of the Project Book to show
the placement of the noise source, reference mics (4), speakers (4), and
error mics (4 — head-zone grid).
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


OUTPUT_PATH = (Path(__file__).resolve().parent.parent.parent
               / 'output' / 'plots' / 'book' / 'cabin_layout.png')

# Geometry — matches scripts/plots/plot_cancellation_1x5_cabin.py
ROOM = [4.5, 1.85, 1.2]  # x (front-rear), y (left-right), z (height) in metres
NOISE = [0.5, 0.92, 0.4]
REF_MIC_PRIMARY = [1.1, 0.92, 0.8]
REF_MICS_EXTRA = {
    'firewall':  [0.3, 0.92, 0.5],
    'floor':     [2.0, 0.55, 0.15],
    'a_pillar':  [0.5, 0.15, 1.0],
    'dashboard': [0.9, 0.92, 0.8],
}
SPEAKERS = {
    'door_L':  [2.0, 0.10, 0.4],
    'door_R':  [2.0, 1.75, 0.4],
    'dash_L':  [0.8, 0.25, 0.9],
    'dash_R':  [0.8, 1.60, 0.9],
}
ERROR_CENTER = [2.5, 0.55, 1.05]
HEAD_ZONE = [(0.05, 0.05), (-0.05, 0.05), (0.05, -0.05), (-0.05, -0.05)]
ERROR_MICS = [
    [ERROR_CENTER[0], ERROR_CENTER[1] + dy, ERROR_CENTER[2] + dz]
    for dy, dz in HEAD_ZONE
]


def draw_top_view(ax):
    ax.set_xlim(-0.2, ROOM[0] + 0.2)
    ax.set_ylim(-0.2, ROOM[1] + 0.2)
    ax.set_aspect('equal')

    # Cabin outline
    ax.add_patch(Rectangle((0, 0), ROOM[0], ROOM[1],
                           fill=False, edgecolor='black', linewidth=1.5))
    # Driver / passenger seats
    ax.add_patch(Rectangle((2.0, 0.2), 0.6, 0.7, fill=False,
                           edgecolor='gray', linestyle='--', linewidth=0.8))
    ax.add_patch(Rectangle((2.0, 1.0), 0.6, 0.7, fill=False,
                           edgecolor='gray', linestyle='--', linewidth=0.8))
    ax.text(2.3, 0.55, 'driver', fontsize=8, ha='center', va='center', color='gray')
    ax.text(2.3, 1.35, 'passenger', fontsize=8, ha='center', va='center', color='gray')

    # Noise source
    ax.plot(NOISE[0], NOISE[1], '*', color='red',
            markersize=18, mec='black', mew=0.8, label='Noise source (engine)')

    # Reference mics (4 strategic + 1 baseline shown)
    for n, p in REF_MICS_EXTRA.items():
        ax.plot(p[0], p[1], '^', color='#3a8a3a',
                markersize=11, mec='black', mew=0.5)
        ax.annotate(n, (p[0], p[1]), xytext=(6, 6),
                    textcoords='offset points', fontsize=7, color='#226022')
    ax.plot(REF_MICS_EXTRA['firewall'][0], REF_MICS_EXTRA['firewall'][1],
            '^', color='#3a8a3a', markersize=11,
            mec='black', mew=0.5, label='Reference microphones (4)')

    # Speakers
    for n, p in SPEAKERS.items():
        ax.plot(p[0], p[1], 's', color='#2c5aa0',
                markersize=10, mec='black', mew=0.5)
        ax.annotate(n, (p[0], p[1]), xytext=(6, -10),
                    textcoords='offset points', fontsize=7, color='#1a3a70')
    ax.plot(SPEAKERS['door_L'][0], SPEAKERS['door_L'][1],
            's', color='#2c5aa0', markersize=10,
            mec='black', mew=0.5, label='Loudspeakers (4)')

    # Error mics (head-zone)
    for p in ERROR_MICS:
        ax.plot(p[0], p[1], 'X', color='#000', markersize=10,
                mec='white', mew=0.8)
    ax.plot(ERROR_MICS[0][0], ERROR_MICS[0][1],
            'X', color='#000', markersize=10,
            mec='white', mew=0.8, label='Error microphones (4, head-zone)')
    ax.add_patch(Circle((ERROR_CENTER[0], ERROR_CENTER[1]), 0.10,
                        fill=False, edgecolor='#444', linestyle=':', linewidth=1))

    ax.set_xlabel('x (m) — front → rear', fontsize=10)
    ax.set_ylabel('y (m) — left ↔ right', fontsize=10)
    ax.set_title('Top-down view (z = 1.05 m, driver-ear height)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.92)
    ax.grid(alpha=0.25, linestyle='--')


def draw_side_view(ax):
    ax.set_xlim(-0.2, ROOM[0] + 0.2)
    ax.set_ylim(-0.2, ROOM[2] + 0.2)
    ax.set_aspect('equal')

    ax.add_patch(Rectangle((0, 0), ROOM[0], ROOM[2],
                           fill=False, edgecolor='black', linewidth=1.5))
    # Seat back hint
    ax.add_patch(Rectangle((2.55, 0.0), 0.05, 0.95, fill=True,
                           color='lightgray', alpha=0.6))

    ax.plot(NOISE[0], NOISE[2], '*', color='red',
            markersize=18, mec='black', mew=0.8)
    for p in REF_MICS_EXTRA.values():
        ax.plot(p[0], p[2], '^', color='#3a8a3a',
                markersize=11, mec='black', mew=0.5)
    for p in SPEAKERS.values():
        ax.plot(p[0], p[2], 's', color='#2c5aa0',
                markersize=10, mec='black', mew=0.5)
    for p in ERROR_MICS:
        ax.plot(p[0], p[2], 'X', color='#000',
                markersize=10, mec='white', mew=0.8)

    ax.set_xlabel('x (m) — front → rear', fontsize=10)
    ax.set_ylabel('z (m) — floor → ceiling', fontsize=10)
    ax.set_title('Side view', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.25, linestyle='--')


def main():
    fig, axes = plt.subplots(2, 1, figsize=(10, 7),
                             gridspec_kw={'height_ratios': [1.4, 1.0]})
    draw_top_view(axes[0])
    draw_side_view(axes[1])
    fig.suptitle('Simulated cabin geometry (4.5 × 1.85 × 1.2 m shoebox)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
