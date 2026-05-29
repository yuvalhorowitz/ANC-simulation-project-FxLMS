"""
Block diagram of the FxLMS Active Noise Control system.

Used as Figure 1 in the Project Book Abstract.
Shows: noise source → primary path → error mic; reference mic → adaptive filter
W(z) → secondary path S(z) → error mic; error feedback through filtered-x using
the secondary-path estimate Ŝ(z) to update W(z).
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT_PATH = (Path(__file__).resolve().parent.parent.parent
               / 'output' / 'plots' / 'book' / 'block_diagram.png')


def box(ax, x, y, w, h, text, fc='#e7eef7', ec='#2c5aa0', fontsize=10, bold=False):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle='round,pad=0.02,rounding_size=0.04',
                       linewidth=1.6, facecolor=fc, edgecolor=ec)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold' if bold else 'normal')


def arrow(ax, x0, y0, x1, y1, color='#333', width=1.6, label=None,
          label_offset=(0, 0.06), curved=False):
    style = 'arc3,rad=0.25' if curved else 'arc3,rad=0.0'
    a = FancyArrowPatch((x0, y0), (x1, y1),
                       arrowstyle='-|>', mutation_scale=14,
                       linewidth=width, color=color,
                       connectionstyle=style)
    ax.add_patch(a)
    if label:
        ax.text((x0 + x1) / 2 + label_offset[0],
                (y0 + y1) / 2 + label_offset[1],
                label, ha='center', va='center', fontsize=9,
                color=color, style='italic')


def main():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.2)
    ax.set_aspect('equal')
    ax.axis('off')

    # Top row — physical signal flow
    box(ax, 0.2, 3.5, 1.4, 0.9, 'Noise\nSource',
        fc='#fde2e2', ec='#b03a3a', bold=True)
    box(ax, 2.2, 3.5, 1.4, 0.9, 'Reference\nMicrophone', fc='#e3f3e0', ec='#3a8a3a')
    box(ax, 4.8, 3.5, 1.8, 0.9, 'Adaptive Filter\nW(z)', fc='#fff3cd', ec='#a8851b', bold=True)
    box(ax, 7.6, 3.5, 1.4, 0.9, 'Loudspeaker', fc='#e7eef7', ec='#2c5aa0')
    box(ax, 9.6, 3.5, 1.2, 0.9, 'Error Mic\n(at ear)',
        fc='#e3f3e0', ec='#3a8a3a', bold=True)

    # Acoustic paths
    arrow(ax, 1.6, 3.95, 2.2, 3.95, label='x(n)')
    arrow(ax, 3.6, 3.95, 4.8, 3.95)
    arrow(ax, 6.6, 3.95, 7.6, 3.95, label='y(n)')
    arrow(ax, 9.0, 3.95, 9.6, 3.95)

    # Primary path — curved, from noise source down to error mic
    box(ax, 0.4, 1.7, 1.0, 0.55, 'Primary\nP(z)', fc='#fde2e2', ec='#b03a3a', fontsize=8)
    arrow(ax, 0.9, 3.5, 0.9, 2.25, color='#b03a3a')
    arrow(ax, 1.4, 1.97, 9.95, 1.97, color='#b03a3a', label='d(n)',
          label_offset=(0, 0.18))
    arrow(ax, 10.2, 2.4, 10.2, 3.5, color='#b03a3a')

    # Secondary path under speaker → error mic
    box(ax, 7.7, 2.5, 1.2, 0.5, 'Secondary\nS(z)', fc='#e7eef7', ec='#2c5aa0', fontsize=8)
    arrow(ax, 8.3, 3.5, 8.3, 3.0, color='#2c5aa0')
    arrow(ax, 8.9, 2.75, 9.85, 2.75, color='#2c5aa0')

    # Estimated secondary path (used for filtered-x)
    box(ax, 4.4, 1.0, 1.4, 0.6, 'Ŝ(z)\n(estimate)', fc='#f1e5fb', ec='#6f42c1', fontsize=9)
    arrow(ax, 2.9, 3.5, 2.9, 1.3, color='#6f42c1', curved=False)
    arrow(ax, 2.9, 1.3, 4.4, 1.3, color='#6f42c1')
    arrow(ax, 5.8, 1.3, 6.9, 1.3, color='#6f42c1', label='x_f(n)',
          label_offset=(0, 0.18))

    # Error feedback (from error mic back to weight update at filter)
    arrow(ax, 10.2, 3.5, 10.2, 0.6, color='#2a7a2a', curved=False)
    arrow(ax, 10.2, 0.6, 7.4, 0.6, color='#2a7a2a', label='e(n)',
          label_offset=(0, 0.18))
    arrow(ax, 7.4, 0.6, 5.7, 1.0, color='#2a7a2a')

    # Update label going up to W(z)
    arrow(ax, 5.2, 1.6, 5.2, 3.5, color='#a8851b', curved=False,
          label='LMS update', label_offset=(0.7, 0))

    # Title
    ax.text(5.5, 4.85, 'Filtered-x LMS Active Noise Control — block diagram',
            ha='center', fontsize=13, fontweight='bold')
    ax.text(5.5, 4.6,
            'noise + anti-noise sum acoustically at the error microphone; '
            'filter weights adapt continuously to drive e(n) → 0',
            ha='center', fontsize=9, style='italic', color='#444')

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
