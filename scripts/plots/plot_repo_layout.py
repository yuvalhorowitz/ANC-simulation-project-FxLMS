"""
Repository layout — visual directory tree for the Project Book
(Chapter 7, Project Documentation).

Renders a clean annotated tree of the project's top-level directories with
a one-line description of each, as a figure rather than a bulleted list.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUTPUT_PATH = (Path(__file__).resolve().parent.parent.parent
               / 'output' / 'plots' / 'book' / 'repo_layout.png')

# (depth, name, description). depth 0 = top-level dir, 1 = sub-item.
ROWS = [
    (0, 'src/core/',          'algorithm implementations: fxlms.py (scalar FxNLMS) and'),
    (1, '',                   'mimo_fxnlms*.py (Stages 1, 2, 3)'),
    (0, 'src/acoustic/',      'acoustic-path extraction from the simulated room'),
    (0, 'src/noise/',         'real-noise loading and mixing'),
    (0, 'src/ml/',            'ML code — the TCN controller and the four step-size classifiers'),
    (0, 'simulations_pyroom/','incremental simulator scripts (step1 … step8) + Stage 1–3'),
    (1, '',                   'MIMO evaluation scripts → output/data/mimo/'),
    (0, 'scripts/plots/',     'plot-generation scripts for every figure in this report'),
    (0, 'scripts/book/',      "this report's pipeline: build_book.py + content/*.md chapters"),
    (0, 'playground/',        'interactive Streamlit app (app.py) over the full controller stack'),
    (0, 'output/',            'generated artefacts: plots/, data/ (JSON), audio/, book/ (this report)'),
    (0, 'docs/',              'narrative source docs — mimo_results.md, ml_journey.md,'),
    (1, '',                   'position_optimization.md, fxlms_explained.md'),
]

ROOT_COLOR = '#2c5aa0'
SUB_COLOR = '#555'


def main():
    n = len(ROWS)
    row_h = 0.52
    fig_h = n * row_h + 0.8
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, n * row_h + 0.6)
    ax.axis('off')

    # Root node
    y_top = n * row_h + 0.25
    ax.text(0.15, y_top, 'ANC-simulation-project-FxLMS/',
            fontsize=13, fontweight='bold', va='center',
            family='monospace', color='#1a1a1a')

    for i, (depth, name, desc) in enumerate(ROWS):
        y = (n - 1 - i) * row_h + 0.25

        if depth == 0:
            # Tree connector
            ax.plot([0.35, 0.75], [y, y], color='#999', linewidth=1.0)
            ax.plot([0.35, 0.35], [y, y + row_h * 0.5], color='#999', linewidth=1.0)
            # Directory chip
            box = FancyBboxPatch((0.8, y - 0.17), 3.05, 0.34,
                                 boxstyle='round,pad=0.02,rounding_size=0.05',
                                 facecolor='#e7eef7', edgecolor=ROOT_COLOR,
                                 linewidth=1.3)
            ax.add_patch(box)
            ax.text(0.92, y, name, fontsize=11, fontweight='bold',
                    va='center', family='monospace', color=ROOT_COLOR)
            ax.text(4.05, y, desc, fontsize=10, va='center', color='#222')
        else:
            # Continuation line for the previous directory's description
            ax.text(4.05, y, desc, fontsize=10, va='center',
                    color=SUB_COLOR, style='italic')

    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
