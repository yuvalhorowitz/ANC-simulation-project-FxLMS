"""
Room Diagram Component for ANC Playground

Creates a 2D top-down view of the room showing component positions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.lines import Line2D
from typing import Dict, List


def plot_room_diagram(dimensions: List[float], positions: Dict[str, List[float]]) -> plt.Figure:
    """
    Create a 2D top-down view of the room with component positions.

    Args:
        dimensions: [length, width, height] of room
        positions: Dictionary of position names to [x, y, z] coordinates

    Returns:
        Matplotlib figure
    """
    length, width, height = dimensions

    fig, ax = plt.subplots(figsize=(8, 5))

    # Draw room outline
    room_rect = FancyBboxPatch(
        (0, 0), length, width,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor='#f0f0f0',
        edgecolor='#333333',
        linewidth=2
    )
    ax.add_patch(room_rect)

    # Component colors and markers
    component_styles = {
        'noise_source': {'color': '#e74c3c', 'marker': 's', 'size': 150, 'label': 'Noise Source'},
        'reference_mic': {'color': '#3498db', 'marker': '^', 'size': 120, 'label': 'Reference Mic'},
        'speaker': {'color': '#2ecc71', 'marker': 'o', 'size': 150, 'label': 'Speaker'},
        'error_mic': {'color': '#9b59b6', 'marker': 'D', 'size': 120, 'label': 'Error Mic (Ear)'},
    }

    # Plot each component
    for name, pos in positions.items():
        if name in component_styles:
            style = component_styles[name]
            ax.scatter(
                pos[0], pos[1],
                c=style['color'],
                marker=style['marker'],
                s=style['size'],
                label=style['label'],
                edgecolors='white',
                linewidths=1.5,
                zorder=5
            )

            # Add position text
            ax.annotate(
                f'({pos[0]:.1f}, {pos[1]:.1f})',
                xy=(pos[0], pos[1]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                color='#666666'
            )

    # Draw signal paths (dashed lines)
    if all(k in positions for k in ['noise_source', 'reference_mic', 'error_mic', 'speaker']):
        # Primary path: noise -> error mic
        ax.plot(
            [positions['noise_source'][0], positions['error_mic'][0]],
            [positions['noise_source'][1], positions['error_mic'][1]],
            'r--', alpha=0.4, linewidth=1.5, label='Primary Path'
        )

        # Secondary path: speaker -> error mic
        ax.plot(
            [positions['speaker'][0], positions['error_mic'][0]],
            [positions['speaker'][1], positions['error_mic'][1]],
            'g--', alpha=0.4, linewidth=1.5, label='Secondary Path'
        )

        # Reference path: noise -> reference mic
        ax.plot(
            [positions['noise_source'][0], positions['reference_mic'][0]],
            [positions['noise_source'][1], positions['reference_mic'][1]],
            'b--', alpha=0.4, linewidth=1.5, label='Reference Path'
        )

    # Set axis properties
    ax.set_xlim(-0.2, length + 0.2)
    ax.set_ylim(-0.2, width + 0.2)
    ax.set_aspect('equal')
    ax.set_xlabel('Length (m)')
    ax.set_ylabel('Width (m)')
    ax.set_title(f'Room Layout (Top View) - {length:.1f}m × {width:.1f}m × {height:.1f}m')

    # Add legend
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        fontsize=9,
        framealpha=0.9
    )

    # Add grid
    ax.grid(True, alpha=0.3, linestyle=':')

    # Add room dimension annotations
    ax.annotate(
        f'{length:.1f}m',
        xy=(length/2, -0.1),
        ha='center',
        fontsize=10,
        color='#666666'
    )
    ax.annotate(
        f'{width:.1f}m',
        xy=(-0.1, width/2),
        ha='center',
        rotation=90,
        fontsize=10,
        color='#666666'
    )

    plt.tight_layout()
    return fig


def plot_room_side_view(dimensions: List[float], positions: Dict[str, List[float]]) -> plt.Figure:
    """
    Create a side view (x-z plane) of the room.

    Args:
        dimensions: [length, width, height] of room
        positions: Dictionary of position names to [x, y, z] coordinates

    Returns:
        Matplotlib figure
    """
    length, width, height = dimensions

    fig, ax = plt.subplots(figsize=(8, 3))

    # Draw room outline
    room_rect = Rectangle(
        (0, 0), length, height,
        facecolor='#f0f0f0',
        edgecolor='#333333',
        linewidth=2
    )
    ax.add_patch(room_rect)

    # Component styles
    component_styles = {
        'noise_source': {'color': '#e74c3c', 'marker': 's', 'size': 100},
        'reference_mic': {'color': '#3498db', 'marker': '^', 'size': 80},
        'speaker': {'color': '#2ecc71', 'marker': 'o', 'size': 100},
        'error_mic': {'color': '#9b59b6', 'marker': 'D', 'size': 80},
    }

    # Plot each component (x-z view)
    for name, pos in positions.items():
        if name in component_styles:
            style = component_styles[name]
            ax.scatter(
                pos[0], pos[2],  # x and z
                c=style['color'],
                marker=style['marker'],
                s=style['size'],
                edgecolors='white',
                linewidths=1,
                zorder=5
            )

    ax.set_xlim(-0.1, length + 0.1)
    ax.set_ylim(-0.1, height + 0.1)
    ax.set_xlabel('Length (m)')
    ax.set_ylabel('Height (m)')
    ax.set_title('Side View (Height)')
    ax.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()
    return fig
