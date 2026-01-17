"""
Interactive Room Diagram Component for ANC Playground

Creates an interactive 2D room view where users can reposition components
using sliders with real-time visual feedback.
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, List, Optional


# Component styling configuration
COMPONENT_CONFIG = {
    'noise_source': {
        'color': '#e74c3c',
        'symbol': 'square',
        'size': 20,
        'name': 'Noise Source',
        'emoji': '🔴'
    },
    'reference_mic': {
        'color': '#3498db',
        'symbol': 'triangle-up',
        'size': 18,
        'name': 'Reference Mic',
        'emoji': '🔵'
    },
    'speaker': {
        'color': '#2ecc71',
        'symbol': 'circle',
        'size': 20,
        'name': 'Speaker',
        'emoji': '🟢'
    },
    'error_mic': {
        'color': '#9b59b6',
        'symbol': 'diamond',
        'size': 18,
        'name': 'Error Mic (Ear)',
        'emoji': '🟣'
    },
}


def create_interactive_room_diagram(
    dimensions: List[float],
    positions: Dict[str, List[float]],
    selected_component: Optional[str] = None
) -> go.Figure:
    """
    Create an interactive Plotly room diagram.

    Args:
        dimensions: [length, width, height] of room
        positions: Dictionary of position names to [x, y, z] coordinates
        selected_component: Currently selected component for repositioning

    Returns:
        Plotly figure
    """
    length, width, height = dimensions

    fig = go.Figure()

    # Draw room outline
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=length, y1=width,
        line=dict(color="#333333", width=3),
        fillcolor="#f8f9fa",
        layer="below"
    )

    # Add grid lines
    for x in range(1, int(length) + 1):
        fig.add_shape(
            type="line",
            x0=x, y0=0, x1=x, y1=width,
            line=dict(color="#dee2e6", width=1, dash="dot"),
            layer="below"
        )
    for y in range(1, int(width) + 1):
        fig.add_shape(
            type="line",
            x0=0, y0=y, x1=length, y1=y,
            line=dict(color="#dee2e6", width=1, dash="dot"),
            layer="below"
        )

    # Draw signal paths (dashed lines)
    if all(k in positions for k in ['noise_source', 'reference_mic', 'error_mic', 'speaker']):
        # Primary path: noise -> error mic
        fig.add_trace(go.Scatter(
            x=[positions['noise_source'][0], positions['error_mic'][0]],
            y=[positions['noise_source'][1], positions['error_mic'][1]],
            mode='lines',
            line=dict(color='#e74c3c', width=2, dash='dash'),
            opacity=0.5,
            name='Primary Path',
            hoverinfo='name'
        ))

        # Secondary path: speaker -> error mic
        fig.add_trace(go.Scatter(
            x=[positions['speaker'][0], positions['error_mic'][0]],
            y=[positions['speaker'][1], positions['error_mic'][1]],
            mode='lines',
            line=dict(color='#2ecc71', width=2, dash='dash'),
            opacity=0.5,
            name='Secondary Path',
            hoverinfo='name'
        ))

        # Reference path: noise -> reference mic
        fig.add_trace(go.Scatter(
            x=[positions['noise_source'][0], positions['reference_mic'][0]],
            y=[positions['noise_source'][1], positions['reference_mic'][1]],
            mode='lines',
            line=dict(color='#3498db', width=2, dash='dash'),
            opacity=0.5,
            name='Reference Path',
            hoverinfo='name'
        ))

    # Plot each component
    for comp_name, pos in positions.items():
        if comp_name in COMPONENT_CONFIG:
            config = COMPONENT_CONFIG[comp_name]
            is_selected = comp_name == selected_component

            fig.add_trace(go.Scatter(
                x=[pos[0]],
                y=[pos[1]],
                mode='markers+text',
                marker=dict(
                    color=config['color'],
                    size=config['size'] + (8 if is_selected else 0),
                    symbol=config['symbol'],
                    line=dict(
                        color='white' if not is_selected else '#ffd700',
                        width=2 if not is_selected else 4
                    )
                ),
                text=[f"({pos[0]:.1f}, {pos[1]:.1f})"],
                textposition="top center",
                textfont=dict(size=11, color='#495057'),
                name=config['name'],
                hovertemplate=(
                    f"<b>{config['name']}</b><br>"
                    f"X: %{{x:.2f}} m<br>"
                    f"Y: %{{y:.2f}} m<br>"
                    f"Z: {pos[2]:.2f} m<br>"
                    "<extra></extra>"
                )
            ))

    # Layout configuration
    fig.update_layout(
        title=dict(
            text=f"Room Layout (Top View) - {length:.1f}m × {width:.1f}m × {height:.1f}m",
            font=dict(size=16)
        ),
        xaxis=dict(
            title="Length (m)",
            range=[-0.4, length + 0.4],
            constrain="domain",
            scaleanchor="y",
            scaleratio=1,
            gridcolor='#e9ecef',
            showgrid=False,
            zeroline=False,
            fixedrange=True
        ),
        yaxis=dict(
            title="Width (m)",
            range=[-0.4, width + 0.4],
            constrain="domain",
            gridcolor='#e9ecef',
            showgrid=False,
            zeroline=False,
            fixedrange=True
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#dee2e6",
            borderwidth=1,
            font=dict(size=11)
        ),
        margin=dict(l=60, r=160, t=60, b=60),
        height=500,
        plot_bgcolor='white',
        hovermode='closest',
        dragmode=False
    )

    return fig


def render_interactive_room(params: dict) -> dict:
    """
    Render interactive room diagram with slider controls for positioning.

    Args:
        params: Current parameter dictionary

    Returns:
        Updated positions dictionary
    """
    st.subheader("🏠 Interactive Room Layout")
    st.caption("Use the sliders below to position each component, then watch the diagram update")

    dimensions = params['dimensions']
    length, width, height = dimensions

    # Initialize positions in a separate session state key to avoid conflicts
    if 'interactive_positions' not in st.session_state:
        st.session_state.interactive_positions = {
            comp: list(params['positions'][comp])
            for comp in ['noise_source', 'reference_mic', 'speaker', 'error_mic']
        }

    positions = st.session_state.interactive_positions

    # Track which component is being edited
    active_component = None

    # Create tabs for each component
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔴 Noise Source",
        "🔵 Reference Mic",
        "🟢 Speaker",
        "🟣 Error Mic"
    ])

    tabs = [
        ('noise_source', tab1),
        ('reference_mic', tab2),
        ('speaker', tab3),
        ('error_mic', tab4)
    ]

    for comp_name, tab in tabs:
        with tab:
            config = COMPONENT_CONFIG[comp_name]
            current_pos = positions[comp_name]

            col1, col2 = st.columns([1, 1])

            with col1:
                # X position slider
                new_x = st.slider(
                    "X Position (Length)",
                    min_value=0.1,
                    max_value=float(length - 0.1),
                    value=float(current_pos[0]),
                    step=0.05,
                    key=f"interactive_{comp_name}_x",
                    help=f"Move {config['name']} along room length"
                )

                # Y position slider
                new_y = st.slider(
                    "Y Position (Width)",
                    min_value=0.1,
                    max_value=float(width - 0.1),
                    value=float(current_pos[1]),
                    step=0.05,
                    key=f"interactive_{comp_name}_y",
                    help=f"Move {config['name']} along room width"
                )

            with col2:
                # Z position slider
                new_z = st.slider(
                    "Z Position (Height)",
                    min_value=0.1,
                    max_value=float(height - 0.1),
                    value=float(current_pos[2]),
                    step=0.05,
                    key=f"interactive_{comp_name}_z",
                    help=f"Set {config['name']} height"
                )

                # Show current position
                st.metric(
                    "Current Position",
                    f"({new_x:.2f}, {new_y:.2f}, {new_z:.2f})"
                )

            # Update position if changed
            new_pos = [new_x, new_y, new_z]
            if new_pos != current_pos:
                st.session_state.interactive_positions[comp_name] = new_pos
                positions[comp_name] = new_pos
                active_component = comp_name
                st.session_state.params_changed = True

    # Display the room diagram
    st.markdown("---")
    fig = create_interactive_room_diagram(dimensions, positions, active_component)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Position summary
    with st.expander("📍 All Positions Summary", expanded=False):
        cols = st.columns(4)
        for i, comp_name in enumerate(['noise_source', 'reference_mic', 'speaker', 'error_mic']):
            config = COMPONENT_CONFIG[comp_name]
            pos = positions[comp_name]
            with cols[i]:
                st.markdown(f"**{config['emoji']} {config['name']}**")
                st.caption(f"X: {pos[0]:.2f} m")
                st.caption(f"Y: {pos[1]:.2f} m")
                st.caption(f"Z: {pos[2]:.2f} m")

    return positions
