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

COMPONENT_ORDER = ['noise_source', 'reference_mic', 'speaker', 'error_mic']


def clamp_position(pos: List[float], dimensions: List[float], margin: float = 0.1) -> List[float]:
    """Clamp a position to stay within room bounds."""
    return [
        max(margin, min(pos[0], dimensions[0] - margin)),
        max(margin, min(pos[1], dimensions[1] - margin)),
        max(margin, min(pos[2], dimensions[2] - margin)),
    ]


def clamp_all_positions(positions: Dict[str, List[float]], dimensions: List[float]) -> Dict[str, List[float]]:
    """Clamp all positions to stay within room bounds."""
    return {
        name: clamp_position(pos, dimensions)
        for name, pos in positions.items()
    }


def create_interactive_room_diagram(
    dimensions: List[float],
    positions: Dict[str, List[float]],
    selected_component: Optional[str] = None
) -> go.Figure:
    """
    Create an interactive Plotly room diagram.
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

    # Draw signal paths
    if all(k in positions for k in COMPONENT_ORDER):
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
    st.caption("Use the sliders to position each component - the diagram updates in real-time")

    dimensions = params['dimensions']
    length, width, height = dimensions

    # Initialize positions in session state
    if 'interactive_positions' not in st.session_state:
        st.session_state.interactive_positions = {
            comp: list(params['positions'][comp])
            for comp in COMPONENT_ORDER
        }

    # Clamp positions to current room dimensions (handles dimension changes)
    st.session_state.interactive_positions = clamp_all_positions(
        st.session_state.interactive_positions, dimensions
    )

    positions = st.session_state.interactive_positions

    # Track which component is being edited
    active_component = None

    # Create two columns: sliders on left, diagram on right
    col_sliders, col_diagram = st.columns([1, 2])

    with col_sliders:
        for comp_name in COMPONENT_ORDER:
            config = COMPONENT_CONFIG[comp_name]
            current_pos = positions[comp_name]

            st.markdown(f"**{config['emoji']} {config['name']}**")

            # X and Y sliders side by side
            c1, c2 = st.columns(2)
            with c1:
                new_x = st.slider(
                    "X",
                    min_value=0.1,
                    max_value=float(length - 0.1),
                    value=float(current_pos[0]),
                    step=0.1,
                    key=f"drag_{comp_name}_x",
                    label_visibility="collapsed"
                )
            with c2:
                new_y = st.slider(
                    "Y",
                    min_value=0.1,
                    max_value=float(width - 0.1),
                    value=float(current_pos[1]),
                    step=0.1,
                    key=f"drag_{comp_name}_y",
                    label_visibility="collapsed"
                )

            # Z slider (height)
            new_z = st.slider(
                f"Height (Z)",
                min_value=0.1,
                max_value=float(height - 0.1),
                value=float(current_pos[2]),
                step=0.1,
                key=f"drag_{comp_name}_z",
                label_visibility="collapsed"
            )

            # Update position if changed
            new_pos = [new_x, new_y, new_z]
            if new_pos != current_pos:
                st.session_state.interactive_positions[comp_name] = new_pos
                positions[comp_name] = new_pos
                active_component = comp_name
                st.session_state.params_changed = True

            st.markdown("---")

    with col_diagram:
        # Display the room diagram
        fig = create_interactive_room_diagram(dimensions, positions, active_component)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Position summary
        st.markdown("**Positions:** " + " | ".join([
            f"{COMPONENT_CONFIG[c]['emoji']} ({positions[c][0]:.1f}, {positions[c][1]:.1f}, {positions[c][2]:.1f})"
            for c in COMPONENT_ORDER
        ]))

    return positions
