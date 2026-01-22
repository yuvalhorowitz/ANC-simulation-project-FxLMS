"""
Interactive Room Diagram Component for ANC Playground

Creates an interactive 2D room view with car interior visualization.
Users can reposition components using sliders or by clicking on the diagram.
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from presets import FOUR_SPEAKER_CONFIG, REF_MIC_4CH_CONFIG


# Component styling configuration
COMPONENT_CONFIG = {
    'noise_source': {
        'color': '#e74c3c',
        'symbol': 'square',
        'size': 22,
        'name': 'Noise Source',
        'emoji': '🔴'
    },
    'reference_mic': {
        'color': '#3498db',
        'symbol': 'triangle-up',
        'size': 20,
        'name': 'Reference Mic',
        'emoji': '🔵'
    },
    'speaker': {
        'color': '#2ecc71',
        'symbol': 'circle',
        'size': 22,
        'name': 'Speaker',
        'emoji': '🟢'
    },
    'error_mic': {
        'color': '#9b59b6',
        'symbol': 'star',
        'size': 22,
        'name': 'Error Mic (Ear)',
        'emoji': '🟣'
    },
}

# 4-speaker styling
SPEAKER_4CH_CONFIG = {
    'front_left': {'color': '#27ae60', 'name': 'Front Left (Door)'},
    'front_right': {'color': '#2ecc71', 'name': 'Front Right (Door)'},
    'dash_left': {'color': '#1abc9c', 'name': 'Dash Left'},
    'dash_right': {'color': '#16a085', 'name': 'Dash Right'},
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


def add_car_interior(fig, length: float, width: float):
    """Add car interior elements to the figure."""

    # Calculate proportional positions based on room dimensions
    # These are relative to a standard sedan (4.5m x 1.85m)
    scale_x = length / 4.5
    scale_y = width / 1.85

    # Windshield (front of car)
    windshield_x = [0, 0, 0.35 * scale_x, 0.35 * scale_x, 0]
    windshield_y = [0.15 * scale_y, width - 0.15 * scale_y,
                   width - 0.25 * scale_y, 0.25 * scale_y, 0.15 * scale_y]
    fig.add_trace(go.Scatter(
        x=windshield_x, y=windshield_y,
        fill='toself',
        fillcolor='rgba(135, 206, 235, 0.4)',
        line=dict(color='#333', width=2),
        name='Windshield',
        hoverinfo='name',
        showlegend=False
    ))

    # Rear window
    rear_x = [length, length, length - 0.25 * scale_x, length - 0.25 * scale_x, length]
    rear_y = [0.2 * scale_y, width - 0.2 * scale_y,
              width - 0.3 * scale_y, 0.3 * scale_y, 0.2 * scale_y]
    fig.add_trace(go.Scatter(
        x=rear_x, y=rear_y,
        fill='toself',
        fillcolor='rgba(135, 206, 235, 0.4)',
        line=dict(color='#333', width=2),
        name='Rear Window',
        hoverinfo='name',
        showlegend=False
    ))

    # Dashboard
    dash_x0 = 0.4 * scale_x
    dash_width = 0.55 * scale_x
    dash_y0 = 0.2 * scale_y
    dash_height = width - 0.4 * scale_y
    fig.add_shape(
        type="rect",
        x0=dash_x0, y0=dash_y0,
        x1=dash_x0 + dash_width, y1=dash_y0 + dash_height,
        fillcolor='rgba(85, 85, 85, 0.5)',
        line=dict(color='#333', width=1),
        layer="below"
    )
    fig.add_annotation(
        x=dash_x0 + dash_width/2, y=width/2,
        text="Dashboard",
        showarrow=False,
        font=dict(size=9, color='white')
    )

    # Driver seat
    seat_x0 = 1.2 * scale_x
    seat_width = 0.9 * scale_x
    seat_y0 = 0.15 * scale_y
    seat_height = 0.7 * scale_y
    fig.add_shape(
        type="rect",
        x0=seat_x0, y0=seat_y0,
        x1=seat_x0 + seat_width, y1=seat_y0 + seat_height,
        fillcolor='rgba(139, 69, 19, 0.6)',
        line=dict(color='#5D3A1A', width=2),
        layer="below"
    )
    fig.add_annotation(
        x=seat_x0 + seat_width/2, y=seat_y0 + seat_height/2,
        text="Driver<br>Seat",
        showarrow=False,
        font=dict(size=9, color='white')
    )

    # Passenger seat
    pass_y0 = width - 0.15 * scale_y - 0.7 * scale_y
    fig.add_shape(
        type="rect",
        x0=seat_x0, y0=pass_y0,
        x1=seat_x0 + seat_width, y1=pass_y0 + seat_height,
        fillcolor='rgba(139, 69, 19, 0.6)',
        line=dict(color='#5D3A1A', width=2),
        layer="below"
    )
    fig.add_annotation(
        x=seat_x0 + seat_width/2, y=pass_y0 + seat_height/2,
        text="Passenger<br>Seat",
        showarrow=False,
        font=dict(size=9, color='white')
    )

    # Center console
    console_x0 = 1.0 * scale_x
    console_width = 1.2 * scale_x
    console_y0 = seat_y0 + seat_height
    console_height = pass_y0 - (seat_y0 + seat_height)
    if console_height > 0.1:
        fig.add_shape(
            type="rect",
            x0=console_x0, y0=console_y0,
            x1=console_x0 + console_width, y1=console_y0 + console_height,
            fillcolor='rgba(68, 68, 68, 0.5)',
            line=dict(color='#333', width=1),
            layer="below"
        )

    # Rear seats
    rear_x0 = 2.9 * scale_x
    rear_width = 1.0 * scale_x
    rear_y0 = 0.2 * scale_y
    rear_height = width - 0.4 * scale_y
    if rear_x0 + rear_width < length:
        fig.add_shape(
            type="rect",
            x0=rear_x0, y0=rear_y0,
            x1=rear_x0 + rear_width, y1=rear_y0 + rear_height,
            fillcolor='rgba(160, 82, 45, 0.4)',
            line=dict(color='#5D3A1A', width=2),
            layer="below"
        )
        fig.add_annotation(
            x=rear_x0 + rear_width/2, y=width/2,
            text="Rear Seats",
            showarrow=False,
            font=dict(size=9, color='white')
        )

    # Steering wheel (circle)
    steering_x = 0.85 * scale_x
    steering_y = 0.5 * scale_y
    steering_r = 0.12 * min(scale_x, scale_y)

    # Draw steering wheel as a circle using scatter
    theta = [i * 3.14159 * 2 / 30 for i in range(31)]
    circle_x = [steering_x + steering_r * 0.8 * (1 - abs(t - 3.14159) / 3.14159) for t in theta]
    # Simplified - just add a marker
    fig.add_trace(go.Scatter(
        x=[steering_x], y=[steering_y],
        mode='markers',
        marker=dict(color='#333', size=15, symbol='circle'),
        name='Steering',
        hoverinfo='name',
        showlegend=False
    ))

    # FRONT label
    fig.add_annotation(
        x=0.15 * scale_x, y=width/2,
        text="FRONT",
        textangle=-90,
        showarrow=False,
        font=dict(size=10, color='#333', family='Arial Black')
    )


def create_interactive_room_diagram(
    dimensions: List[float],
    positions: Dict[str, List[float]],
    selected_component: Optional[str] = None,
    speakers_4ch: Optional[Dict[str, List[float]]] = None,
    ref_mics_4ch: Optional[Dict[str, List[float]]] = None,
    is_car: bool = True
) -> go.Figure:
    """
    Create an interactive Plotly room diagram with car interior.

    Args:
        dimensions: Room [length, width, height]
        positions: Component positions
        selected_component: Currently selected component
        speakers_4ch: Optional dict of 4-speaker positions (for multi-speaker mode)
        ref_mics_4ch: Optional dict of 4-ref-mic positions (for multi-ref-mic mode)
        is_car: Whether to draw car interior elements
    """
    length, width, height = dimensions
    is_multi_speaker = speakers_4ch is not None and len(speakers_4ch) > 0
    is_multi_ref_mic = ref_mics_4ch is not None and len(ref_mics_4ch) > 0

    fig = go.Figure()

    # Draw room outline
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=length, y1=width,
        line=dict(color="#333333", width=3),
        fillcolor="#f5f5f0",
        layer="below"
    )

    # Add car interior elements if this looks like a car (length > width)
    if is_car and length > width * 1.5:
        add_car_interior(fig, length, width)
    else:
        # Add simple grid lines for non-car rooms
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
    if all(k in positions for k in ['noise_source', 'error_mic']):
        # Primary path: noise -> error mic
        fig.add_trace(go.Scatter(
            x=[positions['noise_source'][0], positions['error_mic'][0]],
            y=[positions['noise_source'][1], positions['error_mic'][1]],
            mode='lines',
            line=dict(color='#e74c3c', width=2, dash='dash'),
            opacity=0.5,
            name='Primary Path (Noise)',
            hoverinfo='name'
        ))

        # Reference paths
        if is_multi_ref_mic:
            # Draw paths from noise source to each ref mic
            for ref_name, ref_pos in ref_mics_4ch.items():
                fig.add_trace(go.Scatter(
                    x=[positions['noise_source'][0], ref_pos[0]],
                    y=[positions['noise_source'][1], ref_pos[1]],
                    mode='lines',
                    line=dict(color='#3498db', width=1, dash='dot'),
                    opacity=0.3,
                    name=f'Ref Path ({ref_name})',
                    hoverinfo='name',
                    showlegend=False
                ))
        elif 'reference_mic' in positions:
            # Single reference path: noise -> reference mic
            fig.add_trace(go.Scatter(
                x=[positions['noise_source'][0], positions['reference_mic'][0]],
                y=[positions['noise_source'][1], positions['reference_mic'][1]],
                mode='lines',
                line=dict(color='#3498db', width=2, dash='dash'),
                opacity=0.5,
                name='Reference Path',
                hoverinfo='name'
            ))

        # Secondary paths
        if is_multi_speaker:
            # Draw paths from all 4 speakers to error mic
            for spk_name, spk_pos in speakers_4ch.items():
                fig.add_trace(go.Scatter(
                    x=[spk_pos[0], positions['error_mic'][0]],
                    y=[spk_pos[1], positions['error_mic'][1]],
                    mode='lines',
                    line=dict(color='#2ecc71', width=1, dash='dot'),
                    opacity=0.3,
                    name=f'Secondary ({spk_name})',
                    hoverinfo='name',
                    showlegend=False
                ))
        elif 'speaker' in positions:
            # Single speaker secondary path
            fig.add_trace(go.Scatter(
                x=[positions['speaker'][0], positions['error_mic'][0]],
                y=[positions['speaker'][1], positions['error_mic'][1]],
                mode='lines',
                line=dict(color='#2ecc71', width=2, dash='dash'),
                opacity=0.5,
                name='Secondary Path (Anti-noise)',
                hoverinfo='name'
            ))

    # Plot 4 speakers if in multi-speaker mode
    if is_multi_speaker:
        for spk_name, spk_pos in speakers_4ch.items():
            spk_config = SPEAKER_4CH_CONFIG.get(spk_name, {'color': '#2ecc71', 'name': spk_name})
            fig.add_trace(go.Scatter(
                x=[spk_pos[0]],
                y=[spk_pos[1]],
                mode='markers+text',
                marker=dict(
                    color=spk_config['color'],
                    size=18,
                    symbol='circle',
                    line=dict(color='white', width=2)
                ),
                text=['🔊'],
                textposition="middle center",
                textfont=dict(size=10),
                name=spk_config['name'],
                hovertemplate=(
                    f"<b>{spk_config['name']}</b><br>"
                    f"X: {spk_pos[0]:.2f} m<br>"
                    f"Y: {spk_pos[1]:.2f} m<br>"
                    f"Z: {spk_pos[2]:.2f} m<br>"
                    "<extra></extra>"
                )
            ))

    # Plot 4 reference mics if in multi-ref-mic mode
    if is_multi_ref_mic:
        for ref_name, ref_pos in ref_mics_4ch.items():
            ref_config = REF_MIC_4CH_CONFIG.get(ref_name, {'color': '#3498db', 'name': ref_name})
            fig.add_trace(go.Scatter(
                x=[ref_pos[0]],
                y=[ref_pos[1]],
                mode='markers+text',
                marker=dict(
                    color=ref_config['color'],
                    size=16,
                    symbol='triangle-up',
                    line=dict(color='white', width=2)
                ),
                text=['🎤'],
                textposition="middle center",
                textfont=dict(size=8),
                name=ref_config['name'],
                hovertemplate=(
                    f"<b>{ref_config['name']}</b><br>"
                    f"X: {ref_pos[0]:.2f} m<br>"
                    f"Y: {ref_pos[1]:.2f} m<br>"
                    f"Z: {ref_pos[2]:.2f} m<br>"
                    "<extra></extra>"
                )
            ))

    # Plot each component (skip speaker if in multi-speaker mode, skip reference_mic if in multi-ref-mic mode)
    for comp_name, pos in positions.items():
        if comp_name in COMPONENT_CONFIG:
            # Skip single speaker in multi-speaker mode
            if comp_name == 'speaker' and is_multi_speaker:
                continue
            # Skip single reference_mic in multi-ref-mic mode
            if comp_name == 'reference_mic' and is_multi_ref_mic:
                continue

            config = COMPONENT_CONFIG[comp_name]
            is_selected = comp_name == selected_component

            fig.add_trace(go.Scatter(
                x=[pos[0]],
                y=[pos[1]],
                mode='markers+text',
                marker=dict(
                    color=config['color'],
                    size=config['size'] + (10 if is_selected else 0),
                    symbol=config['symbol'],
                    line=dict(
                        color='white' if not is_selected else '#ffd700',
                        width=2 if not is_selected else 4
                    )
                ),
                text=[f"({pos[0]:.1f}, {pos[1]:.1f})"],
                textposition="top center",
                textfont=dict(size=10, color='#333'),
                name=config['name'],
                hovertemplate=(
                    f"<b>{config['name']}</b><br>"
                    f"X: %{{x:.2f}} m<br>"
                    f"Y: %{{y:.2f}} m<br>"
                    f"Z: {pos[2]:.2f} m<br>"
                    "<b>Use sliders to move</b><br>"
                    "<extra></extra>"
                )
            ))

    # Layout configuration
    mode_parts = []
    if is_multi_speaker:
        mode_parts.append("4-Speaker")
    if is_multi_ref_mic:
        mode_parts.append("4-Ref-Mic")
    title_suffix = f" ({' + '.join(mode_parts)})" if mode_parts else ""

    fig.update_layout(
        title=dict(
            text=f"Car Interior - {length:.1f}m × {width:.1f}m × {height:.1f}m{title_suffix}",
            font=dict(size=14)
        ),
        xaxis=dict(
            title="Length (m) ← Front | Rear →",
            range=[-0.3, length + 0.3],
            constrain="domain",
            scaleanchor="y",
            scaleratio=1,
            gridcolor='#e9ecef',
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Width (m)",
            range=[-0.3, width + 0.3],
            constrain="domain",
            gridcolor='#e9ecef',
            showgrid=False,
            zeroline=False,
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
            font=dict(size=10)
        ),
        margin=dict(l=60, r=160, t=50, b=60),
        height=450,
        plot_bgcolor='white',
        hovermode='closest',
        dragmode=False
    )

    return fig


def on_position_change(comp_name: str, axis: str):
    """Callback when a position slider changes."""
    key = f"drag_{comp_name}_{axis}"
    value = st.session_state[key]

    if 'interactive_positions' not in st.session_state:
        return

    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
    st.session_state.interactive_positions[comp_name][axis_idx] = value
    st.session_state.params_changed = True


def render_interactive_room(params: dict) -> dict:
    """
    Render interactive room diagram with slider controls for positioning.

    Args:
        params: Current parameter dictionary

    Returns:
        Updated positions dictionary
    """
    # Check if we're in multi-speaker mode
    is_multi_speaker = params.get('speaker_mode') == '4-Speaker System'
    speakers_4ch = params.get('speakers') if is_multi_speaker else None

    # Check if we're in multi-ref-mic mode
    is_multi_ref_mic = params.get('ref_mic_mode') == '4-Reference Mic System'
    ref_mics_4ch = params.get('ref_mics') if is_multi_ref_mic else None

    # Build title suffix
    mode_parts = []
    if is_multi_speaker:
        mode_parts.append("4-Speaker")
    if is_multi_ref_mic:
        mode_parts.append("4-Ref-Mic")
    title_suffix = f" ({' + '.join(mode_parts)})" if mode_parts else ""

    st.subheader(f"🚗 Car Interior Layout{title_suffix}")
    st.caption("Use the sliders below to position each component")

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

    # Create two columns: diagram on left, sliders on right
    col_diagram, col_sliders = st.columns([2, 1])

    with col_diagram:
        # Display the room diagram
        fig = create_interactive_room_diagram(
            dimensions, positions, active_component, speakers_4ch, ref_mics_4ch
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Legend - hide components that are shown in multi-mode
        components_to_show = [
            c for c in COMPONENT_ORDER
            if not (c == 'speaker' and is_multi_speaker)
            and not (c == 'reference_mic' and is_multi_ref_mic)
        ]
        legend_text = " | ".join([
            f"{COMPONENT_CONFIG[c]['emoji']} {COMPONENT_CONFIG[c]['name']}"
            for c in components_to_show
        ])
        st.caption(f"**Legend:** {legend_text}")

    with col_sliders:
        st.markdown("### Component Positions")

        for comp_name in COMPONENT_ORDER:
            # Skip single speaker control in multi-speaker mode
            if comp_name == 'speaker' and is_multi_speaker:
                continue
            # Skip single reference_mic control in multi-ref-mic mode
            if comp_name == 'reference_mic' and is_multi_ref_mic:
                continue

            config = COMPONENT_CONFIG[comp_name]
            current_pos = positions[comp_name]

            with st.expander(f"{config['emoji']} {config['name']}", expanded=False):
                # X slider
                st.slider(
                    "X (Length)",
                    min_value=0.1,
                    max_value=float(length - 0.1),
                    value=float(current_pos[0]),
                    step=0.05,
                    key=f"drag_{comp_name}_x",
                    on_change=on_position_change,
                    args=(comp_name, 'x'),
                )
                # Y slider
                st.slider(
                    "Y (Width)",
                    min_value=0.1,
                    max_value=float(width - 0.1),
                    value=float(current_pos[1]),
                    step=0.05,
                    key=f"drag_{comp_name}_y",
                    on_change=on_position_change,
                    args=(comp_name, 'y'),
                )
                # Z slider
                st.slider(
                    "Z (Height)",
                    min_value=0.1,
                    max_value=float(height - 0.1),
                    value=float(current_pos[2]),
                    step=0.05,
                    key=f"drag_{comp_name}_z",
                    on_change=on_position_change,
                    args=(comp_name, 'z'),
                )

        # Show 4-speaker info if in multi-speaker mode
        if is_multi_speaker and speakers_4ch:
            st.markdown("---")
            st.markdown("### 4-Speaker Positions (Fixed)")
            for name, pos in speakers_4ch.items():
                display_name = name.replace('_', ' ').title()
                st.markdown(f"🔊 **{display_name}**: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

        # Show 4-ref-mic info if in multi-ref-mic mode
        if is_multi_ref_mic and ref_mics_4ch:
            st.markdown("---")
            st.markdown("### 4-Ref-Mic Positions (Fixed)")
            for name, pos in ref_mics_4ch.items():
                display_name = name.replace('_', ' ').title()
                st.markdown(f"🎤 **{display_name}**: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

    return positions
