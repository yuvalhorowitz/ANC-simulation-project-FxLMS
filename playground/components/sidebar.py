"""
Sidebar Controls for ANC Playground

Provides all parameter input widgets for the Streamlit sidebar.
"""

import streamlit as st
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from presets import (
    ROOM_PRESETS, SCENARIO_PRESETS, NOISE_PRESETS,
    FXLMS_PRESETS, DEFAULTS
)


def init_session_state():
    """Initialize session state with default values."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.room_preset = DEFAULTS['room_preset']
        st.session_state.scenario_preset = DEFAULTS['scenario_preset']
        st.session_state.fxlms_preset = DEFAULTS['fxlms_preset']

        # Load initial preset values
        preset = ROOM_PRESETS[DEFAULTS['room_preset']]
        st.session_state.dimensions = preset['dimensions'].copy()
        st.session_state.absorption = preset['absorption']
        st.session_state.max_order = preset['max_order']
        st.session_state.positions = {k: v.copy() for k, v in preset['positions'].items()}

        # FxLMS defaults
        fxlms = FXLMS_PRESETS[DEFAULTS['fxlms_preset']]
        st.session_state.filter_length = fxlms['filter_length']
        st.session_state.step_size = fxlms['step_size']
        st.session_state.duration = DEFAULTS['duration']

        # Track if params changed since last run
        st.session_state.params_changed = False


def clear_results_if_changed():
    """Clear old results when parameters change."""
    if st.session_state.get('params_changed', False):
        if 'results' in st.session_state:
            del st.session_state.results
        st.session_state.params_changed = False


def on_param_change():
    """Callback when any parameter changes."""
    st.session_state.params_changed = True


def on_room_preset_change():
    """Callback when room preset changes - update all room values."""
    preset_name = st.session_state.room_preset_select
    preset = ROOM_PRESETS[preset_name]

    st.session_state.room_preset = preset_name
    st.session_state.dim_length = preset['dimensions'][0]
    st.session_state.dim_width = preset['dimensions'][1]
    st.session_state.dim_height = preset['dimensions'][2]
    st.session_state.absorption_val = preset['absorption']
    st.session_state.max_order_val = preset['max_order']

    # Update positions
    for key, pos in preset['positions'].items():
        st.session_state[f'pos_{key}_x'] = pos[0]
        st.session_state[f'pos_{key}_y'] = pos[1]
        st.session_state[f'pos_{key}_z'] = pos[2]

    st.session_state.params_changed = True


def on_fxlms_preset_change():
    """Callback when FxLMS preset changes."""
    preset_name = st.session_state.fxlms_preset_select
    preset = FXLMS_PRESETS[preset_name]

    st.session_state.fxlms_preset = preset_name
    st.session_state.filter_length_val = preset['filter_length']
    st.session_state.step_size_val = preset['step_size']
    st.session_state.params_changed = True


def render_sidebar() -> Dict[str, Any]:
    """
    Render the sidebar with all parameter controls.

    Returns:
        Dictionary of all parameters for simulation
    """
    init_session_state()

    st.sidebar.title("Parameters")

    # Show warning if parameters changed
    if st.session_state.get('params_changed', False) and 'results' in st.session_state:
        st.sidebar.warning("⚠️ Parameters changed. Click 'Run' to update results.")

    params = {}

    # ==================== Room Configuration ====================
    st.sidebar.header("🏠 Room")

    # Room preset selector
    room_preset = st.sidebar.selectbox(
        "Room Preset",
        options=list(ROOM_PRESETS.keys()),
        index=list(ROOM_PRESETS.keys()).index(st.session_state.get('room_preset', DEFAULTS['room_preset'])),
        key="room_preset_select",
        on_change=on_room_preset_change
    )

    preset = ROOM_PRESETS[room_preset]

    # Dimensions - use session state with defaults from preset
    st.sidebar.subheader("Dimensions")

    length = st.sidebar.slider(
        "Length (m)", 2.0, 10.0,
        value=st.session_state.get('dim_length', preset['dimensions'][0]),
        step=0.1, key="dim_length",
        on_change=on_param_change
    )
    width = st.sidebar.slider(
        "Width (m)", 1.0, 5.0,
        value=st.session_state.get('dim_width', preset['dimensions'][1]),
        step=0.1, key="dim_width",
        on_change=on_param_change
    )
    height = st.sidebar.slider(
        "Height (m)", 1.0, 3.0,
        value=st.session_state.get('dim_height', preset['dimensions'][2]),
        step=0.1, key="dim_height",
        on_change=on_param_change
    )
    params['dimensions'] = [length, width, height]

    # Acoustic properties
    st.sidebar.subheader("Acoustics")
    params['absorption'] = st.sidebar.slider(
        "Absorption", 0.1, 0.99,
        value=st.session_state.get('absorption_val', preset['absorption']),
        step=0.05, key="absorption_val",
        help="Higher = more sound absorbed (less reverb)",
        on_change=on_param_change
    )
    params['max_order'] = st.sidebar.slider(
        "Reflection Order", 0, 6,
        value=st.session_state.get('max_order_val', preset['max_order']),
        key="max_order_val",
        help="Higher = more reflections computed",
        on_change=on_param_change
    )

    # ==================== Position Configuration ====================
    st.sidebar.header("📍 Positions")

    # Constrain positions to room dimensions
    def pos_slider(label, key_base, default_pos, dim_max):
        col1, col2, col3 = st.sidebar.columns(3)
        x = col1.number_input(
            "X", 0.1, length-0.1,
            value=min(st.session_state.get(f'{key_base}_x', default_pos[0]), length-0.1),
            step=0.1, key=f'{key_base}_x',
            on_change=on_param_change
        )
        y = col2.number_input(
            "Y", 0.1, width-0.1,
            value=min(st.session_state.get(f'{key_base}_y', default_pos[1]), width-0.1),
            step=0.1, key=f'{key_base}_y',
            on_change=on_param_change
        )
        z = col3.number_input(
            "Z", 0.1, height-0.1,
            value=min(st.session_state.get(f'{key_base}_z', default_pos[2]), height-0.1),
            step=0.1, key=f'{key_base}_z',
            on_change=on_param_change
        )
        return [x, y, z]

    with st.sidebar.expander("Position Details", expanded=False):
        st.markdown("**🔴 Noise Source**")
        noise_pos = pos_slider("Noise", "pos_noise_source", preset['positions']['noise_source'], params['dimensions'])

        st.markdown("**🔵 Reference Mic**")
        ref_pos = pos_slider("Ref", "pos_reference_mic", preset['positions']['reference_mic'], params['dimensions'])

        st.markdown("**🟢 Speaker**")
        spk_pos = pos_slider("Spk", "pos_speaker", preset['positions']['speaker'], params['dimensions'])

        st.markdown("**🟣 Error Mic (Ear)**")
        err_pos = pos_slider("Err", "pos_error_mic", preset['positions']['error_mic'], params['dimensions'])

        params['positions'] = {
            'noise_source': noise_pos,
            'reference_mic': ref_pos,
            'speaker': spk_pos,
            'error_mic': err_pos,
        }

    # Use preset positions if expander not opened
    if 'positions' not in params:
        params['positions'] = {
            'noise_source': [
                st.session_state.get('pos_noise_source_x', preset['positions']['noise_source'][0]),
                st.session_state.get('pos_noise_source_y', preset['positions']['noise_source'][1]),
                st.session_state.get('pos_noise_source_z', preset['positions']['noise_source'][2]),
            ],
            'reference_mic': [
                st.session_state.get('pos_reference_mic_x', preset['positions']['reference_mic'][0]),
                st.session_state.get('pos_reference_mic_y', preset['positions']['reference_mic'][1]),
                st.session_state.get('pos_reference_mic_z', preset['positions']['reference_mic'][2]),
            ],
            'speaker': [
                st.session_state.get('pos_speaker_x', preset['positions']['speaker'][0]),
                st.session_state.get('pos_speaker_y', preset['positions']['speaker'][1]),
                st.session_state.get('pos_speaker_z', preset['positions']['speaker'][2]),
            ],
            'error_mic': [
                st.session_state.get('pos_error_mic_x', preset['positions']['error_mic'][0]),
                st.session_state.get('pos_error_mic_y', preset['positions']['error_mic'][1]),
                st.session_state.get('pos_error_mic_z', preset['positions']['error_mic'][2]),
            ],
        }

    # ==================== Noise Configuration ====================
    st.sidebar.header("🔊 Noise")

    scenario_name = st.sidebar.selectbox(
        "Driving Scenario",
        options=list(SCENARIO_PRESETS.keys()),
        index=list(SCENARIO_PRESETS.keys()).index(st.session_state.get('scenario_preset', DEFAULTS['scenario_preset'])),
        key="scenario_select",
        on_change=on_param_change
    )
    scenario = SCENARIO_PRESETS[scenario_name]
    st.sidebar.caption(scenario['description'])

    params['scenario'] = scenario_name.lower()
    params['noise_mode'] = 'scenario'

    # ==================== FxLMS Parameters ====================
    st.sidebar.header("⚙️ FxLMS Algorithm")

    fxlms_preset = st.sidebar.selectbox(
        "Preset",
        options=list(FXLMS_PRESETS.keys()),
        index=list(FXLMS_PRESETS.keys()).index(st.session_state.get('fxlms_preset', DEFAULTS['fxlms_preset'])),
        key="fxlms_preset_select",
        on_change=on_fxlms_preset_change
    )
    fxlms = FXLMS_PRESETS[fxlms_preset]
    st.sidebar.caption(fxlms['description'])

    params['filter_length'] = st.sidebar.select_slider(
        "Filter Length (taps)",
        options=[64, 128, 192, 256, 320, 384, 512],
        value=st.session_state.get('filter_length_val', fxlms['filter_length']),
        key="filter_length_val",
        on_change=on_param_change
    )

    params['step_size'] = st.sidebar.select_slider(
        "Step Size (μ)",
        options=[0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.05],
        value=st.session_state.get('step_size_val', fxlms['step_size']),
        key="step_size_val",
        format_func=lambda x: f"{x:.4f}",
        on_change=on_param_change
    )

    # ==================== Simulation Settings ====================
    st.sidebar.header("⏱️ Simulation")

    params['duration'] = st.sidebar.slider(
        "Duration (seconds)",
        2.0, 10.0,
        value=st.session_state.get('duration_val', DEFAULTS['duration']),
        step=0.5,
        key="duration_val",
        on_change=on_param_change
    )

    params['sample_rate'] = 16000  # Fixed

    return params


def render_run_button() -> bool:
    """
    Render the Run Simulation button.

    Returns:
        True if button was clicked
    """
    clicked = st.sidebar.button(
        "🚀 Run Simulation",
        type="primary",
        use_container_width=True
    )

    if clicked:
        st.session_state.params_changed = False

    return clicked
