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
    FXLMS_PRESETS, DEFAULTS, FOUR_SPEAKER_CONFIG, SPEAKER_MODES,
    FOUR_REF_MIC_CONFIG, REF_MIC_MODES, NOISE_SOURCE_POSITIONS,
    SCENARIO_NOISE_POSITIONS
)


def init_session_state():
    """Initialize session state with default values."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.room_preset = DEFAULTS['room_preset']
        st.session_state.scenario_preset = DEFAULTS['scenario_preset']
        st.session_state.fxlms_preset = DEFAULTS['fxlms_preset']
        st.session_state.speaker_mode = 'Single Speaker'  # Default to single speaker

        # Load initial preset values
        preset = ROOM_PRESETS[DEFAULTS['room_preset']]
        st.session_state.dimensions = preset['dimensions'].copy()
        st.session_state.absorption = preset['absorption']
        st.session_state.max_order = preset['max_order']
        st.session_state.positions = {k: v.copy() for k, v in preset['positions'].items()}

        # Override noise source position based on default scenario
        default_scenario = DEFAULTS['scenario_preset']
        noise_pos_key = SCENARIO_NOISE_POSITIONS.get(default_scenario, 'Combined (Dashboard)')
        auto_noise_pos = NOISE_SOURCE_POSITIONS[noise_pos_key]
        st.session_state.positions['noise_source'] = auto_noise_pos.copy()

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

    # Reset interactive positions to preset positions
    st.session_state.interactive_positions = {
        k: v.copy() for k, v in preset['positions'].items()
    }

    # Delete slider keys so they get re-initialized with new values
    for comp_name in preset['positions'].keys():
        for axis in ['x', 'y', 'z']:
            key = f"drag_{comp_name}_{axis}"
            if key in st.session_state:
                del st.session_state[key]

    st.session_state.params_changed = True


def on_fxlms_preset_change():
    """Callback when FxLMS preset changes."""
    preset_name = st.session_state.fxlms_preset_select
    preset = FXLMS_PRESETS[preset_name]

    st.session_state.fxlms_preset = preset_name
    st.session_state.filter_length_val = preset['filter_length']
    st.session_state.step_size_val = preset['step_size']
    st.session_state.params_changed = True


def on_scenario_change():
    """Callback when driving scenario changes - update noise source position."""
    scenario_name = st.session_state.scenario_select

    # Get noise position for this scenario
    noise_pos_key = SCENARIO_NOISE_POSITIONS.get(scenario_name, 'Combined (Dashboard)')
    auto_noise_pos = NOISE_SOURCE_POSITIONS[noise_pos_key]

    # Update noise source position in interactive_positions
    if 'interactive_positions' in st.session_state:
        st.session_state.interactive_positions['noise_source'] = auto_noise_pos.copy()

    # Directly set slider key values to the new position
    # This ensures sliders update immediately
    st.session_state['drag_noise_source_x'] = float(auto_noise_pos[0])
    st.session_state['drag_noise_source_y'] = float(auto_noise_pos[1])
    st.session_state['drag_noise_source_z'] = float(auto_noise_pos[2])

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
    st.sidebar.header("🚗 Car")

    # Room preset selector
    room_preset = st.sidebar.selectbox(
        "Car Type",
        options=list(ROOM_PRESETS.keys()),
        index=list(ROOM_PRESETS.keys()).index(st.session_state.get('room_preset', DEFAULTS['room_preset'])),
        key="room_preset_select",
        on_change=on_room_preset_change
    )

    preset = ROOM_PRESETS[room_preset]

    # Dimensions - use session state with defaults from preset
    st.sidebar.subheader("Car Dimensions")

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
    st.sidebar.subheader("Car Acoustics")
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

    # ==================== Speaker Mode ====================
    st.sidebar.header("🔊 Speaker Setup")
    speaker_mode = st.sidebar.radio(
        "Speaker Mode",
        options=list(SPEAKER_MODES.keys()),
        index=0 if st.session_state.get('speaker_mode', 'Single Speaker') == 'Single Speaker' else 1,
        key="speaker_mode_select",
        on_change=on_param_change,
        help="Single speaker or 4-speaker system"
    )
    st.session_state.speaker_mode = speaker_mode
    params['speaker_mode'] = speaker_mode

    mode_info = SPEAKER_MODES[speaker_mode]
    st.sidebar.caption(mode_info['description'])

    if speaker_mode == '4-Speaker System':
        # Show toggles for each speaker
        st.sidebar.markdown("**Enable/Disable Speakers:**")

        enabled_speakers = {}
        speaker_names = {
            'front_left': 'Front Left (Door)',
            'front_right': 'Front Right (Door)',
            'dash_left': 'Dash Left',
            'dash_right': 'Dash Right',
        }

        col1, col2 = st.sidebar.columns(2)
        for i, (key, label) in enumerate(speaker_names.items()):
            col = col1 if i % 2 == 0 else col2
            # Default all speakers to enabled
            default_enabled = st.session_state.get(f'speaker_{key}_enabled', True)
            enabled = col.checkbox(
                label.split('(')[0].strip(),  # Short label
                value=default_enabled,
                key=f'speaker_{key}_enabled',
                on_change=on_param_change,
                help=label
            )
            if enabled:
                enabled_speakers[key] = FOUR_SPEAKER_CONFIG[key].copy()

        if enabled_speakers:
            params['speakers'] = enabled_speakers
        else:
            st.sidebar.warning("⚠️ Enable at least one speaker")
            params['speakers'] = {'front_left': FOUR_SPEAKER_CONFIG['front_left'].copy()}
    else:
        # Clear 4-speaker config when switching to single speaker
        params['speakers'] = None
        if 'speakers' in st.session_state:
            del st.session_state['speakers']

    # ==================== Reference Mic Mode ====================
    st.sidebar.header("🎤 Reference Mic Setup")
    ref_mic_mode = st.sidebar.radio(
        "Reference Mic Mode",
        options=list(REF_MIC_MODES.keys()),
        index=0 if st.session_state.get('ref_mic_mode', 'Single Reference Mic') == 'Single Reference Mic' else 1,
        key="ref_mic_mode_select",
        on_change=on_param_change,
        help="Single or 4 strategic reference mics"
    )
    st.session_state.ref_mic_mode = ref_mic_mode
    params['ref_mic_mode'] = ref_mic_mode

    mode_info = REF_MIC_MODES[ref_mic_mode]
    st.sidebar.caption(mode_info['description'])

    if ref_mic_mode == '4-Reference Mic System':
        # Show toggles for each reference mic
        st.sidebar.markdown("**Enable/Disable Ref Mics:**")

        enabled_ref_mics = {}
        ref_mic_names = {
            'firewall': 'Firewall (Engine)',
            'floor': 'Floor (Road)',
            'a_pillar': 'A-Pillar (Wind)',
            'dashboard': 'Dashboard (General)',
        }

        col1, col2 = st.sidebar.columns(2)
        for i, (key, label) in enumerate(ref_mic_names.items()):
            col = col1 if i % 2 == 0 else col2
            # Default all ref mics to enabled
            default_enabled = st.session_state.get(f'ref_mic_{key}_enabled', True)
            enabled = col.checkbox(
                label.split('(')[0].strip(),  # Short label
                value=default_enabled,
                key=f'ref_mic_{key}_enabled',
                on_change=on_param_change,
                help=label
            )
            if enabled:
                enabled_ref_mics[key] = FOUR_REF_MIC_CONFIG[key].copy()

        if enabled_ref_mics:
            params['ref_mics'] = enabled_ref_mics
        else:
            st.sidebar.warning("⚠️ Enable at least one ref mic")
            params['ref_mics'] = {'firewall': FOUR_REF_MIC_CONFIG['firewall'].copy()}
    else:
        # Clear 4-ref-mic config when switching to single ref mic
        params['ref_mics'] = None

    # Use interactive positions if available, otherwise use preset
    if 'interactive_positions' in st.session_state:
        params['positions'] = st.session_state.interactive_positions.copy()
    else:
        params['positions'] = {k: v.copy() for k, v in preset['positions'].items()}

    # ==================== Noise Configuration ====================
    st.sidebar.header("🔊 Noise")

    scenario_name = st.sidebar.selectbox(
        "Driving Scenario",
        options=list(SCENARIO_PRESETS.keys()),
        index=list(SCENARIO_PRESETS.keys()).index(st.session_state.get('scenario_preset', DEFAULTS['scenario_preset'])),
        key="scenario_select",
        on_change=on_scenario_change
    )
    scenario = SCENARIO_PRESETS[scenario_name]
    st.sidebar.caption(scenario['description'])

    params['scenario'] = scenario_name.lower()
    params['noise_mode'] = 'scenario'

    # Get scenario-based noise position (for display and initial setup only)
    noise_pos_key = SCENARIO_NOISE_POSITIONS.get(scenario_name, 'Combined (Dashboard)')
    auto_noise_pos = NOISE_SOURCE_POSITIONS[noise_pos_key]

    # Only set noise position on first load (when interactive_positions doesn't exist yet)
    # After that, let the user adjust it manually via sliders
    # Scenario changes are handled by on_scenario_change() callback
    if 'interactive_positions' not in st.session_state:
        # First load - set noise position based on scenario
        params['positions']['noise_source'] = auto_noise_pos.copy()

    st.sidebar.caption(f"📍 Default noise source: {noise_pos_key}")

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
        options=[0.001, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02],
        value=st.session_state.get('step_size_val', fxlms['step_size']),
        key="step_size_val",
        format_func=lambda x: f"{x:.4f}",
        on_change=on_param_change
    )

    # ==================== Simulation Settings ====================
    st.sidebar.header("⏱️ Simulation")

    # Adjust duration limits for Dynamic Ride (needs at least 2s per scenario)
    is_dynamic_ride = scenario_name == 'Dynamic Ride'
    min_duration = 8.0 if is_dynamic_ride else 2.0
    max_duration = 20.0 if is_dynamic_ride else 10.0

    # Clamp current value to valid range
    current_duration = st.session_state.get('duration_val', DEFAULTS['duration'])
    current_duration = max(min_duration, min(max_duration, current_duration))

    params['duration'] = st.sidebar.slider(
        "Duration (seconds)",
        min_value=min_duration,
        max_value=max_duration,
        value=current_duration,
        step=0.5,
        key="duration_val",
        on_change=on_param_change
    )

    # Show segment info for Dynamic Ride
    if is_dynamic_ride:
        segment_duration = params['duration'] / 4
        st.sidebar.info(f"4 scenarios × {segment_duration:.1f}s each")

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
