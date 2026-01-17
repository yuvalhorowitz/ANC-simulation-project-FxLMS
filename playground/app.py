"""
ANC Simulation Playground

Interactive GUI for testing Active Noise Cancellation simulations
with real-time parameter control and visualization.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.sidebar import render_sidebar, render_run_button
from components.plots import (
    plot_before_after, plot_spectrum,
    plot_convergence, plot_filter_coefficients
)
from components.audio_player import render_audio_player
from components.room_diagram import plot_room_diagram
from components.room_interactive import render_interactive_room, create_interactive_room_diagram
from simulation.runner import run_simulation, validate_positions


# Page configuration
st.set_page_config(
    page_title="ANC Playground",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<div class="main-header">ANC Simulation Playground</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive Active Noise Cancellation testing with FxLMS algorithm</div>', unsafe_allow_html=True)

    # Get parameters from sidebar
    params = render_sidebar()

    # Run button in sidebar
    run_clicked = render_run_button()

    # Main content area
    if run_clicked:
        # Use interactive positions if available
        if 'interactive_positions' in st.session_state:
            params['positions'] = st.session_state.interactive_positions

        # Validate positions
        is_valid, error_msg = validate_positions(params)
        if not is_valid:
            st.error(f"Invalid configuration: {error_msg}")
            return

        # Run simulation with progress bar
        progress_bar = st.progress(0, text="Initializing simulation...")

        def update_progress(progress, mse):
            progress_bar.progress(progress, text=f"Running... MSE: {mse:.6f}")

        with st.spinner("Running ANC simulation..."):
            results = run_simulation(params, progress_callback=update_progress)

        progress_bar.empty()

        if not results['success']:
            st.error(f"Simulation failed: {results['error_message']}")
            return

        # Store results in session state
        st.session_state.results = results
        st.session_state.params = params
        st.success(f"Simulation complete! Noise reduction: **{results['noise_reduction_db']:.1f} dB**")

    # Display results if available
    if 'results' in st.session_state:
        results = st.session_state.results
        params = st.session_state.params

        # Button to go back to configuration mode
        col_back, col_spacer = st.columns([1, 4])
        with col_back:
            if st.button("⚙️ Change Configuration", use_container_width=True):
                del st.session_state.results
                del st.session_state.params
                # Keep interactive_positions so user can continue editing from current state
                st.rerun()

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Noise Reduction",
                f"{results['noise_reduction_db']:.1f} dB",
                delta=None if results['noise_reduction_db'] < 5 else "Good" if results['noise_reduction_db'] < 15 else "Excellent"
            )

        with col2:
            final_mse = results['mse'][-1] if len(results['mse']) > 0 else 0
            st.metric("Final MSE", f"{final_mse:.2e}")

        with col3:
            st.metric("Filter Taps", f"{len(results['weights'])}")

        with col4:
            st.metric("Duration", f"{results['duration']:.1f} s")

        st.divider()

        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Visualizations", "🔊 Audio", "🏠 Room Layout"])

        with tab1:
            # Visualization plots in 2x2 grid
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Before vs After")
                fig1 = plot_before_after(results)
                st.pyplot(fig1)

                st.subheader("Convergence")
                fig3 = plot_convergence(results)
                st.pyplot(fig3)

            with col2:
                st.subheader("Frequency Spectrum")
                fig2 = plot_spectrum(results)
                st.pyplot(fig2)

                st.subheader("Filter Coefficients")
                fig4 = plot_filter_coefficients(results)
                st.pyplot(fig4)

        with tab2:
            render_audio_player(results)

            st.divider()

            # Audio analysis info
            st.subheader("Audio Analysis")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Original Noise**")
                st.markdown(f"- RMS Level: {np.sqrt(np.mean(results['desired']**2)):.4f}")
                st.markdown(f"- Peak Level: {np.max(np.abs(results['desired'])):.4f}")

            with col2:
                st.markdown("**With ANC**")
                st.markdown(f"- RMS Level: {np.sqrt(np.mean(results['error']**2)):.4f}")
                st.markdown(f"- Peak Level: {np.max(np.abs(results['error'])):.4f}")

        with tab3:
            st.subheader("Room Configuration")

            # Check if in multi-speaker mode
            is_multi_speaker = params.get('speaker_mode') == '4-Speaker System'
            speakers_4ch = params.get('speakers') if is_multi_speaker else None

            col1, col2 = st.columns([2, 1])

            with col1:
                # Use interactive Plotly diagram (pass speakers_4ch for multi-speaker mode)
                fig_room = create_interactive_room_diagram(
                    params['dimensions'],
                    params['positions'],
                    speakers_4ch=speakers_4ch
                )
                st.plotly_chart(fig_room, use_container_width=True)

            with col2:
                st.markdown("**Room Dimensions**")
                st.markdown(f"- Length: {params['dimensions'][0]:.1f} m")
                st.markdown(f"- Width: {params['dimensions'][1]:.1f} m")
                st.markdown(f"- Height: {params['dimensions'][2]:.1f} m")

                st.markdown("**Acoustic Properties**")
                st.markdown(f"- Absorption: {params['absorption']:.2f}")
                st.markdown(f"- Reflection Order: {params['max_order']}")

                st.markdown("**Positions**")
                for name, pos in params['positions'].items():
                    # Skip speaker in multi-speaker mode
                    if name == 'speaker' and is_multi_speaker:
                        continue
                    st.markdown(f"- {name.replace('_', ' ').title()}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

                # Show 4-speaker positions if in multi-speaker mode
                if is_multi_speaker and speakers_4ch:
                    st.markdown("**4 Speakers**")
                    for name, pos in speakers_4ch.items():
                        st.markdown(f"- {name.replace('_', ' ').title()}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

    else:
        # Initial state - show interactive room diagram
        st.info("👈 Adjust parameters in the sidebar and click **Run Simulation** to start. You can also drag components on the diagram below.")

        # Interactive room layout
        updated_positions = render_interactive_room(params)

        # Update params with any position changes from the interactive diagram
        if updated_positions != params['positions']:
            params['positions'] = updated_positions

        # Show parameter summary
        with st.expander("Current Configuration", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Room**")
                st.markdown(f"- Size: {params['dimensions'][0]:.1f} × {params['dimensions'][1]:.1f} × {params['dimensions'][2]:.1f} m")
                st.markdown(f"- Absorption: {params['absorption']:.2f}")

            with col2:
                st.markdown("**FxLMS**")
                st.markdown(f"- Filter: {params['filter_length']} taps")
                st.markdown(f"- Step size: {params['step_size']:.4f}")

            with col3:
                st.markdown("**Simulation**")
                st.markdown(f"- Duration: {params['duration']:.1f} s")
                st.markdown(f"- Sample rate: {params['sample_rate']} Hz")
                st.markdown(f"- Speaker mode: {params.get('speaker_mode', 'Single Speaker')}")


if __name__ == "__main__":
    main()
