from manim import *
import numpy as np

class DataDrivenANC(Scene):
    def construct(self):
        # ==========================================
        # 1. LOAD SIMULATION DATA
        # ==========================================
        try:
            data = np.load('anc_data.npz')
            error_log = data['error']      # The error signal over time
            weight_log = data['weights']   # The filter weights over time
            
            # Calculate max error for graph scaling
            max_error = np.max(np.abs(error_log))
        except FileNotFoundError:
            print("ERROR: 'anc_data.npz' not found. Run 'generate_manim_data.py' first.")
            return

        # ==========================================
        # 2. SETUP THE VISUAL LAYOUT
        # ==========================================
        
        # --- CAR DIAGRAM (LEFT) ---
        car_rect = RoundedRectangle(corner_radius=0.5, height=4.5, width=2.8, color=GREY_C, stroke_width=4)
        car_rect.to_edge(LEFT, buff=1.0)
        
        # Positions relative to car center
        car_center = car_rect.get_center()
        pos_engine = car_center + UP * 1.5
        pos_ear = car_center + DOWN * 1.0
        
        # 4 Speakers (Corners)
        speakers_pos = [
            car_center + UP * 0.8 + LEFT * 0.8,  # FL
            car_center + UP * 0.8 + RIGHT * 0.8, # FR
            car_center + DOWN * 1.5 + LEFT * 0.8, # RL
            car_center + DOWN * 1.5 + RIGHT * 0.8 # RR
        ]
        
        # Dots
        engine_dot = Dot(pos_engine, color=RED, radius=0.2)
        ear_dot = Dot(pos_ear, color=WHITE, radius=0.15)
        speaker_dots = VGroup(*[Dot(p, color=GREEN) for p in speakers_pos])
        
        # Labels
        labels = VGroup(
            Text("Engine Noise", font_size=16, color=RED).next_to(engine_dot, UP),
            Text("Driver Ear", font_size=16, color=WHITE).next_to(ear_dot, DOWN),
            Text("ANC Spk", font_size=16, color=GREEN).next_to(speaker_dots[1], RIGHT)
        )

        self.add(car_rect, engine_dot, ear_dot, speaker_dots, labels)

        # --- DASHBOARD (RIGHT) ---
        # 1. Error Graph (Top Right)
        ax_error = Axes(
            x_range=[0, len(error_log), len(error_log)//4], 
            y_range=[-max_error*1.2, max_error*1.2, max_error],
            x_length=6, y_length=2.5,
            axis_config={"include_tip": False, "include_numbers": False}
        ).to_edge(RIGHT, buff=0.5).to_edge(UP, buff=1.0)
        
        err_label = Text("Error Signal (Convergence)", font_size=20).next_to(ax_error, UP)
        
        # 2. Filter Weights (Bottom Right)
        ax_weights = Axes(
            x_range=[0, 4, 1], 
            y_range=[0, 0.5, 0.25],
            x_length=6, y_length=2.5,
            axis_config={"include_numbers": False}
        ).next_to(ax_error, DOWN, buff=1.5)
        
        w_label = Text("Filter Weights Magnitude (Learning)", font_size=20).next_to(ax_weights, UP)

        self.play(Create(ax_error), Write(err_label), Create(ax_weights), Write(w_label))

        # ==========================================
        # 3. ANIMATION LOGIC
        # ==========================================
        
        # The main timer: 0 to 1 (Simulation Progress)
        progress = ValueTracker(0)

        # --- A. Dynamic Waves (Physics Visualization) ---
        
        # Engine: Always pulsing based on progress
        engine_waves = always_redraw(lambda: VGroup(*[
            Circle(
                radius=(progress.get_value()*30 - i*0.5) % 3.0,
                color=RED,
                stroke_width=2,
                stroke_opacity=max(0, 1 - ((progress.get_value()*30 - i*0.5)%3.0)/3.0)
            ).move_to(pos_engine)
            for i in range(3)
        ]))

        # Speakers: Amplitude depends on the ACTUAL WEIGHTS from data
        def get_current_weight_power():
            # Find index in weight log based on progress
            idx = int(progress.get_value() * (len(weight_log)-1))
            idx = np.clip(idx, 0, len(weight_log)-1)
            # Average weight magnitude at this moment
            return np.mean(np.abs(weight_log[idx]))

        speaker_waves = always_redraw(lambda: VGroup(*[
            Circle(
                radius=(progress.get_value()*30 - i*0.5) % 2.0,
                color=GREEN,
                stroke_width=2,
                # Opacity is directly tied to the learned weights!
                stroke_opacity=max(0, 1 - ((progress.get_value()*30 - i*0.5)%2.0)/2.0) * (get_current_weight_power() * 3)
            ).move_to(p)
            for i in range(2) for p in speakers_pos
        ]))

        # --- B. Dynamic Graphs (Data Visualization) ---

        # Draw the error line up to the current point
        # We perform simple downsampling for performance
        ds_factor = 50 
        downsampled_error = error_log[::ds_factor]
        
        # FIX: Ensure we never pass an empty array to plot_line_graph
        graph_line = always_redraw(lambda: ax_error.plot_line_graph(
            x_values=np.arange(len(downsampled_error))[:max(2, int(progress.get_value()*len(downsampled_error)))] * ds_factor,
            y_values=downsampled_error[:max(2, int(progress.get_value()*len(downsampled_error)))],
            line_color=YELLOW,
            add_vertex_dots=False,
            stroke_width=2
        ))
        
        # Draw the weight bars growing
        def get_bars():
            idx = int(progress.get_value() * (len(weight_log)-1))
            current_W = weight_log[idx] # Shape (4, 64)
            # Get energy of each of the 4 filters
            energies = np.mean(np.abs(current_W), axis=1)
            
            bars = VGroup()
            for i, energy in enumerate(energies):
                # Scale height for visibility
                bar = Rectangle(
                    height=max(0.01, energy * 10), 
                    width=0.8, 
                    color=GREEN, 
                    fill_opacity=0.8
                )
                # Position bar on axis
                bar.move_to(ax_weights.c2p(i + 0.5, 0), aligned_edge=DOWN)
                bars.add(bar)
            return bars

        weight_bars = always_redraw(get_bars)

        # ==========================================
        # 4. RUN SCENE
        # ==========================================
        
        self.add(engine_waves, speaker_waves, graph_line, weight_bars)
        
        # Status Text
        status = Text("System Status: Adapting...", font_size=24, color=YELLOW).to_corner(UL)
        self.add(status)

        # Animate progress from 0 to 1 over 8 seconds
        self.play(progress.animate.set_value(1), run_time=8, rate_func=linear)
        
        # Final Success State
        self.remove(status)
        final_status = Text("System Status: Optimized", font_size=24, color=GREEN).to_corner(UL)
        
        # Draw Silence Zone
        silence_ring = Circle(radius=0.5, color=BLUE, fill_opacity=0.3).move_to(pos_ear)
        
        self.play(FadeIn(final_status), FadeIn(silence_ring))
        self.wait(2)