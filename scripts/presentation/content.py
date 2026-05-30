"""
Slide specifications for the Final Presentation.

Each entry is a dict describing one slide. The builder (build_presentation.py)
turns these into OOXML slide parts. Prose is editable here without touching
the pptx-generation logic.

Slide `kind` values:
  - 'title'        : title slide (project name / number / students / supervisor)
  - 'bullets'      : title + bulleted body
  - 'image'        : title + single centered image (+ optional caption)
  - 'image_bullets': title + bullets on the left, image on the right
  - 'two_image'    : title + two images side by side (+ optional sub-captions)
  - 'table'        : title + a table (rows of cells; first row = header)
  - 'demo'         : title + image + two embedded audio clips (before / after)

Image paths are relative to the project root, OR one of the special tokens
'@playground' / '@beforeafter' / '@logo' which the builder resolves from the
cloned mid-presentation media.

`section` is the required-section number (1-8) for traceability; the builder
renders it as a small tag.
"""

TITLE = {
    'kind': 'title',
    'project_title': 'Active Noise Reduction System In a Vehicle',
    'subtitle': 'FxLMS-Based Multi-Channel ANC for the Car Cabin — '
                'from SISO to Full MIMO',
    'project_number': '25-1-1-3214',
    'students': 'Ariel Turnowski (206483513)   ·   Yuval Horowitz (206587719)',
    'supervisor': 'Supervisor: Dr. Lior Arbel',
    'location': 'The Iby and Aladar Fleischman Faculty of Engineering, '
                'Tel Aviv University',
    'footer': 'Final Presentation',
}

SLIDES = [
    TITLE,

    # ---- Section 2: project topic / abstract ----
    {
        'kind': 'image_bullets',
        'section': 2,
        'title': 'What is Active Noise Control?',
        'bullets': [
            ('A microphone captures noise early; a filter computes an inverted '
             'copy; a loudspeaker radiates it so the two cancel at the ear.', 0),
            ('Targets the low-frequency cabin noise (engine, road, wind) that '
             'passive insulation cannot economically absorb.', 0),
            ('We build and evaluate the full system in simulation, driven by '
             'real Los-Angeles driving recordings.', 0),
            ('Headline result: a full MIMO controller reaches +14.3 dB on idle, '
             '+12.6 dB on cruising, +9.9 dB on acceleration.', 0),
        ],
        'image': 'output/plots/book/block_diagram.png',
    },

    {
        'kind': 'bullets',
        'section': 2,
        'title': 'Background — the Cancellation Principle',
        'bullets': [
            ('Two coherent sound waves of equal magnitude and opposite phase '
             'sum to silence at the point where they meet.', 0),
            ('Cancellation is local: it only holds where the anti-noise is '
             'phase-matched to the disturbance.', 0),
            ('The reference signal is filtered and continuously adapted so the '
             'residual error at the ear is driven toward zero.', 0),
            ('Two acoustic paths matter:', 0),
            ('Primary path P(z): noise source → error microphone.', 1),
            ('Secondary path S(z): loudspeaker → error microphone (the part '
             'that makes naive LMS unstable).', 1),
        ],
    },

    # ---- Section 3: engineering motivation and goals ----
    {
        'kind': 'bullets',
        'section': 3,
        'title': 'Engineering Motivation',
        'bullets': [
            ('Sits at the intersection of Digital Signal Processing, acoustics '
             'and machine learning.', 0),
            ('Improves passenger comfort and reduces driver fatigue on long '
             'drives.', 0),
            ('Low-frequency noise (50–300 Hz) is exactly where adding mass / '
             'insulation gives the least return.', 0),
            ('Unlike headphones or cockpit headsets, a car needs a broad zone '
             'of quiet that tolerates natural head movement — which demands a '
             'multi-channel adaptive system.', 0),
        ],
    },

    {
        'kind': 'table',
        'section': 3,
        'title': 'Project Goals & Targets',
        'table': [
            ['Goal', 'Target'],
            ['Noise reduction at driver ear', '≥ 10 dB (stretch 10–20 dB)'],
            ['Frequency range', '20 – 1000 Hz'],
            ['Convergence time', '< 2 seconds'],
            ['Stability', 'across idle / cruising / acceleration'],
            ['Configurations', 'SISO → SIMO → full MIMO (multi-speaker / mic)'],
        ],
        'note': 'Evaluated on real recordings, not synthetic noise — the '
                'harder and more honest test.',
    },

    # ---- Section 4: methods and implementation ----
    {
        'kind': 'bullets',
        'section': 4,
        'title': 'Methods — System Architecture',
        'bullets': [
            ('Acoustic simulation: pyroomacoustics shoebox cabin '
             '(4.5 × 1.85 × 1.2 m), image-source reflections, '
             'frequency-dependent absorption.', 0),
            ('Controller: Filtered-x Normalised LMS (FxNLMS) adaptive filter — '
             'the de-facto ANC standard.', 0),
            ('Scaled across five configurations of growing channel count, '
             'SISO through full 4×4×4 MIMO.', 0),
            ('Two ML research threads explored in parallel (step-size selection '
             'and an end-to-end neural controller).', 0),
            ('Everything wrapped in an interactive Streamlit playground for '
             'live experimentation.', 0),
        ],
    },

    {
        'kind': 'image_bullets',
        'section': 4,
        'title': 'Simulation Environment & Cabin Geometry',
        'bullets': [
            ('Shoebox cabin at 16 kHz, reflections to order 3, per-surface '
             'absorption (carpet, headliner, glass, dashboard).', 0),
            ('Noise source in the engine bay; 4 candidate reference mics '
             '(firewall, floor, A-pillar, dashboard).', 0),
            ('4 loudspeakers in door / dashboard positions.', 0),
            ('4 error mics in a 2×2 head-zone grid (±5 cm) around the '
             'driver ear.', 0),
        ],
        'image': 'output/plots/book/cabin_layout.png',
    },

    {
        'kind': 'table',
        'section': 4,
        'title': 'Excitation — Real Driving Scenarios',
        'table': [
            ['Scenario', 'Recording', 'Character'],
            ['Idle', 'la_idle.wav', 'Stationary engine, low-frequency harmonics'],
            ['Cruising', 'la_medium_cruise.wav', 'Steady tyre / wind, broadband'],
            ['Acceleration', 'la_varying.wav', 'Non-stationary spectrum, transients'],
        ],
        'note': 'Real Los-Angeles downtown driving — the acceleration clip is '
                'the regime where FxLMS is most challenged.',
    },

    {
        'kind': 'bullets',
        'section': 4,
        'title': 'FxLMS → FxNLMS — the Core Algorithm',
        'bullets': [
            ('FxLMS filters the reference through an estimate of the secondary '
             'path before the weight update — this is what keeps it stable.', 0),
            ('Update rule (normalised form):', 0),
            ('w(n+1) = w(n) + μ · e(n) · x′(n) / (δ + ‖x′(n)‖²)', 1),
            ('w — filter weights;  μ — step size;  e(n) — error at the mic.', 1),
            ('x′(n) — reference filtered through the secondary-path estimate;  '
             'δ — regularisation.', 1),
            ('FxLMS adapts ~16,000 times per second (one update per sample at '
             '16 kHz).', 0),
        ],
        'params': [
            ['Parameter', 'Value'],
            ['Filter length', '512 taps (256 for MIMO)'],
            ['Step size μ', '0.003 (0.001 for MIMO)'],
            ['Regularisation δ', '1e-4'],
            ['Sample rate', '16 kHz'],
        ],
    },

    {
        'kind': 'bullets',
        'section': 4,
        'title': 'From SISO to MIMO — the Taxonomy',
        'bullets': [
            ('SISO (1×1×1): one ref, one speaker, one error mic — small zone of '
             'quiet, baseline.', 0),
            ('Pseudo-SIMO (1×4×1): four speakers driven by ONE shared filter — '
             'cheap, but limited.', 0),
            ('Stage 1 — true SIMO (1×4×1): four independent filters.', 0),
            ('Stage 2 — SIMO + multi-error (1×4×4): four head-zone error mics → '
             'uniform quiet zone.', 0),
            ('Stage 3 — full MIMO (4×4×4): four reference mics exploit different '
             'noise paths → biggest gain.', 0),
            ('More channels = more degrees of freedom = better cancellation '
             '(at higher compute cost).', 0),
        ],
    },

    {
        'kind': 'two_image',
        'section': 4,
        'title': 'Implementation — SISO vs Pseudo-SIMO',
        'image_left': 'output/plots/heatmaps_la_varying/heatmap_SISO.png',
        'image_right': 'output/plots/heatmaps_la_varying/heatmap_Pseudo-SIMO.png',
        'caption_left': 'SISO — sharp, tiny quiet point',
        'caption_right': 'Pseudo-SIMO — wider, shared filter',
    },

    {
        'kind': 'two_image',
        'section': 4,
        'title': 'Implementation — Stage 1 SIMO vs Stage 2 Multi-Error',
        'image_left': 'output/plots/heatmaps_la_varying/heatmap_Stage_1_SIMO.png',
        'image_right': 'output/plots/heatmaps_la_varying/heatmap_Stage_2_SIMOpmulti-err.png',
        'caption_left': 'Stage 1 — independent filters',
        'caption_right': 'Stage 2 — uniform head-zone quiet',
    },

    {
        'kind': 'image_bullets',
        'section': 4,
        'title': 'Implementation — Stage 3 Full MIMO (4×4×4)',
        'bullets': [
            ('Four reference mics feed a 4×4×256 = 4,096-weight controller.', 0),
            ('Each reference captures a different noise path — engine via '
             'firewall, road via floor, wind via A-pillar.', 0),
            ('Step size reduced to 0.001 to stay stable with 4× more weights.', 0),
            ('Best result of all configurations — but ~13× real-time compute.', 0),
        ],
        'image': 'output/plots/heatmaps_la_varying/heatmap_Stage_3_Full_MIMO.png',
    },

    {
        'kind': 'image_bullets',
        'section': 4,
        'title': 'ML Thread A — Adaptive Step-Size Selection',
        'bullets': [
            ('Idea: a small classifier picks the FxLMS step size from short-term '
             'audio features.', 0),
            ('Four attempts: synthetic-static (+0.37 dB), multi-channel '
             '(+0.06 dB), dynamic per-segment (−0.12 dB), rolling-sim labels '
             '(−0.07 dB).', 0),
            ('Finding: FxLMS already adapts 16,000×/s — little headroom left for '
             'a 0.5-s-granularity classifier.', 0),
            ('ML only helps when the optimum changes faster than the filter can '
             'self-tune.', 0),
        ],
        'image': 'output/plots/ml_journey.png',
    },

    {
        'kind': 'bullets',
        'section': 4,
        'title': 'ML Thread B — End-to-End Neural Control (TCN)',
        'bullets': [
            ('Idea: replace FxLMS with a Temporal Convolutional Network that '
             'learns anti-noise directly from the reference.', 0),
            ('Causal dilated convolutions; secondary path placed INSIDE the '
             'training loss (the neural filtered-x idea).', 0),
            ('Variants: baseline +2.2 dB; RIR-pre-filtered ≈ +8.9 dB '
             '(controlled); others inconsistent or amplified the noise.', 0),
            ('Failure modes: amplitude overshoot, weak phase alignment, poor '
             'generalisation to unseen cabins.', 0),
            ('Lesson: a black box lacks the microsecond phase/amplitude '
             'precision that FxLMS enforces structurally → we chose physics-'
             'structured MIMO.', 0),
        ],
    },

    # ---- Section 5: demonstration ----
    {
        'kind': 'demo',
        'section': 5,
        'title': 'Demonstration',
        'image': '@playground',
        'bullets': [
            ('Interactive Streamlit playground: switch algorithm, scenario and '
             'transducer positions live.', 0),
            ('Listen to the before / after on the right (embedded audio).', 0),
            ('Live app:  streamlit run playground/app.py', 0),
        ],
        'audio_before': 'output/audio/diagnostic/original.wav',
        'audio_after': 'output/audio/diagnostic/cancelled.wav',
        'label_before': 'Original noise',
        'label_after': 'With ANC',
    },

    # ---- Section 6: results and conclusions ----
    {
        'kind': 'image',
        'section': 6,
        'title': 'Results — Scenario Performance',
        'image': 'output/plots/scenario_comparison_table.png',
        'caption': 'Five algorithms × three scenarios. Stage 3 Full MIMO wins '
                   'every scenario: +14.3 / +12.6 / +9.9 dB.',
    },

    {
        'kind': 'image',
        'section': 6,
        'title': 'Results — Spatial Field & the Waterbed Effect',
        'image': 'output/plots/cancellation_heatmap_1x5_cabin.png',
        'caption': 'Deep quiet zone at the ear, but energy is pushed up '
                   'elsewhere — ANC redistributes, it does not delete.',
    },

    {
        'kind': 'image_bullets',
        'section': 6,
        'title': 'Results — the Zone of Quiet',
        'bullets': [
            ('SISO quiet zone collapses within ~5 cm of the error mic.', 0),
            ('Multi-error Stages 2 & 3 hold >10 dB across a 10 cm head-zone '
             'radius.', 0),
            ('This is the practically useful regime: the listener can move '
             'their head without losing cancellation.', 0),
        ],
        'image': 'output/plots/speaker_distance_vs_reduction.png',
    },

    {
        'kind': 'bullets',
        'section': 6,
        'title': 'Conclusions',
        'bullets': [
            ('All three goals met: Stage 3 reaches +14.3 dB (idle) / +9.9 dB '
             '(acceleration), converging in ~1.4 s.', 0),
            ('More channels genuinely help — multi-reference MIMO is the biggest '
             'single jump in performance.', 0),
            ('Transducer placement matters as much as the algorithm: position '
             'optimisation alone adds 6–8 dB.', 0),
            ('Both ML directions lost to a well-tuned FxLMS — physics structure '
             'beat the black box here.', 0),
        ],
    },

    # ---- Section 7: suggestions for follow-up ----
    {
        'kind': 'bullets',
        'section': 7,
        'title': 'Suggestions for Future Work',
        'bullets': [
            ('Online secondary-path identification — track drift with '
             'temperature, seats and occupants (recovers lost dB).', 0),
            ('Vectorised / GPU MIMO — bring Stage 3 from ~13× real-time into '
             'real-time on a DSP.', 0),
            ('Long-form non-stationary audio — where adaptive step-size '
             'selection should finally pay off.', 0),
            ('Improved neural ANC — cascaded magnitude/phase TCN with RIR '
             'pre-filtering.', 0),
            ('Real-cabin measurements & a hardware port to validate the '
             'simulator.', 0),
        ],
    },

    # ---- Section 8: project documentation ----
    {
        'kind': 'image_bullets',
        'section': 8,
        'title': 'Project Documentation',
        'bullets': [
            ('Public GitHub repository with all code, simulations, plots and '
             'this presentation.', 0),
            ('Project Book (20 pp) documents theory, simulation, results.', 0),
            ('Interactive playground: streamlit run playground/app.py', 0),
            ('Reproducible: every figure has a generator script.', 0),
        ],
        'image': 'output/plots/book/repo_layout.png',
    },

    {
        'kind': 'bullets',
        'section': 8,
        'title': 'Thank You',
        'bullets': [
            ('Active Noise Reduction System In a Vehicle — Project 25-1-1-3214', 0),
            ('Ariel Turnowski   ·   Yuval Horowitz', 0),
            ('Supervisor: Dr. Lior Arbel', 0),
            ('Questions?', 0),
        ],
        'center': True,
    },
]
