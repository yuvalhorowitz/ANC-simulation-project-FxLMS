## Overview

The implementation comprises five FxNLMS controller variants of increasing channel count, plus two parallel machine-learning threads. All controllers share the same simulator, FIR filter front-end and secondary-path estimation methodology; they differ only in the structure of the weight tensor. None of the source code is included in this report; it lives in the public repository documented in Chapter 7.

## SISO FxNLMS (1 × 1 × 1)

The baseline. One reference microphone (driver-side firewall), one dashboard loudspeaker, one error microphone at the driver-ear position. The weight vector has L = 512 taps and is updated sample-by-sample by the standard FxNLMS rule. Its main limitation is that the zone of quiet is small — typically a few centimetres around the error microphone — so any movement of the listener's head out of that zone severely degrades the perceived attenuation.

## Pseudo-SIMO (1 × 4 × 1, shared filter)

Four loudspeakers driven by the same anti-noise signal y(n). From the controller's point of view, the four loudspeakers behave as a single effective loudspeaker whose secondary path is the sum of the four loudspeaker-to-error-microphone paths. The same scalar FxNLMS update is used. This corresponds to the way many low-cost in-cabin systems are wired — one filter saves cost and DSP cycles, but every loudspeaker emits the same signal, constraining the cancellation field.

## Stage 1 — true SIMO (1 × 4 × 1, independent filters)

The first true multi-channel controller. The single reference signal feeds four independent FIR filters Wₘ(z), one per loudspeaker, each updated with its own filtered reference x_f^{m}(n). The cost function remains the squared error at the single driver-ear microphone. Stage 1 beats pseudo-SIMO on 11/13 real recordings (mean +0.54 dB), and degenerates to scalar FxNLMS when M = 1 (verified bit-exact in unit tests).

## Stage 2 — SIMO + multi-error head-zone (1 × 4 × 4)

Stage 1 produces a deep but very narrow zone of quiet. Stage 2 widens that zone with four error microphones in a 2×2 head-zone grid (±5 cm in y and z). The cost function is the sum of squared errors across the four error microphones. The mean head-zone reduction is +4.73 dB (Stage 2) vs +4.32 dB (Stage 1), but the more important property is that the cancellation is uniform across the head zone — per-microphone variance below 0.5 dB — so head movement no longer collapses the perceived attenuation.

## Stage 3 — Full MIMO (4 × 4 × 4)

The richest configuration. Four reference microphones (firewall, floor, A-pillar, dashboard) feed a 4 × 4 × 256 = 4 096-weight controller with four loudspeakers and four head-zone error microphones. The added reference inputs let the controller exploit different propagation paths — engine harmonics through the firewall, road noise through the floor, wind through the A-pillar — instead of relying on a single mixed reference. Because the weight tensor is four times larger than Stage 2 the step size has to be reduced to 0.001 to keep the algorithm stable. Stage 3 reaches +14.3 dB on idle, +12.6 dB on cruising and +9.9 dB on acceleration — a margin of 3–4 dB over Stage 2 on the same recordings.

## ML thread A — Adaptive step-size selection

Four small classifiers were trained to pick the FxLMS step size adaptively from short-term audio features.

![ML journey: four step-size selectors evaluated against a fixed-μ baseline](output/plots/ml_journey.png)

The first attempt (single-channel, synthetic, static classification at t = 0) produced a small +0.37 dB mean improvement. The second retrained the same classifier on the four-loudspeaker configuration and the result collapsed to +0.06 dB — acoustic summing across four loudspeakers flattened the optimal step-size landscape, leaving almost nothing for the classifier to exploit. The third pivoted to real recordings with dynamic per-segment labelling and produced −0.12 dB; the per-segment labelling was wrong because it ignored that FxLMS weights carry across segment boundaries. The fourth fixed the labelling with a rolling-simulation lookahead and reached −0.07 dB — better label distribution, still worse than the fixed-μ baseline.

The fundamental observation is that FxLMS is itself an adaptive algorithm operating at the sample rate — at fs = 16 kHz, the controller updates its weights roughly 16 000 times per second, leaving very little room for a 0.5-second-granularity classifier to add value. Adding a classifier on top to choose the FxLMS step size is a form of meta-learning that only contributes when the optimal hyperparameters change faster than the underlying adaptive algorithm can self-tune; our short, internally-consistent recordings of a single driving condition do not provide such variation.

## ML thread B — Full-ML control with a TCN

A second, more ambitious thread investigated replacing the FxLMS controller entirely with a Temporal Convolutional Network that learns a non-linear mapping from a window of recent reference samples to the next anti-noise sample. The model used causal dilated 1-D convolutions (kernel sizes 3–5, dilation factors 1, 2, 4 …, receptive fields up to several hundred samples to span one full cycle of the engine fundamental near 50 Hz) and a tanh output to constrain the predicted control sample within the loudspeaker's amplitude range.

The crucial design correction was to include the secondary path inside the training loss. The network does not predict the unwanted noise at the ear; it predicts an electrical control signal u(n) that is then convolved with the secondary-path impulse response ĥ_s(n) inside the differentiable training loop. The loss is the residual after acoustic propagation, e(n) = d(n) + (u ∗ ĥ_s)(n) — the neural analogue of the filtered-x idea. Optimising on the raw network output without this physical layer in the loss produced waveforms that looked plausible electrically but amplified the sound field after passing through the loudspeaker.

A sequence of variants was trained: a baseline TCN with secondary-path loss (full-band mean +2.17 dB, engine-band +3.26 dB); a tanh-output amplitude-bounded variant with frequency-weighted loss (collapsed to −9.7 dB; loss shaping alone failed to fix phase); an RIR-pre-filtered variant where the reference was preconditioned with the room impulse response (≈ +8.9 dB in a controlled setting); and a late-stage cascaded magnitude-and-phase TCN, which remained inconsistent on blind scenarios. Dominant failure modes: amplitude overshoot, weak phase correlation, high-frequency amplification accompanying low-frequency cancellation, and a large generalisation gap to unseen reflective cabins. The TCN branch did not become the deployed solution. The diagnostics it produced reveal the headline limitation of black-box ANC: a neural controller lacks the microsecond phase and amplitude precision required for physical destructive interference unless that precision is enforced structurally — through causality (a positive primary-minus-secondary-path delay budget), explicit secondary-path compensation, bounded amplitude, and anti-phase alignment after acoustic propagation. Every one of these is enforced by FxLMS for free, which is why the project's contribution shifted to physics-structured MIMO.
