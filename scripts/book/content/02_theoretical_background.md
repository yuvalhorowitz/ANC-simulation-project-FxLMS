## The cancellation principle

Active Noise Control exploits the linearity of acoustic wave propagation: two coherent disturbances of equal magnitude and opposite phase sum to silence at the point where they meet. ANC implements this electronically — a microphone captures the disturbance early, an electronic filter computes an inverted copy, and a loudspeaker radiates the anti-noise so that destructive interference takes place at a chosen listening point. Cancellation is local: the geometry of microphones and loudspeakers has a first-order effect on achievable performance.

## The standard LMS adaptive filter

For an FIR filter of length L driven by a reference signal x(n) with target d(n), the standard LMS update minimises the instantaneous squared error by gradient descent on the weights. Three quantities are computed at each sample n:

    Filter output         (1)    y(n) = Σ_{k=0..L−1} wₖ(n) · x(n−k)
    Error signal          (2)    e(n) = d(n) − y(n)
    Weight update         (3)    w(n+1) = w(n) + μ · e(n) · x(n)

Here wₖ(n) is the k-th filter tap at time n and x(n−k) is the reference delayed by k samples; equation (1) is the standard FIR convolution. Equation (3) updates each tap in proportion to the error e(n) and the current reference x(n), with μ controlling adaptation speed and steady-state misadjustment.

## Why LMS alone fails for ANC

In an ANC system the filter output y(n) does not reach the error microphone directly — it must travel through the DAC, the amplifier, the loudspeaker and the acoustic path from the loudspeaker to the error microphone. We call this combined transfer function the secondary path S(z). The disturbance reaches the error microphone through a different primary path P(z). Plain LMS does not account for the phase shift introduced by S(z) and quickly becomes unstable for any non-trivial step size.

## The Filtered-x LMS solution

FxLMS (Burgess 1981, Morgan 1980) fixes the stability problem with a single change: before being used in the weight update, the reference signal is filtered through an estimate Ŝ(z) of the secondary path. The two equations of FxLMS are:

    Filtered reference    (4)    x_f(n) = Σ_{k=0..L−1} ŝₖ · x(n−k)
    Weight update         (5)    w(n+1) = w(n) + μ · e(n) · x_f(n)

The estimate Ŝ(z) is identified offline by playing a known stimulus through the loudspeaker. We model this estimate as the true acoustic path corrupted by 5 % multiplicative noise, representative of what offline identification typically achieves.

## Filtered-x normalised LMS

Plain FxLMS is sensitive to input power — the effective step size scales with |x|² and is hard to keep stable across scenarios. The normalised variant divides each update by the energy of the filtered-reference buffer:

    Weight update         (6)    w(n+1) = w(n) + μ · e(n) · x_f(n) / (δ + ‖x_f(n)‖²)

with δ a small regularisation constant. FxNLMS is the controller used throughout this project. The optional leakage term (1 − μγ) on w(n) prevents weight drift in long-running deployments and is enabled in our implementation with γ = 0.

## The waterbed effect

ANC is fundamentally local. Pressing one part of the acoustic field down lifts another, because cancelling at one point in a reflective enclosure necessarily redistributes acoustic energy elsewhere. This is the waterbed effect, visible in the spatial heatmaps of Chapter 5: a small zone of quiet around the error microphone is paired with regions of amplification (sometimes >5 dB) elsewhere. Multi-error control mitigates the effect within an extended head zone but cannot eliminate it; only adding more loudspeakers genuinely changes the trade-off.

## Multi-channel extensions

The SISO FxNLMS controller extends straightforwardly to multi-channel cases. With M loudspeakers, N reference microphones xₙ(n) and K error microphones eₖ(n), the weights of loudspeaker m driven by reference n update as:

    Multi-channel update  (7)    w_{m,n}(n+1) = w_{m,n}(n) + (μ / norm) · Σₖ eₖ(n) · x_f^{m,n,k}(n)

where norm = δ + Σₖ ‖x_f^{m,n,k}(n)‖² and x_f^{m,n,k}(n) is the n-th reference filtered through the secondary-path estimate from loudspeaker m to error microphone k. The cost function is the sum of instantaneous squared errors across all K error microphones. Stage 1 (M=4, K=1, N=1), Stage 2 (M=4, K=4, N=1) and Stage 3 (M=4, K=4, N=4) of this project all follow this template. The matrix of loudspeaker-to-error secondary paths is the cross-coupling matrix; its size grows as M · K and dominates the per-sample compute cost.
