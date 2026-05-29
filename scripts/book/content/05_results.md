## Headline numbers

The five algorithm variants were evaluated on the three real driving scenarios from Chapter 3. Noise reduction is measured at the driver-ear position (or, for the multi-error configurations, averaged across the four head-zone microphones), as 10·log10 of the ratio of desired-signal power to residual error power, evaluated over the second half of the simulation. Convergence time is the wall-clock time at which the windowed error reaches 90 % of its final attenuation.

| Algorithm | Idle (dB) | Cruise (dB) | Accel (dB) | Conv. (s, idle) |
|---|---:|---:|---:|---:|
| SISO FxNLMS | +1.59 | +0.81 | +0.96 | 1.16 |
| Pseudo-SIMO | +5.32 | +4.10 | +4.21 | 0.25 |
| Stage 1 SIMO | +6.15 | +4.12 | +5.05 | 0.98 |
| Stage 2 SIMO+multi-err | +5.42 | +2.76 | +4.51 | 1.41 |
| Stage 3 Full MIMO | +14.33 | +12.64 | +9.89 | 1.41 |

![Five-algorithm × three-scenario performance comparison](output/plots/scenario_comparison_table.png)

Three observations. First, every multi-channel variant beats SISO by several decibels — adding loudspeakers helps regardless of how the filter is structured. Second, the move from pseudo-SIMO to Stage 1 SIMO is small (≈+1 dB); most of the multi-loudspeaker gain comes from simply having more loudspeakers. Third, the move from Stage 2 to Stage 3 (adding multi-reference inputs) is large — +9 dB on idle, +10 dB on cruising — the dominant benefit of MIMO is not the multi-error formulation but the multi-reference inputs that let the controller exploit different propagation paths from the noise source. A separate position-optimisation study confirmed that placing the reference microphone close to the noise source and the loudspeaker close to the listener contributes a further +6 to +8 dB on top of any controller, swamping any algorithmic gain.

## Spatial structure of the cancellation field

A point measurement at the driver ear is informative but does not reveal what happens elsewhere. We compute, for each algorithm, a 5 × 5 cm grid of evaluation microphones at driver-ear height across the entire 4.5 × 1.85 m cabin and run the controllers with frozen weights to measure the noise reduction at every grid point.

![Full-cabin cancellation patterns for all five algorithms (la_varying.wav, 5 × 5 cm grid)](output/plots/cancellation_heatmap_1x5_cabin.png)

The heatmaps show the central trade-off of ANC. Around the error microphones every algorithm produces a deep zone of quiet — Stage 3 reaches more than 25 dB attenuation locally — but outside that zone the cancellation field is far from uniform. SISO has a sharply localised quiet point and amplifies noise (red regions) over roughly three quarters of the cabin. Pseudo-SIMO smears the quiet zone wider but also amplifies less. Stage 2 produces a uniform head-zone quiet zone but pays for it with deeper amplification elsewhere — the strict waterbed trade-off. Stage 3 has a similar amplification footprint to Stage 1 outside the head zone: more degrees of freedom enable deeper local cancellation but cannot eliminate the redistribution of acoustic energy.

For the SISO controller the quiet zone collapses to ≤+5 dB within five centimetres of the error microphone — consistent with the classical result that single-error ANC produces a quiet sphere of radius approximately one tenth of a wavelength at the dominant frequency. The multi-error configurations of Stage 2 and Stage 3 maintain >+10 dB attenuation across a 10 cm head-zone radius before degrading, which is the practically-useful regime: a listener can move their head freely within the head-rest area without losing the cancellation.

## Limits and known caveats

The simulator captures geometry, image-source reflections up to order 3, and frequency-dependent surface absorption, but it does not model loudspeaker non-linearities, microphone self-noise, transducer back-coupling, or the complex acoustics of a real upholstered cabin. The 5 % secondary-path estimation error is a lower bound on what offline identification can achieve; in real deployment the secondary path drifts with temperature, occupant load and seat configuration, and would need to be re-identified online. The performance numbers above should therefore be read as upper bounds attainable in a controlled environment, not as predictions of what would happen in a production vehicle without further engineering.
