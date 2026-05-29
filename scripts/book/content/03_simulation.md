## Simulation environment

Experiments are conducted in software using pyroomacoustics, which implements the image-source method for shoebox geometries with frequency-dependent surface absorption and air absorption. The simulator returns sample-accurate room impulse responses between every source–microphone pair. We build the primary, secondary and reference acoustic paths from these impulse responses and feed them to the FxLMS controllers.

## Cabin geometry

The cabin is modelled as a 4.5 × 1.85 × 1.2 m shoebox at fs = 16 kHz, with image-source reflections up to order 3 and air absorption enabled. Surface absorption is tuned per surface to approximate a typical car interior — the carpeted floor and headliner ceiling are highly absorbent, the windscreen and side glass are highly reflective, and the dashboard and rear shelf sit between. The geometry is intentionally simple: the algorithmic phenomena we want to characterise are insensitive to fine geometry detail and far more sensitive to the placement of the transducers within the volume.

![Simulated cabin geometry showing source, reference mics, loudspeakers and head-zone error mics](output/plots/book/cabin_layout.png)

The noise source sits in the engine bay (front-left). Four candidate reference microphones are evaluated — firewall, floor, A-pillar and dashboard — with the firewall location used by default in the multi-reference Stage 3 controller. Four loudspeakers are mounted in the door and dashboard locations typical of production cars. The error microphones form a 2×2 head-zone grid centred on the driver-ear position with ±5 cm offsets in y and z.

## Excitation signals

The simulator is excited with real recordings from a driving session in downtown Los Angeles, segmented into three representative scenarios:

| Scenario | Source recording | Duration | Character |
|---|---|---|---|
| Idle | la_idle.wav | 20 s | Stationary engine, dominant low-frequency harmonics |
| Cruising | la_medium_cruise.wav | 20 s | Steady tyre / wind noise, broadband |
| Acceleration | la_varying.wav | 20 s | Non-stationary spectrum, transients |

Real recordings are deliberately chosen over synthetic colour-noise because the spectral non-stationarity of road traffic — particularly the acceleration scenario — is the regime where FxLMS is most challenged and where the differences between algorithm variants are most pronounced.

## Default controller parameters

Unless otherwise stated, all FxLMS variants use the following parameters:

| Parameter | Default | Stage 3 override |
|---|---|---|
| Sample rate fs | 16 kHz | 16 kHz |
| Filter length L | 512 taps | 256 taps |
| Step size μ | 0.003 | 0.001 |
| Regularisation δ | 1e−4 | 1e−4 |
| Secondary-path estimate error | 5 % multiplicative noise | 5 % multiplicative noise |
| Head-zone radius | ±5 cm in y and z | ±5 cm in y and z |

The Stage 3 step size is reduced because its weight tensor is four times larger than Stage 2 — its effective gain at the error microphones is therefore higher, and aggressive step sizes that work for Stage 1 cause Stage 3 to diverge.
