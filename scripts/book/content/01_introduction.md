## Motivation

Cabin noise inside a moving passenger car is dominated by low-frequency components: engine harmonics in the 50–250 Hz range, tyre/road excitation around 80–300 Hz, and wind buffeting that grows with speed. These are precisely the frequencies that are least attenuated by passive insulation — adding mass to the door panel costs both money and weight while delivering only a few decibels per octave below 250 Hz. Modern vehicles therefore rely on signal-processing solutions to keep low-frequency cabin noise within acceptable limits. The problem sits at the intersection of digital signal processing, acoustics and machine learning, and the engineering pay-off — improved passenger comfort, reduced driver fatigue, intelligibility of in-car communication — is large enough to motivate every premium production cabin shipping today. Unlike noise-cancelling headphones or aircraft cockpit headsets, an in-car solution must produce a broad zone of quiet that accommodates natural head movement at the listening position, which makes the multi-channel adaptive system studied here both necessary and challenging.

## Goal

The objective of this project is to design and evaluate an Active Noise Control system that targets at least 10 dB of broadband noise reduction at the driver-ear position on real driving recordings, with stable convergence under two seconds, and to characterise the trade-offs that arise when scaling from a single-channel controller up to a full multi-input multi-output configuration.

## Approach

The system is built around the Filtered-x LMS family of adaptive filters, the de-facto standard for feed-forward ANC. Above the single-channel baseline we implement four progressively richer multi-channel variants. All experiments run in a pyroomacoustics shoebox cabin model excited with real LA Downtown driving recordings (idle, cruising, acceleration). Two complementary ML threads were also investigated: one that uses a small classifier to pick the FxLMS step size adaptively, and one that replaces the FxLMS controller entirely with a Temporal Convolutional Network. FxLMS itself dates back to Burgess (1981) and Morgan (1980); multi-channel extensions for car cabins are documented by Kuo and Morgan (1996, 1999), and the present project follows their MIMO formulation.

## Structure of this report

Chapter 2 develops FxLMS and its multi-channel extensions formally. Chapter 3 describes the simulator. Chapter 4 walks through the implementation of the five algorithm variants and the ML thread. Chapter 5 reports quantitative and spatial results. Chapter 6 summarises the conclusions. Chapter 7 lists project deliverables and the public repository.
