# FxLMS Algorithm Explained

A simple, step-by-step explanation of how the Filtered-x Least Mean Squares (FxLMS) algorithm works for Active Noise Cancellation.

---

## Table of Contents
1. [The Problem: Canceling Noise](#1-the-problem-canceling-noise)
2. [The Basic Idea](#2-the-basic-idea)
3. [Why Not Just Use LMS?](#3-why-not-just-use-lms)
4. [The FxLMS Solution](#4-the-fxlms-solution)
5. [Step-by-Step Walkthrough](#5-step-by-step-walkthrough)
6. [The Math (Simplified)](#6-the-math-simplified)
7. [Applied to Car ANC](#7-applied-to-car-anc)
8. [Code Walkthrough](#8-code-walkthrough)

---

## 1. The Problem: Canceling Noise

Imagine you're in a car and the engine makes an annoying low-frequency rumble. You want to cancel this noise at your ear.

**The physics principle:** If you play a sound wave that is the exact opposite (inverted) of the noise, they cancel out:

```
Noise:      ∿∿∿∿∿∿∿∿
            +
Anti-noise: ∿∿∿∿∿∿∿∿ (inverted)
            =
Result:     ________ (silence!)
```

**The challenge:** The noise is always changing, and there's a delay between when you detect the noise and when your anti-noise reaches the ear. You need a system that can adapt in real-time.

---

## 2. The Basic Idea

Here's the setup for Active Noise Cancellation:

```
                                    PRIMARY PATH
                                   (noise travels)
    Noise ─────────────────────────────────────────────────┐
    Source                                                 │
       │                                                   ▼
       │                                              ┌─────────┐
       └──────► Reference ──────► Adaptive ──────►   │  Error  │ ──► Error
                  Mic              Filter    Speaker │   Mic   │    Signal
                   │                 │               │  (Ear)  │
                   │                 │               └─────────┘
                   x(n)              y(n)                 │
                                                         e(n)
                                                         │
                              ◄──────────────────────────┘
                              (feedback to update filter)
```

**Components:**
- **Reference Mic:** Picks up the noise early (before it reaches your ear)
- **Adaptive Filter:** Creates the anti-noise signal
- **Speaker:** Plays the anti-noise
- **Error Mic:** Measures what's left after cancellation (at your ear)

**The Goal:** Adjust the adaptive filter so that e(n) → 0 (silence at the ear)

---

## 3. Why Not Just Use LMS?

The standard LMS algorithm updates filter weights like this:

```
w(n+1) = w(n) + μ · e(n) · x(n)

Where:
  w = filter weights
  μ = step size (learning rate)
  e = error signal
  x = reference signal
```

**The Problem:** There's a speaker and acoustic path between the filter output and the error mic!

```
Filter Output ───► Speaker ───► [Acoustic Path] ───► Error Mic
     y(n)                            S(z)              e(n)
```

This acoustic path (called the **Secondary Path**) introduces:
- **Delay:** Sound takes time to travel from speaker to ear
- **Filtering:** The room/car changes the sound (echoes, absorption)

If we ignore this path, the LMS algorithm gets confused and may become unstable!

---

## 4. The FxLMS Solution

The key insight: **Filter the reference signal through an estimate of the secondary path** before using it to update the weights.

```
                                    PRIMARY PATH P(z)
    Noise ──────────────────────────────────────────────────┐
    Source                                                  │
       │                                                    ▼
       ▼                                                    +
    Reference ───────────► Adaptive ───────► Speaker ───────┤ Error
      Mic                   Filter W(z)           │         │  Mic
       │                       ▲                  │         │
       x(n)                    │           Secondary Path   │
       │                       │              S(z)          │
       │                       │                            │
       ▼                       │                            ▼
    ┌──────────┐               │                          e(n)
    │ Ŝ(z)     │ ◄─────────────┴────────────────────────────┘
    │(estimate)│              Filtered-x
    └──────────┘              Update
         │
         ▼
       x'(n)  (filtered reference)
```

**The "Filtered-x" name comes from:** filtering x(n) through Ŝ(z) to get x'(n)

---

## 5. Step-by-Step Walkthrough

Here's exactly what happens at each time step n:

### Step 1: Capture Reference Signal
```
x(n) = signal from reference microphone
```
This picks up the noise before it reaches your ear.

### Step 2: Generate Anti-Noise
```
y(n) = W(z) * x(n)

      = w₀·x(n) + w₁·x(n-1) + w₂·x(n-2) + ... + wₗ·x(n-L)
```
The adaptive filter W(z) with L taps convolves with the reference signal to create anti-noise.

### Step 3: Anti-Noise Travels Through Secondary Path
```
y'(n) = S(z) * y(n)
```
The anti-noise goes through the speaker and acoustic path to reach the error mic. This happens physically (we simulate it).

### Step 4: Noise Travels Through Primary Path
```
d(n) = P(z) * noise(n)
```
Meanwhile, the original noise travels through the primary path to the error mic.

### Step 5: Calculate Error
```
e(n) = d(n) + y'(n)
```
The error mic hears both:
- d(n) = the noise
- y'(n) = the anti-noise

If y'(n) ≈ -d(n), they cancel and e(n) ≈ 0

### Step 6: Filter the Reference (The "Filtered-x" Part)
```
x'(n) = Ŝ(z) * x(n)
```
We filter the reference signal through our **estimate** of the secondary path. This compensates for the delay and filtering that the anti-noise experiences.

### Step 7: Update the Filter Weights
```
w(n+1) = w(n) + μ · e(n) · x'(n)
```
Now we update using the **filtered** reference x'(n) instead of x(n).

### Repeat!
Go back to Step 1 for the next sample.

---

## 6. The Math (Simplified)

### The Adaptive Filter (FIR Filter)

```
y(n) = Σ wₖ · x(n-k)   for k = 0 to L-1
```

In vector form:
```
y(n) = wᵀ · x(n)

Where:
  w = [w₀, w₁, w₂, ..., wₗ₋₁]ᵀ     (filter weights)
  x(n) = [x(n), x(n-1), ..., x(n-L+1)]ᵀ  (input buffer)
```

### The LMS Update

Standard LMS:
```
w(n+1) = w(n) + μ · e(n) · x(n)
```

### The FxLMS Update

```
w(n+1) = w(n) + μ · e(n) · x'(n)

Where:
  x'(n) = Ŝ(z) * x(n)   (reference filtered through secondary path estimate)
```

### The FxNLMS Update (Normalized)

To improve stability, we normalize by the signal power:

```
w(n+1) = w(n) + μ · e(n) · x'(n) / (||x'(n)||² + δ)

Where:
  ||x'(n)||² = sum of squared values in x'(n) buffer
  δ = small regularization constant (prevents division by zero)
```

---

## 7. Applied to Car ANC

In our car simulation:

```
┌─────────────────────────────────────────────────────────────────┐
│                         CAR INTERIOR                            │
│                                                                 │
│   Engine     Firewall    Dashboard    Driver Seat               │
│    ███         ║            ═══         ┌───┐                   │
│   NOISE ──────►║◄── Ref ──────────────► │EAR│ ◄── Error Mic    │
│   Source       ║     Mic                └───┘                   │
│                ║              Speaker ────┘                     │
│                ║                 │                              │
│                ║                 │                              │
│                ║           ┌─────┴─────┐                        │
│                ║           │  FxLMS    │                        │
│                ║           │  Filter   │                        │
│                ║           └───────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

### The Paths:

**Primary Path P(z):** Engine → Firewall → Air → Driver's Ear
- This is the path the noise takes to reach your ear
- Includes reflections off windows, seats, dashboard

**Reference Path:** Engine → Firewall → Reference Mic
- How noise reaches the reference microphone
- Should be shorter than primary path (gives us time to react)

**Secondary Path S(z):** Speaker → Air → Driver's Ear
- How the anti-noise travels from speaker to ear
- We need to estimate this for FxLMS to work

### Why Reference Mic Position Matters:

```
Good: Reference mic BEFORE the ear (gives processing time)

    Noise ──► Ref Mic ──────────────────────► Ear
                 │                             ▲
                 └──► FxLMS ──► Speaker ───────┘
                      (has time to process)


Bad: Reference mic too close to ear (no processing time)

    Noise ──────────────────────► Ear ◄── Ref Mic
                                   ▲
                     Speaker ──────┘
                     (too late!)
```

### Why Speaker Position Matters:

**Closer speaker = better cancellation**
```
Headrest speaker (very close): 15-20 dB reduction
Door speaker (far away):       5-10 dB reduction
```

The secondary path is shorter, making it easier to control.

---

## 8. Code Walkthrough

Here's how our implementation works:

### The FxNLMS Class (`src/core/fxlms.py`)

```python
class FxNLMS:
    def __init__(self, filter_length, step_size, secondary_path_estimate):
        self.L = filter_length              # Number of filter taps
        self.mu = step_size                 # Learning rate
        self.weights = np.zeros(L)          # Adaptive filter weights
        self.x_buffer = np.zeros(L)         # Input buffer
        self.xf_buffer = np.zeros(L)        # Filtered-x buffer
        self.S_hat = secondary_path_estimate # Estimate of secondary path
```

### Step-by-Step in Code:

**1. Receive new reference sample:**
```python
def generate_antinoise(self, x):
    # Shift buffer and add new sample
    self.x_buffer = np.roll(self.x_buffer, 1)
    self.x_buffer[0] = x

    # Generate anti-noise: y = w · x
    y = np.dot(self.weights, self.x_buffer)
    return y
```

**2. Filter the reference through secondary path estimate:**
```python
def filter_reference(self, x):
    # x'(n) = Ŝ(z) * x(n)
    self.xf_buffer = np.roll(self.xf_buffer, 1)
    self.xf_buffer[0] = np.dot(self.S_hat, self.x_buffer)
```

**3. Update weights using error:**
```python
def update_weights(self, error):
    # Normalized update: w = w + μ * e * x' / (||x'||² + δ)
    power = np.dot(self.xf_buffer, self.xf_buffer) + self.delta
    self.weights += self.mu * error * self.xf_buffer / power
```

### The Simulation Loop:

```python
for i in range(n_samples):
    # 1. Get noise sample
    noise_sample = noise_signal[i]

    # 2. Noise through reference path → reference mic
    x = reference_path.filter(noise_sample)

    # 3. Noise through primary path → error mic (what we want to cancel)
    d = primary_path.filter(noise_sample)

    # 4. Generate anti-noise
    y = fxlms.generate_antinoise(x)

    # 5. Anti-noise through secondary path → error mic
    y_at_ear = secondary_path.filter(y)

    # 6. Error = noise + anti-noise (should approach zero)
    e = d + y_at_ear

    # 7. Update filter weights
    fxlms.filter_reference(x)
    fxlms.update_weights(e)
```

---

## Key Parameters

| Parameter | Description | Typical Value | Effect |
|-----------|-------------|---------------|--------|
| `filter_length` | Number of taps in adaptive filter | 64-512 | More taps = can model longer impulse responses, but slower |
| `step_size` (μ) | Learning rate | 0.001-0.02 | Larger = faster adaptation, but less stable |
| `regularization` (δ) | Prevents division by zero | 1e-4 | Small positive constant |

### Trade-offs:

**Step Size (μ):**
- Too small → Slow convergence, may not track fast-changing noise
- Too large → Unstable, filter may diverge

**Filter Length:**
- Too short → Can't model the acoustic paths, poor cancellation
- Too long → More computation, slower adaptation

---

## Summary

1. **Reference mic** picks up noise early
2. **Adaptive filter** creates anti-noise: `y = W * x`
3. **Anti-noise** plays through speaker, travels to ear via **secondary path**
4. **Error mic** measures residual noise: `e = d + y'`
5. **Filtered-x trick**: Filter reference through secondary path estimate before updating
6. **Update weights**: `w += μ * e * x' / (power + δ)`
7. **Repeat** every sample (16,000 times per second at 16kHz)

The filter learns to create the perfect anti-noise that, after traveling through the secondary path, arrives at the ear exactly out of phase with the original noise.

---

## References

- Widrow, B., & Stearns, S. D. (1985). Adaptive Signal Processing
- Kuo, S. M., & Morgan, D. R. (1996). Active Noise Control Systems
- Elliott, S. J. (2001). Signal Processing for Active Control
