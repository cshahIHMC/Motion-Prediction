"""
Lag-lead filter demo
=====================

What this does, step by step:

1. Builds a "continuous" signal (really a very densely sampled discrete
   signal, which is how you fake continuous-time in numpy) made of TWO
   sine waves added together — a slow one and a fast one — so there are
   two distinct things in the signal for the filter to act on.

2. Applies an AMPLITUDE scale factor to that raw signal before filtering,
   just to demonstrate "modifying the signal" independently of the filter
   itself.

3. Defines a simple lag-lead compensator as a Laplace-domain transfer
   function:

              s + z
      H(s) = K -----          (K = GAIN sits at the top / numerator)
              s + p

   - If z < p, the zero comes in before the pole -> LEAD behavior
     (phase boost, high frequencies passed/boosted).
   - If z > p, the pole comes in before the zero -> LAG behavior
     (low frequencies passed, high frequencies attenuated).

   Here we chain a lag stage and a lead stage together (classic
   "lag-lead compensator") and put one overall GAIN on top of the whole
   thing.

4. Runs the amplitude-scaled signal through H(s) using
   scipy.signal.lsim (true continuous-time simulation, no manual
   discretization needed) to get the "modified" signal.

5. Plots the original signal and the filtered/modified signal on the
   same axes with matplotlib and calls plt.show().
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def build_lag_lead_tf(gain=2.0, z_lag=5.0, p_lag=0.5, z_lead=2.0, p_lead=20.0):
    """
    Build a lag-lead compensator transfer function:

              s + z_lag       s + z_lead
      H(s) = gain * --------- * -----------
              s + p_lag        s + p_lead

    `gain` is the numerator ("top") scalar. Lag stage needs its pole
    closer to the origin than its zero (p_lag < z_lag) -> boosts low
    frequencies, negative phase in between, settles to 1 at high
    frequency. Lead stage needs its zero closer to the origin than its
    pole (z_lead < p_lead) -> boosts high frequencies, positive phase
    in between. Returns a scipy.signal.lti system.
    """
    lag_num, lag_den = [1.0, z_lag], [1.0, p_lag]
    lead_num, lead_den = [1.0, z_lead], [1.0, p_lead]

    num = gain * np.polymul(lag_num, lead_num)
    den = np.polymul(lag_den, lead_den)
    return signal.lti(num, den)


def apply_lag_lead_filter(u, fs_hz, amplitude=1.0, **tf_kwargs):
    """
    Run a 1D array of samples `u` (e.g. a model-predicted signal —
    doesn't have to be a sine wave, or come from any particular
    source) through the lag-lead filter.

    u        : array_like, shape (N,) — your signal, one sample per
               timestep at a fixed rate `fs_hz`. This is exactly the
               shape a model's predicted output array would already
               be in.
    fs_hz    : float — sample rate of `u`, in Hz.
    amplitude: scalar applied to `u` before filtering.
    tf_kwargs: passed through to build_lag_lead_tf (gain, z_lag, p_lag,
               z_lead, p_lead).

    Returns (t, u_scaled, y) — the time vector, the amplitude-scaled
    input, and the filtered output, all shape (N,).
    """
    u = np.asarray(u, dtype=float)
    t = np.arange(len(u)) / fs_hz
    u_scaled = amplitude * u

    tf = build_lag_lead_tf(**tf_kwargs)
    _, y, _ = signal.lsim(tf, U=u_scaled, T=t)
    return t, u_scaled, y


# ---------------------------------------------------------------------
# Demo signal: two sine waves ("two things") summed, standing in for
# whatever array your model actually predicts. In real use, replace
# `signal_original` below with your model's output array and set
# FS_HZ to its actual sample rate — everything else is unchanged.
# ---------------------------------------------------------------------
DURATION_S = 2.0          # length of the signal, seconds
FS_HZ = 2000.0            # sample rate (Hz) — match your model's output rate

t_demo = np.arange(0.0, DURATION_S, 1.0 / FS_HZ)

f1_hz, amp1 = 1.0, 1.0    # slow component
f2_hz, amp2 = 1.0, 1.0    # fast component

signal_original = amp1 * np.sin(2 * np.pi * f1_hz * t_demo) + \
                   amp2 * np.sin(2 * np.pi * f2_hz * t_demo)

AMPLITUDE = 1.0           # amplitude gain applied before filtering
GAIN = 1.0                # filter's own gain, at the top of H(s)

t, signal_scaled, signal_modified = apply_lag_lead_filter(
    signal_original, FS_HZ, amplitude=AMPLITUDE, gain=GAIN,
)

print("Lag-lead compensator H(s) built with:")
print(f"  gain (top) = {GAIN}, amplitude scale = {AMPLITUDE}")

# ---------------------------------------------------------------------
# Plot original vs. modified signal.
# ---------------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(t, signal_original, label="Original signal (2 sine components)")
plt.plot(t, signal_modified, label=f"Modified signal (x{AMPLITUDE} amplitude -> lag-lead filter, gain={GAIN})")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Original vs. Lag-Lead Filtered Signal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
