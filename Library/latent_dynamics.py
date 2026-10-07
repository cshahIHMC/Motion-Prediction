"""
Library/latent_dynamics.py

Numpy/scipy-only diagnostics on the (t, phi, f, a) limit-cycle
representation of a latent gait trajectory -- phi (phase) deliberately
excluded from anything meant to read as "stability", since phase always
moves fast in steady-state locomotion (it's a limit cycle: phi_dot is
large and roughly constant by design, not a fault).
"""

import numpy as np
from scipy.signal import savgol_filter


def latent_slow_velocity(t, f, a, smooth_window=5, log_transform=False, labels=None, eps=1e-8):
    """
    "Slow" latent velocity: the rate of change of the limit-cycle's own
    shape parameters (frequency f, amplitude a) -- NOT phase velocity
    phi_dot, which is excluded on purpose. phi spins fast even in
    confident steady-state walking (that's what a limit cycle is), so it
    shouldn't count as instability; what SHOULD move slowly in steady
    state -- and is therefore worth flagging when it doesn't -- is how
    fast the cycle's own frequency/amplitude are changing.

    t, f, a : 1D array_like, shape (N,). t need not be uniformly sampled
        (np.gradient is used against t directly, not a fixed-step diff).
    smooth_window : Savitzky-Golay window length, in samples, applied to
        each derivative separately (before combining) -- coerced to a
        valid odd value >= 3. Too-short series (N < smooth_window) skip
        smoothing rather than erroring.
    log_transform : if True, log-transform f and a (log1p of their
        shifted-positive values) before z-scoring, instead of z-scoring
        the raw values. Use this when f/a are strictly positive and span
        more than about a decade, so a proportional change contributes
        comparably regardless of operating point.
    labels : optional 1D array_like, shape (N,) -- e.g. a task/regime tag
        per sample. When given, also returns the per-label mean velocity,
        so you can sanity-check that steady segments score low and
        transitions score high.
    eps : numerical floor for std/division and the log shift, avoids
        div-by-zero on a near-constant channel.

    Returns
    -------
    velocity : (N,) ndarray -- sqrt(f_dot_norm**2 + a_dot_norm**2), smoothed.
    segment_means : dict {label: mean velocity}, or None if labels is None.
    """
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    a = np.asarray(a, dtype=float)
    if t.ndim != 1 or t.shape != f.shape or t.shape != a.shape:
        raise ValueError(f"t, f, a must be 1D and the same shape; got {t.shape}, {f.shape}, {a.shape}")

    def _zscore(x):
        if log_transform:
            x = np.log1p(x - x.min() + eps)   # shift to strictly positive, then log1p
        sigma = x.std()
        return (x - x.mean()) / (sigma + eps)

    f_dot = np.gradient(_zscore(f), t)
    a_dot = np.gradient(_zscore(a), t)

    # Smooth each derivative separately, before combining -- raw np.gradient
    # amplifies sample-to-sample noise, and smoothing AFTER sqrt(sum of
    # squares) would bias the magnitude upward instead of denoising it.
    n = len(t)
    win = max(3, int(smooth_window) | 1)          # force odd, >= 3
    win = min(win, n if n % 2 == 1 else n - 1)     # can't exceed series length
    if n >= win >= 3:
        polyorder = min(2, win - 1)
        f_dot = savgol_filter(f_dot, win, polyorder)
        a_dot = savgol_filter(a_dot, win, polyorder)

    velocity = np.sqrt(f_dot**2 + a_dot**2)

    segment_means = None
    if labels is not None:
        labels = np.asarray(labels)
        segment_means = {lab: float(velocity[labels == lab].mean()) for lab in np.unique(labels)}

    return velocity, segment_means
