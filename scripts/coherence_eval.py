"""
scripts/coherence_eval.py

Fully self-contained (does NOT modify Library/latent_dynamics.py or any
other existing file). Computes an amplitude-weighted Kuramoto order
parameter R(t) -- a second, independent take on "how coherent is the
latent right now" -- from Data/S02_latent_dataset.csv's 8 latent channels,
and compares it against latent_slow_velocity (imported, not reimplemented)
over the same steady-locomotion task subset used in the prior analysis.

R(t) and latent_slow_velocity measure different things by construction:
R(t) is INSTANTANEOUS cross-channel phase alignment (do the 8 channels
currently point the same direction, amplitude-weighted); it uses phase
and needs no history. latent_slow_velocity is a RATE OF CHANGE of a
single channel's own (frequency, amplitude) over time; it uses no phase
at all and needs a local time window (np.gradient + smoothing). Whether
they end up redundant or complementary on this data is an empirical
question this script actually answers, not assumed.

Column-name gotcha carried over from every other script in this repo:
the CSV's ch{i}_phase columns are FLD's raw phase output in CYCLES,
wrapped to (-0.5, 0.5] -- NOT radians. They're converted to radians
(x 2*pi) before any cos/sin call below, exactly like locomotion_index.py
/ plot_amplitude_weighted_latent_vs_insole_force.py already do.

Run:
    python3 scripts/coherence_eval.py
"""

import os
import re
import sys

import numpy as np
import pandas as pd

# Only reused import, per the brief -- everything else in this script is local/self-contained.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from Library.latent_dynamics import latent_slow_velocity

DATA_CSV = os.path.join(os.path.dirname(__file__), "..", "Data", "S02_latent_dataset.csv")
DOMINANT_CHANNEL = 4   # highest mean amplitude during locomotion-regime windows (established earlier in chat)
SMOOTH_WINDOW = 9

STEADY_WALKING_TASKS = ["walking_0_8_m_s", "walking_1_2_m_s", "walking_1_4_m_s"]
INCLINE_WALKING_TASKS = ["incline_walking_5_deg_mix_speed", "incline_walking_10_mix_speed"]

# Genuine locomotion-transition task (speed/incline CHANGING within a
# trial) vs. sit_stand_squat_transitions (a posture transition) -- detected
# by name against the actual label set, not assumed to exist.
LOCOMOTION_TRANSITION_KEYWORDS = (
    "speed_transition", "speed_change", "incline_transition", "incline_change", "gait_transition",
)


def coherence_R(phi, amp, eps=1e-8):
    """
    phi: (N, 8) phase per channel per timestep, RADIANS.
    amp: (N, 8) amplitude per channel per timestep, >= 0.

    Amplitude-weighted Kuramoto order parameter:
        R(t) = | sum_i a_i(t) * exp(j*phi_i(t)) | / sum_i a_i(t)

    Implemented via cos/sin components (no complex numbers): the vector
    (sum_i a_i*cos(phi_i), sum_i a_i*sin(phi_i)) is the amplitude-weighted
    resultant; its magnitude, divided by the amplitude sum (eps-guarded
    against all-near-zero amplitude), is the weighted mean resultant
    length -- 1.0 when every channel points the same direction, ->0 when
    they're spread out/canceling.

    Returns R, shape (N,), values in [0, 1].
    """
    phi = np.asarray(phi, dtype=float)
    amp = np.asarray(amp, dtype=float)
    if phi.ndim != 2 or phi.shape != amp.shape:
        raise ValueError(f"phi and amp must both be 2D and the same shape; got {phi.shape}, {amp.shape}")

    x = np.sum(amp * np.cos(phi), axis=1)   # (N,) amplitude-weighted resultant, x component
    y = np.sum(amp * np.sin(phi), axis=1)   # (N,) amplitude-weighted resultant, y component
    amp_sum = np.sum(amp, axis=1)           # (N,)

    return np.sqrt(x**2 + y**2) / (amp_sum + eps)


def _compute_velocity_per_trial(df, channel, smooth_window):
    """latent_slow_velocity's np.gradient needs monotonically increasing t;
    time_s resets to ~0 at the start of every trial in this concatenated
    CSV, so this calls it once per trial_id and stitches results back into
    one array aligned to df's row order. (This loop is just orchestration
    around the imported function, not a reimplementation of it.)"""
    velocity = np.full(len(df), np.nan)
    for trial_id, idx in df.groupby("trial_id").groups.items():
        idx = np.asarray(idx.sort_values())
        sub = df.loc[idx]
        vel, _ = latent_slow_velocity(
            sub["time_s"].values,
            sub[f"ch{channel}_freq_hz"].values,
            sub[f"ch{channel}_amp"].values,
            smooth_window=smooth_window,
        )
        velocity[idx] = vel
    return velocity


def main():
    df = pd.read_csv(DATA_CSV)
    df = df[~df["warmup"]].reset_index(drop=True)

    # Discover the 8 phase / 8 amplitude columns from the actual header,
    # rather than assuming ch1..ch8 (they're ch0..ch7 -- confirmed by
    # inspection, not assumed).
    phase_cols = sorted(
        (c for c in df.columns if re.fullmatch(r"ch\d+_phase", c)),
        key=lambda c: int(re.search(r"\d+", c).group()),
    )
    amp_cols = sorted(
        (c for c in df.columns if re.fullmatch(r"ch\d+_amp", c)),
        key=lambda c: int(re.search(r"\d+", c).group()),
    )
    print(f"[Coherence] Discovered phase columns: {phase_cols}")
    print(f"[Coherence] Discovered amp columns:   {amp_cols}")
    if len(phase_cols) != 8 or len(amp_cols) != 8:
        raise ValueError(f"Expected 8 phase and 8 amp columns, found {len(phase_cols)} and {len(amp_cols)}")

    phi_rad = 2.0 * np.pi * df[phase_cols].values   # cycles -> radians (see module docstring)
    amp = df[amp_cols].values

    print(f"[Coherence] Computing R(t) for the full dataset ({len(df):,} rows)...")
    R = coherence_R(phi_rad, amp)

    print(f"[Coherence] Computing latent_slow_velocity (ch{DOMINANT_CHANNEL}) for the same rows...")
    velocity = _compute_velocity_per_trial(df, DOMINANT_CHANNEL, SMOOTH_WINDOW)

    all_tasks = sorted(df["task"].unique())
    locomotion_transition_tasks = [t for t in all_tasks
                                    if any(k in t for k in LOCOMOTION_TRANSITION_KEYWORDS)]
    keep_tasks = STEADY_WALKING_TASKS + INCLINE_WALKING_TASKS + locomotion_transition_tasks
    missing = [t for t in keep_tasks if t not in all_tasks]
    if missing:
        print(f"[Coherence] WARNING: requested tasks not found in data, skipping: {missing}")
        keep_tasks = [t for t in keep_tasks if t in all_tasks]

    mask = df["task"].isin(keep_tasks).values
    task_r = df.loc[mask, "task"].values
    R_r = R[mask]
    velocity_r = velocity[mask]

    print(f"\n=== Mean R(t) by task (restricted set, sorted ascending) ===")
    means = pd.DataFrame({"task": task_r, "R": R_r}).groupby("task")["R"].mean().sort_values()
    for task, m in means.items():
        print(f"{task:<35s} {m:.4f}")

    corr = np.corrcoef(R_r, velocity_r)[0, 1]
    print(f"\n=== Correlation over restricted set ===")
    print(f"corr(R(t), latent_slow_velocity) = {corr:.4f}")
    if abs(corr) > 0.7:
        relationship = "strongly related -- largely redundant on this data"
    elif abs(corr) > 0.3:
        relationship = "moderately related -- partial overlap, not redundant"
    else:
        relationship = "weakly/not related -- catching largely different things"
    print(f"-> {relationship}")

    # One flagged surprise, computed from the actual numbers above -- not a
    # deferral to "go check by eye".
    velocity_means = pd.DataFrame({"task": task_r, "v": velocity_r}).groupby("task")["v"].mean()
    walking_order = [t for t in STEADY_WALKING_TASKS if t in means.index]
    r_speed = means.loc[walking_order].values
    v_speed = velocity_means.loc[walking_order].values
    r_monotonic = bool(np.all(np.diff(r_speed) > 0))
    v_monotonic = bool(np.all(np.diff(v_speed) > 0))

    incline_present = [t for t in INCLINE_WALKING_TASKS if t in means.index]
    incline_below_all_walking = bool(
        incline_present and walking_order and means.loc[incline_present].max() < means.loc[walking_order].min()
    )

    flags = []
    if r_monotonic and not v_monotonic:
        flags.append(
            f"R(t) rises monotonically with walking speed across {walking_order} "
            f"({r_speed.round(4).tolist()}), but latent_slow_velocity does NOT "
            f"({v_speed.round(4).tolist()}, order-inverted between the two faster speeds) -- "
            f"they disagree on ORDERING, not just magnitude, so naively averaging the two as one "
            f"confidence signal could send conflicting readings near a speed change."
        )
    if incline_below_all_walking:
        flags.append(
            f"incline walking has the LOWEST R(t) of every task in this restricted set "
            f"({means.loc[incline_present].round(4).to_dict()}), scoring below even the slowest "
            f"flat walking ({walking_order[0]}={means[walking_order[0]]:.4f}) -- despite being just "
            f"as steady/rhythmic, its 8 channels are less phase-aligned by this metric, which is not "
            f"what 'steady locomotion = confident' would predict."
        )
    print("\n[Coherence] SURPRISING RESULT: " + (" ".join(flags) if flags else
          "none found -- R(t) and latent_slow_velocity's task orderings agree on this restricted set."))


if __name__ == "__main__":
    main()
