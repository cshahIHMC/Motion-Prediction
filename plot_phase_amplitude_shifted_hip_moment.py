"""
plot_phase_amplitude_shifted_hip_moment.py

150Hz / teacher-student STUDENT-branch counterpart of the earlier plain-FLD
version of this script: combines PHASE shift with a simple MEAN-AMPLITUDE
scalar gain (the old K5 log-sum-exp gain and the flat end-of-pipeline
output-scale knob are gone -- replaced by this simpler mechanism, per
explicit request) into a modified torque COMMAND signal, now for FOUR hip
moment signals at once -- hip_flexion_r/l_moment (flexion/extension) and
hip_adduction_r/l_moment (adduction/abduction) -- each its own subplot,
sharing one time axis, in one figure.

  1. PHASE shift, applied as a genuinely TIME-VARYING delay/lead
     (apply_variable_delay() below): theta(phi) -> delta_t(t) is the same
     per-window computation plot_explicit_phase_shift_torque.py uses
     (theta_of_phase(), reused from there), but is NOT collapsed to one
     nominal (mean) delay for a fixed LTI filter the way that script (and an
     earlier revision of this one) does -- averaging delta_t(t) away only
     lost nothing when theta(phi) was flat/constant; now that THETA_COEFFS
     makes theta(phi) swing between delay and lead over the gait cycle,
     that average would silence the exact effect being asked for. Instead,
     each output sample is read directly off the ground-truth signal at its
     own shifted query time via linear interpolation: tau_shifted(t) =
     hip_moment(t - delta_t(t)). This is exact (no Pade approximation
     error) and still preserves the real signal's own envelope/magnitude
     (a template-resample approach couldn't).
  2. AMPLITUDE gain -- simple mean of all 8 encoded amplitude channels at
     each timestep, mean-1 normalized so it multiplies tau_shifted without
     changing its overall scale:
         avg_amp(t)      = mean_i( amp_i(t) )
         avg_amp_gain(t) = avg_amp(t) / mean_t( avg_amp(t) )
     No per-channel weighting, no log-sum-exp, no tunable strength knob --
     just the plain cross-channel mean used directly as the scalar gain.

    tau_cmd(t) = avg_amp_gain(t) * tau_shifted(t)

is this script's final output -- the "modified torque command signal".
Ground truth and tau_cmd are plotted in N.m/kg, their native
weight-normalized units. Per explicit request, insole force is still NOT
plotted -- each subplot shows exactly two lines: ground truth and the
modified command. avg_amp_gain is mean-1, so it does not systematically
shrink or grow the signal's overall scale, but it is NOT a pure positive
scalar (it varies over time), so the printed correlation DOES change from
the phase-only case.

A SECOND figure plots the commanded phase shift itself, delta_t(t) (one
shared line, the EXACT SAME delta_t(t) Mechanism 1 applies directly, per
timestep, above -- no collapsing/averaging happens anywhere anymore, so
this plot is a faithful picture of what Mechanism 1 actually does), over
time -- in a DISPLAY sign convention where DELAY is negative and LEAD is
positive. Note this is the OPPOSITE of apply_variable_delay's internal sign
convention (positive delta_t = lag/delay there, per its own docstring) --
flipped here purely for this plot, per explicit request, so "commanding a
delay" reads as a negative number and "commanding a lead" as positive.

Model: the encoder is the pinned 70-epoch
runs/FLDTeacherStudent_20260921_163758 checkpoint's STUDENT branch
(150Hz, history_horizon=151, forecast_horizon=20, trained on
Data/all_data_150_corrected.csv via train_fld_teacher_student.py; earlier
revisions of this script pinned the 160-epoch horizon-10
FLDTeacherStudent_20260915_165611 and, before that, a 100Hz no-future model).
The notes below on the student/teacher choice and input width still apply --
the encoder is the STUDENT branch
specifically -- NOT the teacher branch, even though the teacher is
available in the raw trial file too -- because this script's whole point is
to produce a torque COMMAND signal, which at deployment only ever has the
student's wearable-only view available. phi(t) and f(t) below are
therefore the student's own phase/frequency, and tau_template is built by
binning ground-truth moment against the STUDENT's phase (not the
teacher's) -- consistent with "what the deployed controller would actually
have to work with". This checkpoint trains FLD with forecast_horizon=20
(genuinely-future prediction, 20 frames @150Hz ahead). Its architecture is
the [64, 32] encoder / [32, 64] decoder shapes, and its STUDENT input is
36ch (DataLoader/teacher_student_data_loader.py's SENSOR_COLUMNS -- body IMU
(30ch) PLUS insole force/COP (6ch)).

Reuses (rather than reimplementing):
  - _load_fld_teacher_student_model from plot_pca_manifold_train.py (150Hz
    checkpoint loading) -- the only cross-script import left; every other helper below
    (encode_student_series, theta_of_phase, check_nonnegative) is now
    defined LOCALLY, reproduced verbatim from the scripts they used to be
    imported from (plot_pca_manifold_train_100hz.py,
    plot_explicit_phase_shift_torque.py, plot_energy_gain_variants_vs_biomech.py
    respectively) -- all three have since been removed from this repo.
    (build_delay_filter, from the same now-removed
    plot_explicit_phase_shift_torque.py, is NOT reproduced here -- this
    script applies delta_t(t) directly/time-varying via its own
    apply_variable_delay() instead of a fixed LTI filter's single nominal
    delay; compute_energy_gain_variants, the old K5/log-sum-exp gain, is
    likewise not reproduced -- replaced by the plain cross-channel mean
    described above.)

Read-only / display-only: loads the checkpoint, slides the student encoder
across the trial, computes the combined reconstruction for the 4 moment
signals, and shows ONE figure via plt.show() -- nothing saved, nothing
retrained.

Run:
    python3 plot_phase_amplitude_shifted_hip_moment.py
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from DataLoader.teacher_student_data_loader import SENSOR_COLUMNS as TS_SENSOR_COLUMNS   # 36ch, 150Hz:
                                                       # body IMU (30) + insole force/COP (6) -- matches the
                                                       # pinned horizon-10 checkpoint's STUDENT input width
from plot_pca_manifold_train import _load_fld_teacher_student_model   # 150Hz checkpoint loader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SUBJECT = "S02"
TRIAL_NUMBER = "3"
FLD_TS_RUN = "FLDTeacherStudent_20260921_163758"   # pinned -- latest trained FLDTeacherStudent checkpoint,
                                                    # 70 completed epochs, history_horizon=151/forecast_horizon=20,
                                                    # trained on Data/all_data_150_corrected.csv (train_fld_teacher_student.py)

DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")
RATE = "150"   # matches Data/<SUBJECT>/150/trial_<TRIAL_NUMBER>_all_data.csv

LATENT_STRIDE = 1
MAX_ENCODE_BATCH = 256

# Left/right vertical GRF, force PLATE (not insole) -- same convention as the other scripts in this
# repo: *_plate_Fz reads ~0 N unloaded and strongly NEGATIVE when loaded, so abs() is plotted.
LEFT_GRF_COLUMN = "L_plate_Fz"
RIGHT_GRF_COLUMN = "R_plate_Fz"

# The four target signals -- flexion/extension (hip_flexion) and
# adduction/abduction (hip_adduction), both legs.
HIP_MOMENT_COLUMNS = [
    "hip_flexion_r_moment", "hip_flexion_l_moment",
    "hip_adduction_r_moment", "hip_adduction_l_moment",
]

# Phase-shift knob -- identical mechanism/meaning to plot_explicit_phase_shift_torque.py.
# theta(phi) = theta0 + c1*cos(phi) + s1*sin(phi) + c2*cos(2*phi) + s2*sin(2*phi)
# theta0=0.0, c1=0.6 -> theta(phi) swings smoothly between -0.6 (phi=pi, ~mid-swing)
# and +0.6 (phi=0, ~heel-strike/early-stance) once per gait cycle -- a genuinely
# phase-DEPENDENT commanded shift (was: flat theta0=0.5 everywhere, a constant
# shift regardless of gait phase).
N_HARMONICS = 2
THETA_COEFFS = [0.0, 0.6, 0.0, 0.0, 0.0]   # theta0=0.0, c1=0.6, s1=c2=s2=0
assert len(THETA_COEFFS) == 1 + 2 * N_HARMONICS, (
    f"THETA_COEFFS must have length 1 + 2*N_HARMONICS = {1 + 2 * N_HARMONICS}, got {len(THETA_COEFFS)}"
)
DT_MAX_LAG_MS = 125
DT_MAX_LEAD_MS = 30

# Constant multiplier on the whole amplitude-gain curve. The mean-1 gain (avg_amp / mean(avg_amp)) is
# multiplied by this, so tau_cmd's overall scale becomes AMP_GAIN_SCALE x the shifted ground truth
# (1.0 = the original mean-1 behaviour, 0.5 = the command at 50% of ground-truth scale on average).
AMP_GAIN_SCALE = 0.5


def _trial_path(subject_name, trial_number):
    """Data/<subject_name>/150/trial_<trial_number>_all_data.csv (one contiguous recording).
    (Previously imported from plot_pca_manifold_train.py, which now reads its trial from the
    S05 held-out CSV instead and no longer has this helper.)"""
    path = os.path.join(DATA_DIR, subject_name, RATE, f"trial_{trial_number}_all_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- check SUBJECT/TRIAL_NUMBER match a file under "
            f"Data/{subject_name}/{RATE}/ (e.g. trial_2_all_data.csv -> TRIAL_NUMBER=\"2\")."
        )
    return path


def theta_of_phase(phi, coeffs):
    """Phase-dependent theta(phi), a truncated Fourier series in phi (radians):
        theta(phi) = theta0 + sum_n [c_n*cos(n*phi) + s_n*sin(n*phi)]
    coeffs = [theta0, c1, s1, c2, s2, ...]. Clipped to [-1, 1] so it stays a
    valid theta regardless of harmonic amplitude. With every harmonic
    coefficient 0, this reduces exactly to the flat theta(phi) == theta0
    case. (Previously imported from plot_explicit_phase_shift_torque.py,
    which no longer exists in this repo -- reproduced here verbatim.)"""
    phi = np.asarray(phi, dtype=float)
    theta0 = coeffs[0]
    val = np.full_like(phi, theta0)
    for n in range(1, (len(coeffs) - 1) // 2 + 1):
        c_n, s_n = coeffs[2 * n - 1], coeffs[2 * n]
        val = val + c_n * np.cos(n * phi) + s_n * np.sin(n * phi)
    return np.clip(val, -1.0, 1.0)


def check_nonnegative(name, x):
    """Sanity check + print for a latent quantity expected to be nonnegative
    (e.g. amplitude, frequency in Hz). (Previously imported from
    plot_energy_gain_variants_vs_biomech.py, which no longer exists in this
    repo -- reproduced here verbatim.)"""
    x = np.asarray(x)
    if np.any(x < 0):
        print(f"[EnergyGain] WARNING: {name} has negative values! min={x.min()}")
    else:
        print(f"[EnergyGain] OK: {name} is nonnegative (min={x.min()})")


def encode_student_series(model, sensor_data_norm, history_horizon, device, stride=1, batch_size=256):
    """Slide the STUDENT encoder across a whole trial (causal-last-frame
    convention: frame_indices = window_start + history_horizon - 1).
    Model/sensor-agnostic -- only calls model.forward_student() on whatever
    (N, sensor_dim) tensor it's given, so it works for any FLDTeacherStudent
    checkpoint regardless of sample rate/sensor width. (Previously imported
    from plot_pca_manifold_train_100hz.py, which no longer exists in this
    repo -- reproduced here verbatim rather than reimplemented differently.)
    """
    N = sensor_data_norm.shape[0]
    starts = list(range(0, N - history_horizon + 1, stride))
    if not starts:
        raise ValueError(f"Trial too short ({N} rows) for history_horizon={history_horizon}")

    phases, freqs, amps, offsets = [], [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(starts), batch_size):
            batch_starts = starts[i:i + batch_size]
            wins = torch.stack([sensor_data_norm[s:s + history_horizon].T for s in batch_starts]).to(device)
            phase, freq, amp, offset = model.forward_student(wins)
            phases.append(phase.cpu())
            freqs.append(freq.cpu())
            amps.append(amp.cpu())
            offsets.append(offset.cpu())

    phase = torch.cat(phases, dim=0).numpy()
    freq_w = torch.cat(freqs, dim=0).numpy()
    amp = torch.cat(amps, dim=0).numpy()
    offset = torch.cat(offsets, dim=0).numpy()
    frame_indices = np.array(starts) + history_horizon - 1   # causal (last-frame) convention
    return phase, freq_w, amp, offset, frame_indices


def apply_variable_delay(t, x, delta_t):
    """Time-VARYING delay/lead, applied directly (no LTI averaging to a
    single nominal value): output(t) = x(t - delta_t(t)), via linear
    interpolation of x's own (t, x) samples at the shifted query times.
    Positive delta_t = lag/delay (output looks BACK at an earlier x sample);
    negative = lead (looks AHEAD at a later one) -- same sign convention
    plot_explicit_phase_shift_torque.py's build_delay_filter() docstring
    uses. Query times outside [t.min(), t.max()] are clamped to the nearest
    edge sample (np.interp's default), same edge behavior any filter's
    startup/end transient would have."""
    query_t = t - delta_t
    return np.interp(query_t, t, x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PhaseAmp] Using device: {device}")

    model, config, norm_stats = _load_fld_teacher_student_model(device, FLD_TS_RUN)
    H = config["history_horizon"]
    N = config["latent_dim"]
    sample_rate = config["sample_rate"]

    path = _trial_path(SUBJECT, TRIAL_NUMBER)
    print(f"[PhaseAmp] Trial file: {path}")
    usecols = list(dict.fromkeys(TS_SENSOR_COLUMNS + HIP_MOMENT_COLUMNS
                                 + ["subject_weight", "task", LEFT_GRF_COLUMN, RIGHT_GRF_COLUMN]))
    df = pd.read_csv(path, usecols=usecols)
    task_label = str(df["task"].iloc[0])
    trial_id = f"{SUBJECT}_trial_{TRIAL_NUMBER}"
    print(f"[PhaseAmp] Trial: {trial_id} (subject={SUBJECT}, task={task_label}), "
          f"{len(df):,} rows @{sample_rate:.0f}Hz")

    # Moment columns weight-normalized (N.m -> N.m/kg), same convention as
    # DataLoader/teacher_student_data_loader.py's build_datasets() -- only the
    # 4 loaded HIP_MOMENT_COLUMNS here, not all 10 TS_MOMENT_COLUMNS (the
    # other 6 were never read into usecols above).
    df[HIP_MOMENT_COLUMNS] = df[HIP_MOMENT_COLUMNS].div(df["subject_weight"], axis=0)

    raw_sensor = df[TS_SENSOR_COLUMNS].values.astype("float32")
    norm_sensor = torch.tensor(
        (raw_sensor - norm_stats["mean_sensor"]) / norm_stats["std_sensor"], dtype=torch.float32,
    )

    print(f"[PhaseAmp] Encoding STUDENT latent series (stride={LATENT_STRIDE}, {len(df):,} rows)...")
    phase, freq_w, amp, offset, frame_indices = encode_student_series(
        model, norm_sensor, H, device, stride=LATENT_STRIDE, batch_size=MAX_ENCODE_BATCH,
    )
    T = phase.shape[0]
    t = np.arange(T) / sample_rate
    freq_hz = freq_w * sample_rate / H
    print(f"[PhaseAmp] Latent series: T={T:,} windows, N={N} channels")

    check_nonnegative("freq_hz", freq_hz)
    check_nonnegative("amp", amp)

    dom_ch = int(np.argmax(amp.mean(axis=0)))
    print(f"[PhaseAmp] Dominant channel: ch{dom_ch} (mean amp={amp.mean(axis=0)[dom_ch]:.4f})")
    phi_t = np.mod(2.0 * np.pi * phase[:, dom_ch], 2 * np.pi)
    f_t = freq_hz[:, dom_ch]

    # --- Mechanism 1: phase shift, applied as a genuinely TIME-VARYING
    # delay/lead -- computed ONCE, shared across all 4 signals. Same
    # theta(phi) -> delta_t(t) as plot_explicit_phase_shift_torque.py, but
    # (unlike that script, and unlike an earlier revision of THIS script)
    # NOT collapsed to one nominal (mean) delay for a fixed LTI filter --
    # that collapse only threw away information when theta(phi) was flat
    # (constant either way), but now that THETA_COEFFS makes theta(phi)
    # swing between delay and lead over the gait cycle, averaging it away
    # would silence exactly the effect this mechanism is supposed to show.
    # apply_variable_delay() instead looks up each output sample directly
    # at its own (possibly different) shifted time.
    theta_t = theta_of_phase(phi_t, THETA_COEFFS)
    dt_max_t = np.where(theta_t >= 0, DT_MAX_LAG_MS, DT_MAX_LEAD_MS) / 1000.0
    delta_t = theta_t * dt_max_t
    print(f"[PhaseAmp] theta(phi) range=[{theta_t.min():+.3f}, {theta_t.max():+.3f}]  "
          f"delta_t range=[{delta_t.min()*1000:+.2f}, {delta_t.max()*1000:+.2f}]ms "
          f"(applied directly, per-timestep -- no single-nominal averaging)")

    # Commanded phase shift, per timestep, in DISPLAY sign convention: DELAY
    # negative, LEAD positive -- the OPPOSITE of build_delay_filter's own
    # internal convention (there, positive delta_t means lag/delay -- see its
    # docstring). Flipped here purely for plotting, per explicit request.
    commanded_shift_ms = -delta_t * 1000.0
    print(f"[PhaseAmp] commanded phase shift (delay -, lead +) range="
          f"[{commanded_shift_ms.min():+.2f}, {commanded_shift_ms.max():+.2f}]ms")

    # --- Mechanism 2: amplitude gain -- plain mean of all 8 amplitude
    # channels at each timestep, mean-1 normalized, shared across all 4
    # signals (same convention as the old K5 gain, just a simpler statistic).
    avg_amp = amp.mean(axis=1)
    avg_amp_gain = AMP_GAIN_SCALE * avg_amp / avg_amp.mean()   # mean-1 gain, times the constant AMP_GAIN_SCALE
    print(f"[PhaseAmp] avg_amp range=[{avg_amp.min():.4f}, {avg_amp.max():.4f}]  "
          f"avg_amp_gain range=[{avg_amp_gain.min():.3f}, {avg_amp_gain.max():.3f}] (mean-1 normalized, x AMP_GAIN_SCALE={AMP_GAIN_SCALE:g})")

    print(f"\n=== Modified torque command (phase shift + mean-amplitude gain) vs. ground truth ===")
    fig, axes = plt.subplots(len(HIP_MOMENT_COLUMNS), 1, figsize=(14, 3.2 * len(HIP_MOMENT_COLUMNS)), sharex=True)
    for ax, moment_col in zip(axes, HIP_MOMENT_COLUMNS):
        hip_moment = df[moment_col].values.astype("float64")[frame_indices]

        tau_shifted = apply_variable_delay(t, hip_moment, delta_t)   # time-varying delay/lead, applied directly

        tau_cmd = avg_amp_gain * tau_shifted   # the modified torque command signal

        corr = np.corrcoef(tau_cmd, hip_moment)[0, 1]
        corr_phase_only = np.corrcoef(tau_shifted, hip_moment)[0, 1]   # delay filter alone, no amplitude gain
        print(f"{moment_col:<26s} corr(tau_cmd, ground truth)={corr:+.4f}  "
              f"(phase-only [delay filter, no gain] baseline={corr_phase_only:+.4f}, not plotted)")

        ax.plot(t, hip_moment, color="tab:blue", lw=1.0, alpha=0.85, label=f"{moment_col} (ground truth, N·m/kg)")
        ax.plot(t, tau_cmd, color="black", lw=1.2, label="modified torque command (phase + mean amplitude), N·m/kg")
        ax.set_title(moment_col, fontsize=10)
        ax.set_ylabel("N·m/kg")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Modified Torque Command (Time-Varying Phase Shift + Mean-Amplitude Gain) vs. Ground Truth -- "
                 f"{trial_id} ({SUBJECT}, {task_label}), shift range="
                 f"[{delta_t.min()*1000:+.1f}, {delta_t.max()*1000:+.1f}]ms "
                 f"(from theta0={THETA_COEFFS[0]:+.2f}, {N_HARMONICS} harmonic(s)), "
                 f"150Hz student branch",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    # --- Commanded phase shift (delay/lead) vs. time, with the left/right vertical
    # GRF beneath it on a shared time axis (top: shared line, computed once above,
    # same for all 4 moment signals, display convention: DELAY negative, LEAD
    # positive; middle/bottom: |L/R plate Fz|, sampled at the same frame_indices
    # as the moment traces above so all three line up in time).
    l_grf = np.abs(df[LEFT_GRF_COLUMN].values.astype("float64")[frame_indices])
    r_grf = np.abs(df[RIGHT_GRF_COLUMN].values.astype("float64")[frame_indices])

    fig2, (ax2, ax_l, ax_r) = plt.subplots(3, 1, figsize=(14, 8.5), sharex=True,
                                            gridspec_kw={"height_ratios": [1.3, 1, 1]})
    ax2.axhline(0.0, color="gray", lw=0.8)
    ax2.fill_between(t, commanded_shift_ms, 0.0, where=(commanded_shift_ms < 0),
                      color="crimson", alpha=0.15, interpolate=True, label="delay")
    ax2.fill_between(t, commanded_shift_ms, 0.0, where=(commanded_shift_ms >= 0),
                      color="seagreen", alpha=0.15, interpolate=True, label="lead")
    ax2.plot(t, commanded_shift_ms, color="black", lw=1.2, label="commanded shift")
    ax2.set_ylabel("Commanded shift (ms)\ndelay (-)  /  lead (+)")
    ax2.set_title(f"Commanded Phase Shift vs. Time -- {trial_id} ({SUBJECT}, {task_label})\n"
                   f"(from theta0={THETA_COEFFS[0]:+.2f}, {N_HARMONICS} harmonic(s); "
                   f"applied directly, per-timestep -- this IS what Mechanism 1 uses, no averaging)",
                   fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(alpha=0.2)

    ax_l.plot(t, l_grf, color="darkorange", lw=1.0)
    ax_l.set_ylabel("Force (N)")
    ax_l.set_title("Left plate vertical GRF, |Fz|", fontsize=9, fontweight="bold", loc="left")
    ax_l.grid(alpha=0.2)

    ax_r.plot(t, r_grf, color="crimson", lw=1.0)
    ax_r.set_ylabel("Force (N)")
    ax_r.set_title("Right plate vertical GRF, |Fz|", fontsize=9, fontweight="bold", loc="left")
    ax_r.set_xlabel("Time (s)")
    ax_r.grid(alpha=0.2)
    fig2.tight_layout()

    print("[PhaseAmp] Displaying figures (close the windows, or Ctrl+C, to exit).")
    plt.show()


if __name__ == "__main__":
    main()
