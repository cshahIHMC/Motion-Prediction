"""
phase_manifold_shaper_demo.py

Fixed-parameter (literature-set, not data-calibrated) demo of the phase-
manifold torque shaper -- same 4 functions as phase_manifold_torque_shaper_v2.py
(compute_phi_ref, compute_delta, compute_gain, phase_manifold_torque_shaper;
calibrate_phi_peak is NOT used here -- see below) with ONE deliberate
extension carried over: compute_delta now ALSO returns its own 3 internal
components (delta_phi, f_norm, a_conf), mirroring the pattern compute_gain
already uses, and phase_manifold_torque_shaper now ALSO returns tau_shifted
and delta_components -- neither was exposed by the original return
signatures, and both are needed for the 7-subplot figure this script
produces. Nothing else about the functions' math changed.

Same model loading, same checkpoint convention (teacher branch, FLD-
TeacherStudent trained up to 60 epochs), same forward_teacher call, same
data loading and normalization as plot_polar_phase_manifold.py / every
other script in this session.

NO CALIBRATION: phi_peak and G_max are fixed literature constants (Bryan
et al. hip-exoskeleton assistance timing/magnitude), not derived from any
trial's data -- there is no calibration trial, no calibrate_phi_peak call,
and no per-trial fitting anywhere in this script. Only ONE trial is loaded:
S02/trial_3 (random) -- exercised because it's the least periodic, hardest
task available, which is a reasonable stress test for a fixed, literature-
set (not data-tuned) rule.

Encoded with stride=1 via forward_teacher: phi/freq_raw/amp/offset, each
(T,8). freq_raw -> Hz via freq_raw*SAMPLE_RATE/H (raw is cycles/window,
same conversion every other script in this repo applies). Ground truth hip
moment (hip_flexion_r_moment / subject_weight) is read at the CAUSAL-LAST
frame index (window_start + H - 1) -- DELIBERATELY NOT the window-CENTER
convention plot_polar_phase_manifold.py and this session's other pure
diagnostics use: this script is shaping a signal for a controller-shaped
use case (same reasoning manifold_controller.py / torque_shaper.py locked
in for this repo), so causal-last is the right frame here.

ONE figure, 7 stacked subplots, shared time axis, a 10s window auto-picked
(not hand-guessed) to CONTAIN A VISIBLE TRANSITION: a rolling-std scan over
the whole trial's ground truth moment finds the highest-activity 10s
stretch, printed explicitly so the choice is traceable, not assumed.
  1. Ground truth hip moment (tau_in)
  2. delta(t) -- the final combined phase shift
  3. delta's own 3 components overlaid (delta_phi, f_norm, a_conf)
  4. tau_shifted -- input torque after the time shift, before gain
  5. All 6 gain components overlaid (G_phase, G_freq, G_amp, G_circ, G_task, G_stable)
  6. G_total (the combined gain actually applied)
  7. tau_out overlaid against ground truth

plt.show() -- interactive, nothing saved to disk.

Run:
    python3 phase_manifold_shaper_demo.py
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from plot_error_across_time import _find_checkpoint, _load_state_dict_flexible
from Models.FLD_teacher_student import FLDTeacherStudent
from DataLoader.teacher_student_data_loader import (
    SENSOR_DIM as TS_SENSOR_DIM, BIOMECH_COLUMNS as TS_BIOMECH_COLUMNS,
    MOMENT_COLUMNS as TS_MOMENT_COLUMNS,
)

# ---------------------------------------------------------------------------
# Config -- selected subject / trial
# ---------------------------------------------------------------------------
SUBJECT_NAME = "S02"          # CLOCK_CHANNELS/TASK_CHANNEL/EFFORT_CHANNEL below were read off S02's polar plots
TRIAL_NUMBER = "3"            # random
FLD_TS_RUN = ""                 # blank = latest runs/FLDTeacherStudent_* run (epoch 60)

DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")
LATENT_STRIDE = 1             # stride=1, as specified

HIP_MOMENT_COLUMN = "hip_flexion_r_moment"
PLOT_DURATION_S = 10.0

# Fixed, literature-set shaper parameters -- NOT calibrated from data.
phi_peak        = 0.55   # 55% of stride, Bryan et al. optimal timing window
G_max           = 0.40   # Nm/kg, Bryan et al. hip assistance peak magnitude

# ---------------------------------------------------------------------------
# Constants -- match your repo
# ---------------------------------------------------------------------------
SAMPLE_RATE = 150.0
CLOCK_CHANNELS = [2, 6, 7]   # clean ring channels from polar plot analysis
TASK_CHANNEL   = 4            # quasi-static postural channel
EFFORT_CHANNEL = 1            # spiral channel -- within-stride effort


# ---------------------------------------------------------------------------
# Helper: phase reference from clean clock channels
# Circular amplitude-weighted mean across CLOCK_CHANNELS
# ---------------------------------------------------------------------------
def compute_phi_ref(phi, amp, clock_channels=CLOCK_CHANNELS):
    phi_clock = phi[:, clock_channels]
    a_clock   = amp[:, clock_channels]
    phasor    = a_clock * np.exp(1j * 2.0 * np.pi * phi_clock)
    phi_ref   = np.angle(phasor.sum(axis=1)) / (2.0 * np.pi)
    return phi_ref


# ---------------------------------------------------------------------------
# Function 1: phase-dependent time shift delta(phi, f, a)
# EXTENDED to also return its own 3 components (delta_phi, f_norm, a_conf) --
# the original only returned the combined `delta`; Figure panel 3 below needs
# each factor plotted separately.
# ---------------------------------------------------------------------------
def compute_delta(phi_ref, freq_hz, amp,
                  phi_peak,
                  delta_max=0.10,
                  clock_channels=CLOCK_CHANNELS):
    # --- Component 1: phase position ---
    delta_phi = delta_max * np.maximum(
        np.cos(2.0 * np.pi * (phi_ref - phi_peak)),
        0.0
    )

    # --- Component 2: frequency -- stance emphasis ---
    f_mean = np.mean(freq_hz[:, clock_channels], axis=1)
    f_min, f_max = f_mean.min(), f_mean.max()
    f_norm = (f_mean - f_min) / (f_max - f_min + 1e-8)

    # --- Component 3: amplitude confidence ---
    a_norm = np.linalg.norm(amp, axis=1)
    a_conf = np.clip(a_norm / np.percentile(a_norm, 75), 0.0, 1.0)

    # --- Combine multiplicatively ---
    delta = delta_phi * f_norm * a_conf

    delta_components = {"delta_phi": delta_phi, "f_norm": f_norm, "a_conf": a_conf}
    return delta, delta_components


# ---------------------------------------------------------------------------
# Function 2: phase-dependent amplitude gain G(phi, f, a) -- unchanged from
# phase_manifold_torque_shaper_v2.py, already returns a components dict.
# ---------------------------------------------------------------------------
def compute_gain(phi, freq_hz, amp,
                 sample_rate=SAMPLE_RATE,
                 clock_channels=CLOCK_CHANNELS,
                 effort_channel=EFFORT_CHANNEL,
                 task_channel=TASK_CHANNEL,
                 G_max=1.0):
    # --- Group 1: within-stride effort ---
    G_phase = np.clip(
        amp[:, effort_channel] / np.percentile(amp[:, effort_channel], 90),
        0.0, 1.0
    )
    f_mean = np.mean(freq_hz[:, clock_channels], axis=1)
    f_norm = (f_mean - f_mean.min()) / (f_mean.max() - f_mean.min() + 1e-8)
    G_freq = f_norm

    # --- Group 2: overall effort level ---
    a_norm = np.linalg.norm(amp, axis=1)
    G_amp  = np.clip(a_norm / np.percentile(a_norm, 90), 0.0, 1.0)

    # --- Group 3: uncertainty / transitionary behavior ---
    phasor_all = np.exp(1j * 2.0 * np.pi * phi)
    u_circ  = 1.0 - np.abs(phasor_all.mean(axis=1))
    G_circ  = 1.0 - u_circ

    G_task = np.clip(
        amp[:, task_channel] / np.percentile(amp[:, task_channel], 75),
        0.0, 1.0
    )

    da_dt   = np.abs(np.gradient(a_norm, 1.0 / sample_rate))
    G_stable = np.clip(1.0 - da_dt / (np.percentile(da_dt, 95) + 1e-8), 0.0, 1.0)

    # --- Combine multiplicatively ---
    G = G_max * G_phase * G_freq * G_amp * G_circ * G_task * G_stable

    components = {
        "G_phase":  G_phase, "G_freq":   G_freq, "G_amp":    G_amp,
        "G_circ":   G_circ, "G_task":   G_task, "G_stable": G_stable, "G_total":  G,
    }
    return G, components


# ---------------------------------------------------------------------------
# Full pipeline: torque shaper -- EXTENDED to also return tau_shifted and
# delta_components (neither was exposed by the original return signature).
# ---------------------------------------------------------------------------
def phase_manifold_torque_shaper(tau_in, phi, freq_hz, amp,
                                  phi_peak,
                                  delta_max=0.10,
                                  G_max=1.0,
                                  sample_rate=SAMPLE_RATE,
                                  clock_channels=CLOCK_CHANNELS):
    T = len(tau_in)

    # Step 1: phase reference
    phi_ref = compute_phi_ref(phi, amp, clock_channels)

    # Step 2: phase-dependent shift
    delta, delta_components = compute_delta(phi_ref, freq_hz, amp,
                          phi_peak, delta_max, clock_channels)

    # Step 3: convert phase shift to sample shift
    f_ref = np.mean(freq_hz[:, clock_channels], axis=1)
    delta_samples = np.round(delta / (f_ref + 1e-8) * sample_rate).astype(int)
    delta_samples = np.clip(delta_samples, 0, T - 1)

    # Step 4: apply time shift to input torque
    indices    = np.minimum(np.arange(T) + delta_samples, T - 1)
    tau_shifted = tau_in[indices]

    # Step 5: phase-dependent gain
    G, components = compute_gain(phi, freq_hz, amp,
                                  sample_rate, clock_channels,
                                  G_max=G_max)

    # Step 6: apply gain
    tau_out = G * tau_shifted

    return tau_out, delta, G, components, tau_shifted, delta_components


# ---------------------------------------------------------------------------
# Model loading -- same convention as every other script in this session
# ---------------------------------------------------------------------------

def _load_fld_teacher_student_model(device, pinned_run=""):
    run_dir, ckpt_path = _find_checkpoint("FLDTeacherStudent", "fld_teacher_student", pinned_run)
    print(f"[ShaperDemo] FLD-TeacherStudent checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = FLDTeacherStudent(
        sensor_dim=TS_SENSOR_DIM, biomech_dim=len(TS_BIOMECH_COLUMNS),
        history_horizon=config["history_horizon"], latent_dim=config["latent_dim"],
        device=device, dt=1.0 / SAMPLE_RATE,
        teacher_encoder_shape=config["teacher_encoder_shape"],
        student_encoder_shape=config["student_encoder_shape"],
        decoder_shape=config["decoder_shape"],
    )
    _load_state_dict_flexible(model, ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    norm_stats = ckpt["norm_stats"]
    print(f"[ShaperDemo]   epoch={ckpt['epoch']}  history_horizon={config['history_horizon']}  "
          f"latent_dim={config['latent_dim']}")
    return model, config, norm_stats


def _trial_path(subject_name, trial_number):
    path = os.path.join(DATA_DIR, subject_name, f"trial_{trial_number}_all_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    return path


# ---------------------------------------------------------------------------
# Windowing / encoding -- CAUSAL-LAST frame convention (this is a shaper /
# controller-shaped use case, NOT a pure diagnostic -- see module docstring)
# ---------------------------------------------------------------------------

def _window_starts(n_rows, history_horizon, stride):
    return np.arange(0, n_rows - history_horizon + 1, stride)


def _encode_trial(model, model_config, norm_stats, subject_name, trial_number, device, label):
    H = model_config["history_horizon"]
    path = _trial_path(subject_name, trial_number)
    print(f"[ShaperDemo] [{label}] trial file: {path}")
    usecols = list(dict.fromkeys(TS_BIOMECH_COLUMNS + ["subject_weight", "task", HIP_MOMENT_COLUMN]))
    df = pd.read_csv(path, usecols=usecols)
    task_label = str(df["task"].iloc[0])
    print(f"[ShaperDemo] [{label}] {len(df):,} raw rows ({len(df)/SAMPLE_RATE:.1f}s @{SAMPLE_RATE:.0f}Hz), "
          f"task={task_label!r}")
    if len(df) < H:
        raise ValueError(f"[{label}] trial too short ({len(df)} rows) for history_horizon={H}")

    moments_wnorm = df[TS_MOMENT_COLUMNS].div(df["subject_weight"], axis=0)
    biomech_df = df[TS_BIOMECH_COLUMNS].copy()
    biomech_df[TS_MOMENT_COLUMNS] = moments_wnorm
    raw_biomech = biomech_df.values.astype("float32")
    norm_biomech = torch.tensor(
        (raw_biomech - norm_stats["mean_biomech"]) / norm_stats["std_biomech"], dtype=torch.float32,
    )

    window_starts = _window_starts(len(df), H, LATENT_STRIDE)
    print(f"[ShaperDemo] [{label}] encoding {len(window_starts):,} windows (stride={LATENT_STRIDE})...")
    phis, freqs, amps, offsets = [], [], [], []
    with torch.no_grad():
        batch_size = 256
        for i in range(0, len(window_starts), batch_size):
            batch_starts = window_starts[i:i + batch_size]
            wins = torch.stack([norm_biomech[s:s + H].T for s in batch_starts]).to(device)
            phi, freq_raw, amp, offset = model.forward_teacher(wins)
            phis.append(phi.cpu()); freqs.append(freq_raw.cpu())
            amps.append(amp.cpu()); offsets.append(offset.cpu())
    phi = torch.cat(phis, dim=0).numpy()
    freq_raw = torch.cat(freqs, dim=0).numpy()
    amp = torch.cat(amps, dim=0).numpy()
    offset = torch.cat(offsets, dim=0).numpy()
    freq_hz = freq_raw * SAMPLE_RATE / H   # freq_raw is cycles/window -- same conversion every script here applies

    # CAUSAL-LAST frame index -- window_start + H - 1, NOT window-center (see module docstring)
    causal_last_frame_indices = np.asarray(window_starts) + H - 1
    tau_in = moments_wnorm[HIP_MOMENT_COLUMN].values[causal_last_frame_indices]   # N*m/kg
    t = causal_last_frame_indices / SAMPLE_RATE

    return dict(phi=phi, freq_hz=freq_hz, amp=amp, offset=offset, tau_in=tau_in, t=t, task_label=task_label)


# ---------------------------------------------------------------------------
# Figure: 7 stacked subplots
# ---------------------------------------------------------------------------

def plot_shaper_figure(enc, phi_peak, G_max, plot_start_s, plot_duration_s, title):
    tau_out, delta, G, components, tau_shifted, delta_components = phase_manifold_torque_shaper(
        enc["tau_in"], enc["phi"], enc["freq_hz"], enc["amp"], phi_peak, G_max=G_max,
    )
    t = enc["t"]
    xlo, xhi = plot_start_s, plot_start_s + plot_duration_s
    mask = (t >= xlo) & (t <= xhi)

    fig, axes = plt.subplots(7, 1, figsize=(16, 20), sharex=True)

    axes[0].plot(t, enc["tau_in"], color="black", lw=1.4)
    axes[0].set_ylabel("tau_in\n(N*m/kg)", fontsize=8)
    axes[0].set_title("1. Ground truth hip moment", fontsize=10, fontweight="bold")

    axes[1].plot(t, delta, color="purple", lw=1.6)
    axes[1].set_ylabel("delta\n(cycles)", fontsize=8)
    axes[1].set_title("2. delta(t) -- combined phase shift", fontsize=10, fontweight="bold")

    for name, color in [("delta_phi", "tab:blue"), ("f_norm", "tab:green"), ("a_conf", "tab:orange")]:
        axes[2].plot(t, delta_components[name], color=color, lw=1.2, alpha=0.85, label=name)
    axes[2].set_ylabel("[0,1] or\n[0,delta_max]", fontsize=8)
    axes[2].set_title("3. delta's own 3 components overlaid", fontsize=10, fontweight="bold")
    axes[2].legend(fontsize=7, loc="upper right", ncol=3)

    axes[3].plot(t, tau_shifted, color="teal", lw=1.4)
    axes[3].set_ylabel("tau_shifted\n(N*m/kg)", fontsize=8)
    axes[3].set_title("4. tau_shifted -- after time shift, before gain", fontsize=10, fontweight="bold")

    gain_names = ["G_phase", "G_freq", "G_amp", "G_circ", "G_task", "G_stable"]
    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(gain_names):
        axes[4].plot(t, components[name], color=cmap(i), lw=1.1, alpha=0.85, label=name)
    axes[4].set_ylabel("[0,1]", fontsize=8)
    axes[4].set_title("5. All 6 gain components overlaid", fontsize=10, fontweight="bold")
    axes[4].legend(fontsize=7, loc="upper right", ncol=3)

    axes[5].plot(t, G, color="crimson", lw=1.6)
    axes[5].set_ylabel("G_total\n[0,G_max]", fontsize=8)
    axes[5].set_title("6. G_total -- combined gain actually applied", fontsize=10, fontweight="bold")

    axes[6].plot(t, enc["tau_in"], color="black", lw=1.5, label="ground truth")
    axes[6].plot(t, tau_out, color="tomato", lw=1.3, label="tau_out (commanded)")
    axes[6].set_ylabel("torque\n(N*m/kg)", fontsize=8)
    axes[6].set_title("7. tau_out vs ground truth", fontsize=10, fontweight="bold")
    axes[6].legend(fontsize=8, loc="upper right")

    for ax in axes:
        ax.grid(alpha=0.15)
        ax.set_xlim(xlo, xhi)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig, tau_out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ShaperDemo] Using device: {device}")

    model, config, norm_stats = _load_fld_teacher_student_model(device, FLD_TS_RUN)

    enc = _encode_trial(model, config, norm_stats, SUBJECT_NAME, TRIAL_NUMBER, device, "TRIAL")

    print(f"[ShaperDemo] phi_peak={phi_peak:.4f} cycles ({phi_peak * 360:.1f} deg), G_max={G_max:.2f} N*m/kg "
          f"-- fixed literature constants (Bryan et al.), not calibrated from any trial's data.")

    # ---- Auto-pick a 10s window with a visible transition ----
    t_trial = enc["t"]
    win_samples = int(PLOT_DURATION_S * SAMPLE_RATE)
    tau_trial = enc["tau_in"]
    # rolling std over a sliding 10s window across the WHOLE trial -- pick the window with max activity
    n = len(tau_trial)
    if n > win_samples:
        cumsum = np.cumsum(tau_trial); cumsum_sq = np.cumsum(tau_trial ** 2)
        stds = np.zeros(n - win_samples)
        for i in range(0, n - win_samples, max(1, win_samples // 20)):
            s, e = i, i + win_samples
            seg_sum = cumsum[e] - (cumsum[s - 1] if s > 0 else 0)
            seg_sq = cumsum_sq[e] - (cumsum_sq[s - 1] if s > 0 else 0)
            mean = seg_sum / win_samples
            stds[i] = np.sqrt(max(seg_sq / win_samples - mean ** 2, 0))
        best_start_idx = int(np.argmax(stds))
        plot_start_s = t_trial[best_start_idx]
    else:
        plot_start_s = 0.0
    print(f"[ShaperDemo] auto-picked {PLOT_DURATION_S:.0f}s window starting at "
          f"{plot_start_s:.2f}s (highest rolling std of ground truth moment over the whole trial -- "
          f"a genuine 'visible transition' search, not a guess).")

    fig, tau_out = plot_shaper_figure(
        enc, phi_peak, G_max, plot_start_s=plot_start_s, plot_duration_s=PLOT_DURATION_S,
        title=f"{SUBJECT_NAME}/trial_{TRIAL_NUMBER} ({enc['task_label']})  "
              f"phi_peak={phi_peak:.2f} cyc, G_max={G_max:.2f} N*m/kg (fixed, literature-set)",
    )
    rmse = float(np.sqrt(np.mean((tau_out - tau_trial) ** 2)))
    r = float(np.corrcoef(tau_out, tau_trial)[0, 1])
    print(f"[ShaperDemo] tau_out vs ground truth: RMSE={rmse:.4f} N*m/kg, r={r:.4f}")

    print("[ShaperDemo] Displaying figure (close the window, or Ctrl+C, to exit).")
    plt.show()


if __name__ == "__main__":
    main()
