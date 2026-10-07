"""
analyze_latent_synergies.py

For ONE trial (SUBJECT_NAME/TRIAL_NUMBER below) of the HELD-OUT subject S05
(HELD_OUT_DATA_PATH = Data/all_data_S05_150_corrected.csv -- never trained on;
inputs normalized with the checkpoint's own train-split norm_stats), encodes
EVERY window in that trial through the TEACHER branch of the latest trained
FLD-TeacherStudent checkpoint (runs/FLDTeacherStudent_20260921_163758,
pinned below -- 70 completed epochs, history_horizon=151/forecast_horizon=20,
trained on Data/all_data_150_corrected.csv via train_fld_teacher_student.py;
teacher_encoder_shape=student_encoder_shape=[64, 32], decoder_shape=[32, 64]
-- see that script / Models/FLD_teacher_student.py), does NOT
retrain, and produces:

  1. Amplitude per latent channel over time -- one figure, one subplot PER
     LATENT CHANNEL (latent_dim subplots, shared time x-axis).
  2. Frequency per latent channel over time -- same layout.
  3. SIX subplots (one figure, shared time x-axis, EACH AT ITS OWN NATURAL
     SCALE -- no normalization/overlay):
       a) Average amplitude, mean_i(a_i(t)) -- the plain unweighted average.
       b) Amplitude-weighted average frequency, at each timestep t:
            w_i(t)      = a_i(t) / (sum_j a_j(t) + eps)   -- normalized amplitude weights
            combined(t) = sum_i w_i(t) * freq_i(t)         -- amplitude-weighted average frequency
          (`eps` guards the near-all-zero-amplitude case.) Whichever channel
          is "doing the work" right now (largest amplitude) dominates the
          combined value; quiet channels contribute almost nothing.
          amplitude_weighted_average() is a small reusable helper (not
          hardcoded to frequency) so a later per-channel signal -- e.g. a
          phase-shift or gain value -- can reuse it the same way.
       c) Left plate vertical GRF, |L_plate_Fz| (abs()'d -- *_plate_Fz is
          ~0 N unloaded and strongly NEGATIVE when loaded).
       d) Right plate vertical GRF, |R_plate_Fz|, same convention.
       e) Hip flexion/extension torque, right leg (hip_flexion_r_moment,
          N.m/kg, weight-normalized) -- OVERLAID with that SAME torque
          element-wise multiplied by (a) logistic_scale(avg_amp) and
          (b) logistic_scale(combined_freq) -- NOT the raw avg_amp/
          combined_freq values (three lines/legend entries on this one
          subplot). logistic_scale() is the 6th of bounded_transforms()'s
          six nonlinear [0,1)-bounding functions (plots 4/5 below), pulled
          out standalone so the scaling factor is bounded/self-calibrated
          rather than each signal's own raw, unbounded scale.
       f) Hip flexion/extension torque, left leg (hip_flexion_l_moment),
          same three-line overlay (torque, torque*logistic(avg_amp),
          torque*logistic(combined_freq)).
     All six share one time axis so the gait cycle (GRF) can still be
     visually compared against the latent-parameter and kinetic signals
     even though each subplot keeps its own native units/scale.
  4. Average amplitude, run through SIX nonlinear functions that bound a
     non-negative signal to [0, 1) -- one figure, one subplot per function
     (Michaelis-Menten x/(x+c), exponential 1-exp(-x/c), tanh(x/c), Hill
     x^n/(x^n+c^n), arctan (2/pi)*atan(x/c), logistic sigmoid). Unlike
     min-max normalization (plot 3's predecessor), these use a FIXED
     function shape rather than an affine rescale to this trial's own
     min/max -- see bounded_transforms() for the scale-constant choice
     (self-calibrated to this signal's own median/std by default).
  5. Amplitude-weighted average frequency, same six functions, same layout.
  6. Direct (mean/weighted-sum) vs. PCA(1) channel combination -- one figure,
     two subplots, each overlaying the DIRECT method against a PCA(1)
     alternative reduction of the same underlying per-channel matrix
     (pca_reduce_to_one() -- both min-max normalized, since PCA's own scale
     is arbitrary and only the SHAPE is being compared):
       a) Average amplitude: avg_amp (mean_i(a_i(t))) vs. PCA(1) of the raw
          (n_windows, latent_dim) amplitude matrix.
       b) Amplitude-weighted average frequency: combined_freq (the direct
          sum_i w_i(t)*f_i(t)) vs. PCA(1) of the PER-CHANNEL w_i(t)*f_i(t)
          terms (i.e. PCA replaces the SUM step, not the weighting step).
     Each subplot's legend reports PC1's own explained-variance-ratio --
     how much of the 8-channel signal that single combined value captures.

Deliberately minimal for now -- offset, NMF, PCA, and the earlier
dominant-channel/energy-gating stacked-area plot are NOT included in this
pass; add them back incrementally as needed.

Does NOT modify Models/FLD_teacher_student.py, plot_error_across_time.py, or
DataLoader/teacher_student_data_loader.py -- reuses _find_checkpoint/
_load_state_dict_flexible and that loader's SENSOR_COLUMNS/BIOMECH_COLUMNS/
MOMENT_COLUMNS constants verbatim, same convention every other
plot_pca_manifold*.py script in this repo already follows. TEACHER branch
only (this checkpoint's student encoder is still constructed, since its
weights are part of the checkpoint's state_dict and sensor_dim must match
for load_state_dict to succeed, but forward_student() is never called here).

plt.show() -- interactive, nothing saved to disk. All figures are built
first, then shown together.

Picking the trial: edit SUBJECT_NAME + TRIAL_NUMBER below -- together they form
the trial_id "<SUBJECT_NAME>_trial_<TRIAL_NUMBER>" in HELD_OUT_DATA_PATH (S05 has
trial_1, trial_1_2, trial_2 ... trial_11; _trial_df() lists the available ids
if yours isn't found).

Run:
    python3 analyze_latent_synergies.py
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(__file__))

from plot_error_across_time import _find_checkpoint, _load_state_dict_flexible
from Models.FLD_teacher_student import FLDTeacherStudent
from DataLoader.teacher_student_data_loader import SENSOR_COLUMNS, BIOMECH_COLUMNS, MOMENT_COLUMNS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FLD_TS_RUN = "FLDTeacherStudent_20260921_163758"   # pinned -- latest trained FLDTeacherStudent checkpoint,
                                                    # 70 completed epochs, history_horizon=151/forecast_horizon=20,
                                                    # trained on Data/all_data_150_corrected.csv (train_fld_teacher_student.py)

DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")
HELD_OUT_DATA_PATH = os.path.join(DATA_DIR, "all_data_150_corrected.csv")   # held-out subject, never trained on
SAMPLE_RATE = 150.0

SUBJECT_NAME = "S02"    # trial_id prefix in HELD_OUT_DATA_PATH
TRIAL_NUMBER = "3"      # <-- change to pick a different trial: trial_id = f"{SUBJECT_NAME}_trial_{TRIAL_NUMBER}"
                        #     (S05_trial_4 = walking 1.2 m/s)

LATENT_STRIDE = 1   # frames between encoder windows -- 1 = EVERY window in the trial, per request
                     # (no subsampling; a 100Hz trial has ~n_rows windows, cheap to encode in batches on GPU)

WEIGHT_EPS = 1e-8   # denominator guard for amplitude_weighted_average() -- avoids divide-by-zero
                     # when every channel's amplitude is near 0 at the same timestep

# Left/right vertical GRF, force PLATE (not insole) -- same convention every other PCA/plot
# script in this repo uses: *_plate_Fz reads ~0 N unloaded, strongly negative (down to
# ~-1000 N) when loaded, so we take abs() before plotting (see plot 3 below).
LEFT_GRF_COLUMN = "L_plate_Fz"
RIGHT_GRF_COLUMN = "R_plate_Fz"

# Hip flexion/extension torque, both legs -- already in BIOMECH_COLUMNS/MOMENT_COLUMNS
# (N.m/kg, weight-normalized -- see main()'s teacher_input_df construction below).
HIP_FLEXION_R_MOMENT_COLUMN = "hip_flexion_r_moment"
HIP_FLEXION_L_MOMENT_COLUMN = "hip_flexion_l_moment"


# ---------------------------------------------------------------------------
# Model loading -- TEACHER branch only. sensor_dim is still needed to build
# an architecturally-matching FLDTeacherStudent (the student encoder's
# weights are part of the checkpoint's state_dict even though
# forward_student() is never called below).
# ---------------------------------------------------------------------------

def _load_teacher_model(device, pinned_run=""):
    run_dir, ckpt_path = _find_checkpoint("FLDTeacherStudent", "fld_teacher_student", pinned_run)
    print(f"[LatentSynergies] FLD-TeacherStudent checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    norm_stats = ckpt["norm_stats"]   # keys: mean_sensor/std_sensor/mean_biomech/std_biomech
    model = FLDTeacherStudent(
        sensor_dim=len(SENSOR_COLUMNS), biomech_dim=len(BIOMECH_COLUMNS),
        history_horizon=config["history_horizon"], latent_dim=config["latent_dim"],
        device=device, dt=1.0 / config["sample_rate"],
        teacher_encoder_shape=config["teacher_encoder_shape"],
        student_encoder_shape=config["student_encoder_shape"],
        decoder_shape=config["decoder_shape"],
    )
    _load_state_dict_flexible(model, ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"[LatentSynergies]   epoch={ckpt['epoch']}  history_horizon={config['history_horizon']}  "
          f"latent_dim={config['latent_dim']}  sample_rate={config['sample_rate']}  "
          f"teacher_encoder_shape={config['teacher_encoder_shape']}  decoder_shape={config['decoder_shape']}")
    return model, config, norm_stats


def _trial_df(subject_name, trial_number, usecols):
    """One trial's rows from HELD_OUT_DATA_PATH, selected by trial_id."""
    if not os.path.exists(HELD_OUT_DATA_PATH):
        raise FileNotFoundError(f"{HELD_OUT_DATA_PATH} not found.")
    trial_id = f"{subject_name}_trial_{trial_number}"
    df = pd.read_csv(HELD_OUT_DATA_PATH, usecols=list(dict.fromkeys(usecols + ["trial_id"])))
    trial = df[df["trial_id"] == trial_id].reset_index(drop=True)
    if trial.empty:
        raise ValueError(f"trial_id {trial_id!r} not in {HELD_OUT_DATA_PATH} -- "
                         f"available: {df['trial_id'].unique().tolist()}")
    return trial


# ---------------------------------------------------------------------------
# Windowing / encoding
# ---------------------------------------------------------------------------

def _window_starts(n_rows, history_horizon, stride):
    """Every window start index, evenly strided -- no cap (per request:
    "go over all the windows in the trial"), just bounded so every window
    still fits (start + history_horizon <= n_rows)."""
    hard_upper = n_rows - history_horizon
    if hard_upper < 0:
        return []
    return list(range(0, hard_upper + 1, stride))


def _encode_teacher_params(model, teacher_input_norm, history_horizon, window_starts, device, batch_size=256):
    """Slide the TEACHER encoder across every window start, batched.

    Returns:
        phase, frequency, amplitude, offset -- each (n_windows, latent_dim) np.ndarray.
    """
    phases, freqs, amps, offsets = [], [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(window_starts), batch_size):
            batch_starts = window_starts[i:i + batch_size]
            wins = torch.stack(
                [teacher_input_norm[s:s + history_horizon].T for s in batch_starts]
            ).to(device)
            phase, freq, amp, offset = model.forward_teacher(wins)
            phases.append(phase.cpu()); freqs.append(freq.cpu())
            amps.append(amp.cpu()); offsets.append(offset.cpu())
    return (
        torch.cat(phases, dim=0).numpy(), torch.cat(freqs, dim=0).numpy(),
        torch.cat(amps, dim=0).numpy(), torch.cat(offsets, dim=0).numpy(),
    )


# ---------------------------------------------------------------------------
# Amplitude-weighted combination -- generic, not hardcoded to frequency.
# ---------------------------------------------------------------------------

def amplitude_weighted_average(amp, x, eps=WEIGHT_EPS):
    """Combine a per-channel signal x_i(t) into one value per timestep,
    weighted by each channel's SHARE of the total amplitude at that instant.

        w_i(t)      = a_i(t) / (sum_j a_j(t) + eps)
        combined(t) = sum_i w_i(t) * x_i(t)

    Whichever channel has the most amplitude "right now" dominates the
    combined value; channels with near-zero amplitude contribute almost
    nothing. `eps` guards the (rare) all-amplitudes-near-0 timestep.

    Args:
        amp: (n_windows, latent_dim) -- per-channel amplitude, a_i(t).
        x:   (n_windows, latent_dim) -- the per-channel signal to combine
             (frequency below, but could equally be a phase-shift or gain
             value computed per channel).
        eps: denominator guard.

    Returns:
        (n_windows,) -- combined(t).
    """
    weights = amp / (amp.sum(axis=1, keepdims=True) + eps)   # (n_windows, latent_dim), rows sum to ~1
    return (weights * x).sum(axis=1)


def pca_reduce_to_one(matrix, reference=None):
    """Reduce an (n_windows, n_channels) matrix to ONE combined (n_windows,)
    signal via its first principal component -- an alternative to a plain
    mean/weighted-sum across channels (see main()'s pca_amp/pca_weighted_freq
    below, compared directly against avg_amp/combined_freq).

    PCA's sign is arbitrary (PC1 and -PC1 explain identical variance) -- if
    `reference` is given, the component is flipped so it's positively
    correlated with it, so e.g. the PCA-combined amplitude visually tracks
    the same up/down shape as the plain channel mean instead of an
    arbitrary mirror image of it.

    Args:
        matrix: (n_windows, n_channels).
        reference: optional (n_windows,) signal to sign-align PC1 against.

    Returns:
        (pc1, explained_variance_ratio) -- pc1 is (n_windows,); the ratio is
        PC1's own share of the matrix's total variance (how much of the
        8-channel signal this single combined value actually captures).
    """
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(matrix).ravel()
    explained = float(pca.explained_variance_ratio_[0])
    if reference is not None and np.corrcoef(pc1, reference)[0, 1] < 0:
        pc1 = -pc1
    return pc1, explained


# ---------------------------------------------------------------------------
# Plot 1/2 -- one figure per parameter, one subplot per latent channel
# ---------------------------------------------------------------------------

def _plot_param_per_channel(times, values, latent_dim, param_name, unit, trial_label):
    fig, axes = plt.subplots(latent_dim, 1, figsize=(14, 1.7 * latent_dim), sharex=True)
    axes = np.atleast_1d(axes)
    for ch in range(latent_dim):
        ax = axes[ch]
        ax.plot(times, values[:, ch], lw=0.7, color="tab:blue")
        ax.set_ylabel(f"ch{ch}", fontsize=8, rotation=0, labelpad=22, va="center")
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Teacher {param_name}{f' ({unit})' if unit else ''} per Latent Channel — {trial_label}\n"
                 f"{len(times):,} windows", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 3 -- average amplitude, amplitude-weighted average frequency, L/R
# vertical GRF, and L/R hip flexion/extension torque -- SIX subplots, one
# signal per subplot, EACH IN ITS OWN NATURAL SCALE/UNITS (no normalization,
# unlike the earlier overlaid version this replaces) -- sharing one time
# x-axis so they still line up for visual comparison.
# ---------------------------------------------------------------------------

def _plot_combined_signals_subplots(times, signals, trial_label):
    """
    Args:
        signals: list of (title, ylabel, lines) tuples, one subplot per
            entry, top to bottom, each at its own native scale. `lines` is
            itself a list of (label, values, color) tuples plotted TOGETHER
            on that one subplot -- a single-element list for a plain
            one-trace subplot, more for an overlay (e.g. the hip torque
            subplots below, which also overlay torque scaled by the
            amplitude/frequency signals) -- a legend is added only when a
            subplot actually has more than one line.
    """
    n = len(signals)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.2 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (title, ylabel, lines) in zip(axes, signals):
        for label, values, color in lines:
            ax.plot(times, values, lw=0.8, color=color, alpha=0.85 if len(lines) > 1 else 1.0,
                     label=label if len(lines) > 1 else None)
        ax.set_title(title, fontsize=9, fontweight="bold", loc="left")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7)
        if len(lines) > 1:
            ax.legend(fontsize=7, loc="upper right", ncol=len(lines))
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Latent Combination Signals vs. Gait Kinetics — {trial_label}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 6 -- direct (mean / amplitude-weighted-sum) vs. PCA(1) channel
# combination, one comparison per signal. PCA's own scale is arbitrary (a
# linear combination of channels, not itself a mean or a sum), so unlike
# every other plot in this script these two lines are min-max normalized
# before overlaying -- otherwise one trace could dwarf the other for a
# reason that has nothing to do with their actual SHAPE, which is the only
# thing being compared here.
# ---------------------------------------------------------------------------

def _minmax(x):
    lo, hi = x.min(), x.max()
    if hi > lo:
        return (x - lo) / (hi - lo)
    return np.zeros_like(x)   # degenerate constant signal -- avoid dividing by zero


def _plot_direct_vs_pca(times, avg_amp, pca_amp, amp_evr, combined_freq, pca_weighted_freq, freq_evr, trial_label):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(times, _minmax(avg_amp), lw=0.8, color="tab:green",
                 label=r"Direct mean, $\mathrm{mean}_i(a_i(t))$")
    axes[0].plot(times, _minmax(pca_amp), lw=0.8, color="black", alpha=0.75,
                 label=f"PCA(1) of amplitude channels ({amp_evr * 100:.1f}% variance explained)")
    axes[0].set_title("Average Amplitude — direct mean vs. PCA(1), both min-max normalized",
                       fontsize=10, fontweight="bold", loc="left")
    axes[0].set_ylabel("Normalized [0, 1]", fontsize=8)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].grid(alpha=0.2)

    axes[1].plot(times, _minmax(combined_freq), lw=0.8, color="tab:purple",
                 label=r"Direct weighted sum, $\sum_i w_i(t)\,f_i(t)$")
    axes[1].plot(times, _minmax(pca_weighted_freq), lw=0.8, color="black", alpha=0.75,
                 label=f"PCA(1) of per-channel $w_i(t)\\,f_i(t)$ terms ({freq_evr * 100:.1f}% variance explained)")
    axes[1].set_title("Amplitude-Weighted Average Frequency — direct weighted sum vs. PCA(1), "
                       "both min-max normalized", fontsize=10, fontweight="bold", loc="left")
    axes[1].set_ylabel("Normalized [0, 1]", fontsize=8)
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(alpha=0.2)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Direct Channel Combination vs. PCA(1) Channel Combination — {trial_label}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plots 4/5 -- six nonlinear [0, 1)-bounding functions applied to a single
# non-negative signal (average amplitude / weighted-average frequency),
# one figure per signal, one subplot per function. Every function needs a
# scale reference -- `c` (Michaelis-Menten/exponential/tanh/Hill/arctan) and
# (x0, k) (logistic) below all default to the SIGNAL'S OWN median/std, a
# simple self-calibrating choice (f(median) = 0.5) rather than a hand-picked
# constant -- override via the c/n/x0/k kwargs if a fixed, cross-trial
# reference is wanted instead.
# ---------------------------------------------------------------------------

def logistic_scale(x, x0=None, k=None):
    """Logistic sigmoid, bounding a signal x(t) into (0, 1): the 6th of
    bounded_transforms()'s six functions, pulled out on its own so it can be
    reused directly (e.g. to scale a signal before multiplying it into
    another one -- see main()'s hip-torque overlay below) without going
    through the full six-function dict.

    Args:
        x: (n_windows,) signal.
        x0: midpoint (f(x0) = 0.5). Defaults to median(x).
        k: steepness. Defaults to 1/std(x).

    Returns:
        (logistic, x0, k) -- the transformed (n_windows,) array, plus the
        actual constants used (handy for annotating a plot), after the
        zero/degenerate std guard.
    """
    if x0 is None:
        x0 = float(np.median(x))
    if k is None:
        spread = float(np.std(x))
        k = 1.0 / spread if spread > 0 else 1.0
    return 1.0 / (1.0 + np.exp(-k * (x - x0))), x0, k


def bounded_transforms(x, c=None, n=2.0, x0=None, k=None):
    """Six nonlinear functions mapping a non-negative signal x(t) into [0, 1).

    Args:
        x: (n_windows,) non-negative signal (amplitude, frequency, ...).
        c: scale/half-saturation constant for the first five functions
           (f(c) = 0.5 for Michaelis-Menten/exponential/tanh/arctan; Hill's
           midpoint too). Defaults to median(x).
        n: Hill function steepness exponent. Default 2 (moderate steepness).
        x0, k: logistic sigmoid midpoint/steepness (see logistic_scale()).
           Default to median(x) and 1/std(x) respectively.

    Returns:
        (transforms, c, n, x0, k) -- transforms is an ordered dict of
        {label: (n_windows,) array}; the rest are the actual constants used
        (handy for annotating a plot title), after zero/degenerate guards.
    """
    if c is None:
        c = float(np.median(x))
    if c <= 0:
        c = float(x.mean()) if x.mean() > 0 else 1.0
    logistic, x0, k = logistic_scale(x, x0, k)

    transforms = {
        "Michaelis-Menten:  x / (x + c)": x / (x + c),
        "Exponential:  1 - exp(-x / c)": 1.0 - np.exp(-x / c),
        "tanh:  tanh(x / c)": np.tanh(x / c),
        f"Hill (n={n:g}):  x^n / (x^n + c^n)": (x ** n) / (x ** n + c ** n),
        "Arctan:  (2/π) atan(x / c)": (2.0 / np.pi) * np.arctan(x / c),
        "Logistic:  1 / (1 + exp(-k(x - x0)))": logistic,
    }
    return transforms, c, n, x0, k


def _plot_bounded_transforms(times, x, signal_name, trial_label):
    transforms, c, n, x0, k = bounded_transforms(x)
    rows = len(transforms)
    fig, axes = plt.subplots(rows, 1, figsize=(14, 2.0 * rows), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (label, y) in zip(axes, transforms.items()):
        ax.plot(times, y, lw=0.8, color="tab:blue")
        ax.set_title(label, fontsize=9, fontweight="bold", loc="left")
        ax.set_ylabel("[0, 1)", fontsize=8)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{signal_name} — Bounded to [0, 1) via Six Nonlinear Saturating Functions\n"
                 f"(c={c:.3g}, Hill n={n:g}, logistic x0={x0:.3g}/k={k:.3g} — all self-calibrated "
                 f"from this signal's own median/std)  —  {trial_label}",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LatentSynergies] Using device: {device}")

    model, config, norm_stats = _load_teacher_model(device, FLD_TS_RUN)
    history_horizon = config["history_horizon"]
    latent_dim = config["latent_dim"]
    sample_rate = config["sample_rate"]

    trial_label = f"{SUBJECT_NAME}/trial_{TRIAL_NUMBER}"
    print(f"[LatentSynergies] Held-out data: {HELD_OUT_DATA_PATH}  (trial {SUBJECT_NAME}_trial_{TRIAL_NUMBER})")

    usecols = list(dict.fromkeys(BIOMECH_COLUMNS + ["subject_weight", LEFT_GRF_COLUMN, RIGHT_GRF_COLUMN]))
    df = _trial_df(SUBJECT_NAME, TRIAL_NUMBER, usecols)
    print(f"[LatentSynergies] {trial_label}: {len(df):,} raw rows ({len(df) / sample_rate:.1f}s @{sample_rate:.0f}Hz)")

    # Teacher input == BIOMECH_COLUMNS -- only the moment sub-columns need
    # weight-normalizing (N.m -> N.m/kg), same convention every other script here uses.
    moments_wnorm = df[MOMENT_COLUMNS].div(df["subject_weight"], axis=0)
    teacher_input_df = df[BIOMECH_COLUMNS].copy()
    teacher_input_df[MOMENT_COLUMNS] = moments_wnorm
    raw_teacher_input = teacher_input_df.values.astype("float32")
    norm_teacher_input = torch.tensor(
        (raw_teacher_input - norm_stats["mean_biomech"]) / norm_stats["std_biomech"], dtype=torch.float32,
    )

    window_starts = _window_starts(len(df), history_horizon, LATENT_STRIDE)
    if not window_starts:
        raise RuntimeError(f"{trial_label} too short ({len(df)} rows) for history_horizon={history_horizon}.")
    print(f"[LatentSynergies] {trial_label}: encoding {len(window_starts):,} windows (stride={LATENT_STRIDE}, "
          f"i.e. every window in the trial) through the TEACHER branch...")

    _phase, freq, amp, _offset = _encode_teacher_params(
        model, norm_teacher_input, history_horizon, window_starts, device,
    )

    half = (history_horizon - 1) // 2
    centers = np.asarray(window_starts) + half
    times = centers / sample_rate
    l_grf = df[LEFT_GRF_COLUMN].values.astype("float64")[centers]
    r_grf = df[RIGHT_GRF_COLUMN].values.astype("float64")[centers]
    # Weight-normalized (N.m/kg) torque -- read from teacher_input_df, which already has
    # MOMENT_COLUMNS substituted with moments_wnorm above (same values fed to the encoder).
    hip_r_torque = teacher_input_df[HIP_FLEXION_R_MOMENT_COLUMN].values.astype("float64")[centers]
    hip_l_torque = teacher_input_df[HIP_FLEXION_L_MOMENT_COLUMN].values.astype("float64")[centers]

    # ---- 1/2: per-channel time series ----
    _plot_param_per_channel(times, amp, latent_dim, "Amplitude", "", trial_label)
    _plot_param_per_channel(times, freq, latent_dim, "Frequency", "cycles/window", trial_label)

    # ---- 3: average amplitude, amplitude-weighted average frequency, L/R vertical GRF,
    # and L/R hip flexion/extension torque -- six subplots, each at its own natural scale.
    # The two torque subplots additionally overlay torque SCALED BY avg_amp/combined_freq
    # -- NOT the raw values (that direct multiplication is removed), but each first passed
    # through logistic_scale() (the 6th of bounded_transforms()'s six nonlinear [0,1)-bounding
    # functions) so the scaling factor is bounded and self-calibrated rather than the signal's
    # own raw, unbounded scale. ----
    avg_amp = amp.mean(axis=1)
    combined_freq = amplitude_weighted_average(amp, freq)
    logistic_amp, _amp_x0, _amp_k = logistic_scale(avg_amp)
    logistic_freq, _freq_x0, _freq_k = logistic_scale(combined_freq)
    hip_r_torque_x_amp = hip_r_torque * logistic_amp
    hip_r_torque_x_freq = hip_r_torque * logistic_freq
    hip_l_torque_x_amp = hip_l_torque * logistic_amp
    hip_l_torque_x_freq = hip_l_torque * logistic_freq
    signals = [
        (r"Average amplitude, $\mathrm{mean}_i(a_i(t))$", "Amplitude (a.u.)",
         [("avg amplitude", avg_amp, "tab:green")]),
        (r"Amplitude-weighted avg frequency, $\sum_i w_i(t)\,f_i(t)$", "Frequency\n(cycles/window)",
         [("weighted avg freq", combined_freq, "tab:purple")]),
        ("Left plate vertical GRF, |Fz|", "Force (N)", [("|L Fz|", np.abs(l_grf), "darkorange")]),
        ("Right plate vertical GRF, |Fz|", "Force (N)", [("|R Fz|", np.abs(r_grf), "crimson")]),
        ("Hip flexion/extension torque — right", "Torque (N·m/kg), scaled", [
            ("torque", hip_r_torque, "steelblue"),
            ("torque × logistic(avg amplitude)", hip_r_torque_x_amp, "seagreen"),
            ("torque × logistic(weighted avg freq)", hip_r_torque_x_freq, "darkorchid"),
        ]),
        ("Hip flexion/extension torque — left", "Torque (N·m/kg), scaled", [
            ("torque", hip_l_torque, "slateblue"),
            ("torque × logistic(avg amplitude)", hip_l_torque_x_amp, "seagreen"),
            ("torque × logistic(weighted avg freq)", hip_l_torque_x_freq, "darkorchid"),
        ]),
    ]
    _plot_combined_signals_subplots(times, signals, trial_label)

    # ---- 4/5: six nonlinear [0,1)-bounding functions, one figure per signal ----
    _plot_bounded_transforms(times, avg_amp, "Average Amplitude", trial_label)
    _plot_bounded_transforms(times, combined_freq, "Amplitude-Weighted Average Frequency", trial_label)

    # ---- 6: direct (mean / weighted-sum) vs. PCA(1) channel combination ----
    # Amplitude: PCA(1) straight over the (n_windows, latent_dim) amplitude matrix,
    # compared against avg_amp (the plain per-timestep channel mean already computed above).
    pca_amp, amp_evr = pca_reduce_to_one(amp, reference=avg_amp)

    # Frequency: PCA(1) over the PER-CHANNEL amplitude-weighted frequency TERMS
    # w_i(t)*f_i(t) (NOT yet summed across channels -- that sum is exactly
    # combined_freq, amplitude_weighted_average()'s own direct method), compared
    # against combined_freq itself.
    weights = amp / (amp.sum(axis=1, keepdims=True) + WEIGHT_EPS)
    weighted_freq_channels = weights * freq   # (n_windows, latent_dim)
    pca_weighted_freq, freq_evr = pca_reduce_to_one(weighted_freq_channels, reference=combined_freq)

    print(f"[LatentSynergies] PCA(1) of amplitude channels: {amp_evr * 100:.1f}% variance explained")
    print(f"[LatentSynergies] PCA(1) of per-channel weighted-frequency terms: {freq_evr * 100:.1f}% variance explained")

    _plot_direct_vs_pca(times, avg_amp, pca_amp, amp_evr, combined_freq, pca_weighted_freq, freq_evr, trial_label)

    print(f"[LatentSynergies] Displaying figures (close each window, or Ctrl+C, to exit).")
    plt.show()


if __name__ == "__main__":
    main()
