"""
plot_pca_manifold_train.py

Two-panel diagnostic on ONE trial (SUBJECT_NAME/TRIAL_NUMBER below) of the
HELD-OUT subject, read from HELD_OUT_DATA_PATH (Data/all_data_S05_150_corrected.csv
-- a subject the checkpoint was never trained on, so this is a true
leave-one-subject-out check). Not read through the training data/index files
(Data/all_data_150_corrected.csv / train_val_index_150_corrected.csv) -- the
whole trial is used, and no train/val split applies to a held-out subject.
Inputs are normalized with the CHECKPOINT's own train-split norm_stats, exactly
as at training time:

  Panel 1: PCA phase manifold of the latest FLD-TeacherStudent checkpoint's
           TEACHER branch (biomechanics ground-truth input -- available here
           since this script reads the raw trial file directly), points
           colored red/blue by right-leg stance/swing (|R_plate_Fz| > 40 N =
           stance, else swing; GRF sampled at each window's center frame --
           see _stance_swing_colors()).
  Panel 2: HIP_ANGLE_COLUMN (hip_flexion_r by default), plain line, sampled
           at the exact same window centers as panel 1 -- so point i in
           panel 1 and the i-th sample along panel 2's line are the same
           instant, just two views of it (manifold position vs. raw angle).

plt.show() -- interactive, nothing saved to disk.

Picking the trial: set SUBJECT_NAME + TRIAL_NUMBER below -- together they form
the trial_id "<SUBJECT_NAME>_trial_<TRIAL_NUMBER>" in HELD_OUT_DATA_PATH
(S05 has trial_1, trial_1_2, trial_2 ... trial_11; _trial_df() prints the
available ids if yours isn't found).

Run:
    python3 plot_pca_manifold_train.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))

from plot_error_across_time import _find_checkpoint, _load_state_dict_flexible
from Models.FLD_teacher_student import FLDTeacherStudent
from DataLoader.teacher_student_data_loader import (
    SENSOR_DIM as TS_SENSOR_DIM, BIOMECH_COLUMNS as TS_BIOMECH_COLUMNS,
    MOMENT_COLUMNS as TS_MOMENT_COLUMNS,
)
from Library.fld_plotter import Plotter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FLD_TS_RUN = "FLDTeacherStudent_20260923_173206"   # pinned -- latest trained FLDTeacherStudent checkpoint,
                                                    # 60 completed epochs, history_horizon=151/forecast_horizon=25,
                                                    # leave-subject-out: trained on S01-S04, validated on S05
                                                    # (train_fld_teacher_student.py)

DATA_DIR     = os.path.join(os.path.dirname(__file__), "Data")
HELD_OUT_DATA_PATH = os.path.join(DATA_DIR, "all_data_S05_150_corrected.csv")   # held-out subject, never trained on
SUBJECT_NAME = "S05"   # trial_id prefix in HELD_OUT_DATA_PATH
TRIAL_NUMBER = "4"     # trial_id = f"{SUBJECT_NAME}_trial_{TRIAL_NUMBER}" (e.g. S05_trial_4, walking_1_2_m_s)

SAMPLE_RATE = 150.0             # Hz, matches every other script in this repo
LATENT_STRIDE = 2               # frames between manifold-encoder windows
MAX_MANIFOLD_WINDOWS = 20000    # cap on manifold points, after striding

HIP_ANGLE_COLUMN = "hip_flexion_r"   # panel 2's plain line; must be in TS_BIOMECH_COLUMNS

# Panel 1 stance/swing coloring -- right leg only, for now.
# Force PLATE (not insole) vertical GRF: R_plate_Fz reads ~0 N when the foot
# is unloaded and strongly negative (down to ~-1000 N) when loaded, per this
# dataset's sign convention -- so we threshold on |R_plate_Fz|. (R_insole_force
# was rejected for this: it never drops much below ~190 N even in swing,
# i.e. it looks unzeroed/biased and can't distinguish stance from swing.)
RIGHT_GRF_COLUMN = "R_plate_Fz"
GRF_STANCE_THRESHOLD_N = 40.0   # |right GRF| > this -> stance (loading); <= this -> swing
STANCE_COLOR = "red"
SWING_COLOR = "blue"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_fld_teacher_student_model(device, pinned_run=""):
    run_dir, ckpt_path = _find_checkpoint("FLDTeacherStudent", "fld_teacher_student", pinned_run)
    print(f"[PCA-Train] FLD-TeacherStudent checkpoint: {ckpt_path}")
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
    norm_stats = ckpt["norm_stats"]   # keys: mean_sensor/std_sensor/mean_biomech/std_biomech
    print(f"[PCA-Train]   epoch={ckpt['epoch']}  history_horizon={config['history_horizon']}  "
          f"latent_dim={config['latent_dim']}")
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

def _window_starts(n_rows, history_horizon, stride, max_windows):
    all_starts = np.arange(0, n_rows - history_horizon + 1, stride)
    if max_windows is not None and len(all_starts) > max_windows:
        pick = np.unique(np.round(np.linspace(0, len(all_starts) - 1, max_windows)).astype(int))
        return all_starts[pick].tolist()
    return all_starts.tolist()


def _encode_manifold(encode_fn, data, history_horizon, window_starts, device, batch_size=256):
    """Encode window start indices into the (len(window_starts), 2*latent_dim)
    phasor manifold tensor plot_pca expects -- same math as
    train_fld.py's _plot_pca_manifold.

    encode_fn(wins) -> (phase, frequency, amplitude, offset), each (B, latent_dim).
    """
    phases, amps = [], []
    with torch.no_grad():
        for i in range(0, len(window_starts), batch_size):
            batch_starts = window_starts[i:i + batch_size]
            wins = torch.stack(
                [data[s: s + history_horizon].T for s in batch_starts]
            ).to(device)
            phase, _frequency, amp, _offset = encode_fn(wins)
            phases.append(phase.cpu())
            amps.append(amp.cpu())
    phase = torch.cat(phases, dim=0)
    amp   = torch.cat(amps, dim=0)
    manifold = torch.hstack((
        amp * torch.sin(2.0 * torch.pi * phase),
        amp * torch.cos(2.0 * torch.pi * phase),
    ))
    return manifold


def _stance_swing_colors(grf, window_starts, history_horizon):
    """Per-window stance/swing label from the right force-plate vertical GRF,
    sampled at each window's CENTER frame -- FLD's phase parameterization
    treats the window as centered at reference time 0 (self.args is
    symmetric, -[H-1]/2*dt .. +[H-1]/2*dt), so the center frame is the
    "current" instant each encoded point actually represents, not the
    window's start or end. history_horizon is guaranteed odd (FLD.__init__
    asserts it), so the center index is exact.

    stance: |right GRF| > GRF_STANCE_THRESHOLD_N (loading)
    swing:  |right GRF| <= GRF_STANCE_THRESHOLD_N
    """
    half = (history_horizon - 1) // 2
    centers = np.asarray(window_starts) + half
    stance_mask = np.abs(grf[centers]) > GRF_STANCE_THRESHOLD_N
    colors = np.where(stance_mask, STANCE_COLOR, SWING_COLOR)
    return colors, stance_mask, centers


# ---------------------------------------------------------------------------
# Panel 1 -- PCA phase manifold, stance/swing colored
# ---------------------------------------------------------------------------

def plot_teacher_manifold(ax, plotter, model, norm_stats, df, history_horizon, window_starts):
    device = next(model.parameters()).device

    moments_wnorm = df[TS_MOMENT_COLUMNS].div(df["subject_weight"], axis=0)
    biomech_df = df[TS_BIOMECH_COLUMNS].copy()
    biomech_df[TS_MOMENT_COLUMNS] = moments_wnorm
    raw_biomech = biomech_df.values.astype("float32")
    norm_biomech = torch.tensor(
        (raw_biomech - norm_stats["mean_biomech"]) / norm_stats["std_biomech"], dtype=torch.float32,
    )

    print(f"[PCA-Train] Teacher: encoding {len(window_starts):,} windows (stride={LATENT_STRIDE})...")
    manifold = _encode_manifold(model.forward_teacher, norm_biomech, history_horizon, window_starts, device)

    grf = df[RIGHT_GRF_COLUMN].values.astype("float32")
    point_colors, stance_mask, _centers = _stance_swing_colors(grf, window_starts, history_horizon)
    n_stance = int(stance_mask.sum())
    n_swing = len(window_starts) - n_stance
    print(f"[PCA-Train] Teacher: {n_stance:,} stance / {n_swing:,} swing windows "
          f"(right plate Fz, threshold=|{GRF_STANCE_THRESHOLD_N:.0f}| N)")

    plotter.plot_pca(ax, [manifold], point_color=[point_colors])
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=STANCE_COLOR, markersize=7,
               label=f"Stance (|R plate Fz| > {GRF_STANCE_THRESHOLD_N:.0f} N)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SWING_COLOR, markersize=7,
               label=f"Swing (|R plate Fz| ≤ {GRF_STANCE_THRESHOLD_N:.0f} N)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    ax.set_title(f"Teacher — phase manifold (right-leg stance/swing)\n"
                 f"{len(window_starts):,} windows ({n_stance:,} stance / {n_swing:,} swing)", fontsize=10)


# ---------------------------------------------------------------------------
# Panel 2 -- plain hip-angle line, same window centers as panel 1
# ---------------------------------------------------------------------------

def plot_hip_angle(ax, df, history_horizon, window_starts):
    half = (history_horizon - 1) // 2
    centers = np.asarray(window_starts) + half
    hip_vals = df[HIP_ANGLE_COLUMN].values.astype("float64")[centers]
    times = centers / SAMPLE_RATE

    ax.plot(times, hip_vals, color="tab:blue", lw=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{HIP_ANGLE_COLUMN} (deg)")
    ax.set_title(f"{HIP_ANGLE_COLUMN}\n{len(window_starts):,} points (same windows as panel 1)", fontsize=10)
    ax.grid(alpha=0.2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PCA-Train] Using device: {device}")

    ts_model, ts_config, ts_norm_stats = _load_fld_teacher_student_model(device, FLD_TS_RUN)
    history_horizon = ts_config["history_horizon"]

    print(f"[PCA-Train] Held-out data: {HELD_OUT_DATA_PATH}  (trial {SUBJECT_NAME}_trial_{TRIAL_NUMBER})")
    usecols = list(dict.fromkeys(
        TS_BIOMECH_COLUMNS + [HIP_ANGLE_COLUMN, "subject_weight", RIGHT_GRF_COLUMN]
    ))
    df = _trial_df(SUBJECT_NAME, TRIAL_NUMBER, usecols)
    print(f"[PCA-Train] {len(df):,} raw rows ({len(df) / SAMPLE_RATE:.1f}s @{SAMPLE_RATE:.0f}Hz)")

    if len(df) < history_horizon:
        raise ValueError(f"Trial too short ({len(df)} rows) for history_horizon={history_horizon}")
    window_starts = _window_starts(len(df), history_horizon, LATENT_STRIDE, MAX_MANIFOLD_WINDOWS)

    plotter = Plotter()
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    plot_teacher_manifold(axes[0], plotter, ts_model, ts_norm_stats, df, history_horizon, window_starts)
    plot_hip_angle(axes[1], df, history_horizon, window_starts)

    fig.suptitle(f"Teacher Phase Manifold vs. {HIP_ANGLE_COLUMN} — {SUBJECT_NAME}/trial_{TRIAL_NUMBER}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    print("[PCA-Train] Displaying figure (close the window, or Ctrl+C, to exit).")
    plt.show()


if __name__ == "__main__":
    main()
