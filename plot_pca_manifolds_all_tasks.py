"""
plot_pca_manifolds_all_tasks.py

Survey version of plot_pca_manifold_train.py: instead of one hardcoded
trial, encodes EVERY trial belonging to the HELD-OUT subject in
HELD_OUT_DATA_PATH (Data/all_data_S05_150_corrected.csv -- never trained on,
so this is a true leave-one-subject-out check; inputs normalized with the
checkpoint's own train-split norm_stats) through the TEACHER branch
of the latest trained FLD-TeacherStudent checkpoint
(runs/FLDTeacherStudent_20260923_173206, pinned below -- 60 completed
epochs, history_horizon=151/forecast_horizon=25, leave-subject-out: trained on
S01-S04 (Data/all_data_150_corrected.csv), validated on S05, via
train_fld_teacher_student.py; same
checkpoint plot_pca_manifold_train.py is now also pinned to), and lays out
one PCA-manifold subplot per trial in a single grid figure, titled with
that trial's task label.

Reused, not reimplemented, from plot_pca_manifold_train.py: model loading
(_load_fld_teacher_student_model), the causal window-start builder
(_window_starts), the manifold encoder (_encode_manifold), and the same
weight-normalization of the teacher's moment columns before encoding. This
script only adds the "loop over every trial for a subject, one subplot
each" part on top -- it is NOT a stance/swing diagnostic like the original
(no GRF coloring here, one trial's whole manifold is already a lot to look
at across 12 subplots at once), and unlike the original's plot_pca() calls,
axes are explicitly turned back ON with real PC1/PC2 tick values after each
plot_pca() call -- Library/fld_plotter.py's Plotter.plot_pca() hides them
(ax.set_axis_off()) by default, appropriate for its usual side-by-side
overlay use but not for reading actual scale/spread here.

Each trial's PCA is fit INDEPENDENTLY (not one shared PCA across all 12
trials) -- same convention as plot_pca_manifold_train.py itself, and the
right choice here too: tasks span wildly different amplitude scales
(standing vs. walking vs. squats), so a single shared fit would flatten the
quiet trials into an unreadable dot while stretching the big ones. That
means axis SCALES are not directly comparable subplot-to-subplot (each
panel is auto-scaled to its own manifold) -- read shapes/spread within a
panel, not absolute PC-unit sizes across panels.

plt.show() -- interactive, nothing saved to disk.

Run:
    python3 plot_pca_manifolds_all_tasks.py
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
from Library.fld_plotter import Plotter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SUBJECT_NAME = "S02"   # held-out subject -- must match subject_name in HELD_OUT_DATA_PATH
FLD_TS_RUN = "FLDTeacherStudent_20260923_173206"   # pinned -- latest trained FLDTeacherStudent checkpoint,
                                                    # 60 completed epochs, history_horizon=151/forecast_horizon=25,
                                                    # leave-subject-out: trained on S01-S04, validated on S05
                                                    # (train_fld_teacher_student.py)

DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")
HELD_OUT_DATA_PATH = os.path.join(DATA_DIR, "all_data_150_corrected.csv")   # held-out subject, never trained on

SAMPLE_RATE = 150.0             # Hz, matches every other script in this repo
LATENT_STRIDE = 2               # frames between manifold-encoder windows
MAX_MANIFOLD_WINDOWS = 20000    # cap on manifold points per trial, after striding

POINT_COLOR = "tab:blue"


# ---------------------------------------------------------------------------
# Model loading -- verbatim from plot_pca_manifold_train.py
# ---------------------------------------------------------------------------

def _load_fld_teacher_student_model(device, pinned_run=""):
    run_dir, ckpt_path = _find_checkpoint("FLDTeacherStudent", "fld_teacher_student", pinned_run)
    print(f"[PCA-AllTasks] FLD-TeacherStudent checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = {**ckpt["config"], "loaded_epoch": ckpt["epoch"]}   # loaded_epoch: for the figure title
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
    print(f"[PCA-AllTasks]   epoch={ckpt['epoch']}  history_horizon={config['history_horizon']}  "
          f"latent_dim={config['latent_dim']}")
    return model, config, norm_stats


def _load_subject_trials(subject_name, usecols):
    """Every trial of `subject_name` in HELD_OUT_DATA_PATH, as [(trial_label, df)] in
    file order. trial_label is the trial_id minus its "<subject>_trial_" prefix
    (e.g. "S05_trial_1_2" -> "1_2")."""
    if not os.path.exists(HELD_OUT_DATA_PATH):
        raise FileNotFoundError(f"{HELD_OUT_DATA_PATH} not found.")
    df = pd.read_csv(HELD_OUT_DATA_PATH, usecols=list(dict.fromkeys(usecols + ["trial_id", "subject_name"])))
    df = df[df["subject_name"] == subject_name]
    if df.empty:
        raise ValueError(f"subject {subject_name!r} not found in {HELD_OUT_DATA_PATH}.")
    prefix = f"{subject_name}_trial_"
    return [(str(tid).removeprefix(prefix), g.reset_index(drop=True))
            for tid, g in df.groupby("trial_id", sort=False)]


# ---------------------------------------------------------------------------
# Windowing / encoding -- verbatim from plot_pca_manifold_train.py
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
    train_fld.py's _plot_pca_manifold / plot_pca_manifold_train.py.

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


# ---------------------------------------------------------------------------
# Per-trial subplot
# ---------------------------------------------------------------------------

def plot_trial_manifold(ax, plotter, model, norm_stats, df, history_horizon, trial_number, task_label, device):
    moments_wnorm = df[TS_MOMENT_COLUMNS].div(df["subject_weight"], axis=0)
    biomech_df = df[TS_BIOMECH_COLUMNS].copy()
    biomech_df[TS_MOMENT_COLUMNS] = moments_wnorm
    raw_biomech = biomech_df.values.astype("float32")
    norm_biomech = torch.tensor(
        (raw_biomech - norm_stats["mean_biomech"]) / norm_stats["std_biomech"], dtype=torch.float32,
    )

    window_starts = _window_starts(len(df), history_horizon, LATENT_STRIDE, MAX_MANIFOLD_WINDOWS)
    print(f"[PCA-AllTasks] trial_{trial_number} ({task_label}): {len(df):,} rows -> "
          f"{len(window_starts):,} windows")

    manifold = _encode_manifold(model.forward_teacher, norm_biomech, history_horizon, window_starts, device)

    plotter.plot_pca(ax, [manifold], point_color=[POINT_COLOR])
    # plot_pca() calls ax.set_axis_off() unconditionally -- turn axes back ON so PC1/PC2 tick
    # values and scale are actually visible, per the point of this script.
    ax.set_axis_on()
    ax.set_xlabel("PC 1", fontsize=8)
    ax.set_ylabel("PC 2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.15)
    ax.set_title(f"trial_{trial_number} -- {task_label}", fontsize=10, fontweight="bold")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PCA-AllTasks] Using device: {device}")

    model, config, norm_stats = _load_fld_teacher_student_model(device, FLD_TS_RUN)
    history_horizon = config["history_horizon"]

    usecols = list(dict.fromkeys(TS_BIOMECH_COLUMNS + ["subject_weight", "task"]))
    trials = _load_subject_trials(SUBJECT_NAME, usecols)
    print(f"[PCA-AllTasks] {len(trials)} trials found for held-out subject {SUBJECT_NAME} in "
          f"{HELD_OUT_DATA_PATH}: " + ", ".join(f"trial_{n}" for n, _ in trials))

    n = len(trials)
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 5.0 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    plotter = Plotter()
    for i, (trial_number, df) in enumerate(trials):
        task_label = str(df["task"].iloc[0])
        if len(df) < history_horizon:
            print(f"[PCA-AllTasks] trial_{trial_number} ({task_label}): SKIPPED, "
                  f"{len(df)} rows < history_horizon={history_horizon}")
            axes[i].set_title(f"trial_{trial_number} -- {task_label}\n(too short, skipped)",
                               fontsize=10, fontweight="bold")
            axes[i].set_axis_on()
            continue
        plot_trial_manifold(axes[i], plotter, model, norm_stats, df, history_horizon,
                             trial_number, task_label, device)

    for j in range(len(trials), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Teacher phase-manifold PCA, every trial -- HELD-OUT subject {SUBJECT_NAME}\n"
        f"(FLD-TeacherStudent, epoch {config['loaded_epoch']}, teacher branch; each subplot fit independently)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    print("[PCA-AllTasks] Displaying figure (close the window, or Ctrl+C, to exit).")
    plt.show()


if __name__ == "__main__":
    main()
