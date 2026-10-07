"""
train_fld_teacher_student.py

Training script for FLD_teacher_student.py (Models/FLD_teacher_student.py +
DataLoader/teacher_student_data_loader.py). Does NOT modify train_fld.py --
runs side by side with it, reusing its `compute_reconstruction_loss` (same
multi-step propagation loss, same gamma-decay convention) rather than
reimplementing it, per this repo's "import it rather than reimplementing"
convention (see also Models/FLD_teacher_student.py's own docstring).

Three loss terms, each independently weighted:
  a) Teacher loss      -- (teacher encoder -> shared decoder) vs. genuinely-
                          future biomech ground truth. compute_reconstruction_loss,
                          same gamma decay as train_fld.py.
  b) Alignment loss     -- student params vs. teacher params, teacher params
                          stop-gradiented (student chases teacher, never the
                          reverse). L2 on frequency/amplitude/offset; a
                          wrap-safe 1-cos(2*pi*(phase_s-phase_t)) on phase
                          (phase is a FRACTION OF A CYCLE in this codebase's
                          convention -- see Models/FLD_teacher_student.py's
                          module docstring -- a plain L2 on phase would be a
                          real bug, not a simplification).
  c) Student loss       -- (student encoder -> shared decoder) vs. the same
                          genuinely-future biomech ground truth, same
                          multi-step propagation as (a).

Teacher-only warmup (config.teacher_warmup_enabled, default True): for the
first config.teacher_warmup_epochs epochs, only the teacher loss drives
optimizer.step() (alignment/student losses are still computed and logged
every step, for visibility, just not backpropagated) -- lets the teacher's
own latent manifold stabilize before the student starts chasing a
still-moving target.

Plotting (fires once, at the final epoch only -- same convention train_fld.py
uses, to avoid slowing down every epoch with heavy plotting), saved under
<log_dir>/plots/:
  - Long-trajectory plots, one per modality (angle/velocity/moment): ground
    truth vs. TEACHER prediction vs. STUDENT prediction, tiled across a
    validation trial -- see _plot_long_trajectory_dual (adapted from
    train_fld.py's _plot_long_trajectory for this model's two-branch
    forward_teacher()/forward_student() calling convention, which the
    original function can't call directly).
  - Phase manifold: teacher and student latent params, PCA'd and overlaid on
    the SAME axes via Library/fld_plotter.py's Plotter.plot_pca (reused
    as-is -- it already accepts a list of manifolds to overlay) -- this is
    the actual alignment story: do the two latent spaces coincide?
  - Latent parameter histograms: frequency/amplitude/offset per latent
    channel, teacher vs. student overlaid.

Run:
    python3 combine_data.py            # once, or after adding trials/subjects
    python3 train_fld_teacher_student.py
    python3 train_fld_teacher_student.py --epochs 5 --batch-size 64   # CLI override example
    # resume a prior run for 60 MORE epochs past whatever epoch that checkpoint left off at
    # (--epochs is read as "additional epochs" whenever --resume-from is set -- see
    # TeacherStudentConfig.resume_from's own docstring):
    python3 train_fld_teacher_student.py --resume-from runs/FLDTeacherStudent_.../fld_teacher_student_epoch_60.pt --epochs 60
"""

import argparse
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# Try an interactive backend if a display is available, fall back to
# headless Agg otherwise -- must happen before `import matplotlib.pyplot`.
# Same pattern as plot_joint_velocities.py.
import matplotlib
_HAS_DISPLAY = bool(os.environ.get("DISPLAY")) or os.name == "nt"
if _HAS_DISPLAY:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))

from Models.FLD_teacher_student import FLDTeacherStudent, LatentParams
from DataLoader.teacher_student_data_loader import (
    build_datasets_held_out, SENSOR_DIM, BIOMECH_DIM, BIOMECH_COLUMNS,
)
from train_fld import compute_reconstruction_loss, DistributionBuffer, _show, DATA_DIR   # reused, not reimplemented
from Library.fld_plotter import Plotter

# 150Hz "corrected" data files (built by combine_data.py): the TRAIN file (S01-S04)
# and the held-out VALIDATION file (S05 only -- never trained on).
ALL_DATA_PATH = os.path.join(DATA_DIR, "all_data_150_corrected.csv")
VAL_DATA_PATH = os.path.join(DATA_DIR, "all_data_S05_150_corrected.csv")


# ---------------------------------------------------------------------------
# Config -- dataclass with defaults, overridable by CLI (see parse_args()).
# No magic numbers buried in the model or training loop: every tunable lives
# here.
# ---------------------------------------------------------------------------

@dataclass
class TeacherStudentConfig:
    # Data -- LEAVE-SUBJECT-OUT. No train_val_index CSV is read: TRAIN uses
    # every window of every trial in data_path (Data/all_data_150_corrected.csv,
    # S01-S04), VALIDATION uses every window of val_data_path
    # (Data/all_data_S05_150_corrected.csv, subject S05 -- never trained on).
    # Normalisation stats are fit on the train file only. The two files must
    # hold disjoint subjects (build_datasets_held_out() raises otherwise).
    # `subjects` only restricts which subjects of the TRAIN file participate
    # ([] = all of them, the default); it is never applied to the val file.
    data_path: str = ALL_DATA_PATH
    val_data_path: str = VAL_DATA_PATH
    subjects: List[str] = field(default_factory=list)   # [] = every subject in data_path
    # Held-out trial used for the end-of-training trajectory/manifold plots
    # (S05_trial_4 = walking 1.2 m/s). Falls back to the first val trial if
    # this id isn't in val_data_path.
    plot_trial_id: str = "S05_trial_4"

    # Window / sampling
    history_horizon: int = 151     # must be odd -- see FLD's Conv1d same-padding requirement
    forecast_horizon: int = 25
    sample_rate: float = 150.0

    # Model
    latent_dim: int = 8
    teacher_encoder_shape: List[int] = field(default_factory=lambda: [64, 32])
    student_encoder_shape: List[int] = field(default_factory=lambda: [64, 32])
    decoder_shape: List[int] = field(default_factory=lambda: [32, 64])

    # Loss weights
    weight_teacher: float = 1.0
    weight_alignment: float = 1.0
    weight_student: float = 1.0
    gamma: float = 0.95              # propagation-loss horizon decay, shared by teacher AND student losses
    align_weight_phase: float = 1.0
    align_weight_freq: float = 1.0
    align_weight_amp: float = 1.0
    align_weight_offset: float = 1.0

    # Teacher-only warmup
    teacher_warmup_enabled: bool = True
    teacher_warmup_epochs: int = 8

    # Optimisation
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 5e-4
    epochs: int = 60
    num_workers: int = 4
    save_every: int = 10
    use_amp: bool = True
    seed: int = 0

    # Output
    log_dir: str = "runs"

    # Resume -- "" (default) = train from scratch, the original/unaffected behaviour
    # for every existing caller. Set to a checkpoint .pt path (as saved by _save()
    # below -- model_state_dict/optimizer_state_dict/epoch/config/norm_stats) to
    # continue training from it: model + optimizer state are restored, the epoch
    # counter picks up where that checkpoint left off, and `epochs` above is then
    # read as "how many MORE epochs to run" rather than "total epochs from scratch"
    # (see train()'s own docstring/comments for exactly where that branches).
    # Only meaningful when resuming with the SAME architecture config (encoder/
    # decoder shapes, history_horizon, sample_rate, ...) the checkpoint was
    # trained with -- state_dict shapes must match, same as any checkpoint load
    # elsewhere in this repo.
    resume_from: str = ""


def parse_args(argv: List[str] = None, defaults: TeacherStudentConfig = None) -> TeacherStudentConfig:
    """Build a TeacherStudentConfig from CLI args. `defaults` (optional) lets
    a variant script supply its own base config -- different sample_rate/
    history_horizon/data paths, say -- and get this SAME flag set/help text
    for free instead of re-declaring every `p.add_argument(...)` call (see
    train_fld_teacher_student_100hz.py's own parse_args for exactly that).
    Defaults to TeacherStudentConfig() (this script's own 150Hz values) when
    not given -- the original behaviour, unaffected for this script itself
    and every caller that doesn't pass defaults."""
    d = defaults if defaults is not None else TeacherStudentConfig()
    p = argparse.ArgumentParser(description="Train FLD teacher-student cross-modal gait estimator.")

    p.add_argument("--data-path", type=str, default=d.data_path,
                    help="TRAIN data file (every window of every trial is used).")
    p.add_argument("--val-data-path", type=str, default=d.val_data_path,
                    help="Held-out VALIDATION data file (a subject not present in --data-path).")
    p.add_argument("--subjects", type=str, nargs="+", default=d.subjects,
                    help="Restrict the TRAIN file to these subject_name values ([] = every subject). "
                         "Never applied to the validation file.")
    p.add_argument("--plot-trial-id", type=str, default=d.plot_trial_id,
                    help="Held-out trial_id for the end-of-training plots.")

    p.add_argument("--history-horizon", type=int, default=d.history_horizon)
    p.add_argument("--forecast-horizon", type=int, default=d.forecast_horizon)
    p.add_argument("--sample-rate", type=float, default=d.sample_rate)

    p.add_argument("--latent-dim", type=int, default=d.latent_dim)
    p.add_argument("--teacher-encoder-shape", type=int, nargs="+", default=d.teacher_encoder_shape)
    p.add_argument("--student-encoder-shape", type=int, nargs="+", default=d.student_encoder_shape)
    p.add_argument("--decoder-shape", type=int, nargs="+", default=d.decoder_shape)

    p.add_argument("--weight-teacher", type=float, default=d.weight_teacher)
    p.add_argument("--weight-alignment", type=float, default=d.weight_alignment)
    p.add_argument("--weight-student", type=float, default=d.weight_student)
    p.add_argument("--gamma", type=float, default=d.gamma)
    p.add_argument("--align-weight-phase", type=float, default=d.align_weight_phase)
    p.add_argument("--align-weight-freq", type=float, default=d.align_weight_freq)
    p.add_argument("--align-weight-amp", type=float, default=d.align_weight_amp)
    p.add_argument("--align-weight-offset", type=float, default=d.align_weight_offset)

    p.add_argument("--teacher-warmup-enabled", dest="teacher_warmup_enabled",
                    action="store_true", default=d.teacher_warmup_enabled)
    p.add_argument("--no-teacher-warmup", dest="teacher_warmup_enabled", action="store_false")
    p.add_argument("--teacher-warmup-epochs", type=int, default=d.teacher_warmup_epochs)

    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--save-every", type=int, default=d.save_every)
    p.add_argument("--use-amp", dest="use_amp", action="store_true", default=d.use_amp)
    p.add_argument("--no-amp", dest="use_amp", action="store_false")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--log-dir", type=str, default=d.log_dir)
    p.add_argument("--resume-from", type=str, default=d.resume_from,
                    help="Path to a checkpoint .pt to resume from (\"\" = train from scratch). "
                         "When set, --epochs means additional epochs to run past that checkpoint's own epoch.")

    args = p.parse_args(argv)
    return TeacherStudentConfig(**vars(args))


# Output modality groups -- matches train_fld.py's MODALITY_GROUPS exactly
# (BIOMECH_COLUMNS is imported from the same 30-column convention: 10 angle
# + 10 velocity + 10 moment).
MODALITY_GROUPS = {
    "angle":    {"indices": list(range(0, 10)),  "unit": "°",      "title": "Joint Angles"},
    "velocity": {"indices": list(range(10, 20)), "unit": "°/s",    "title": "Joint Angular Velocities"},
    "moment":   {"indices": list(range(20, 30)), "unit": "N·m/kg", "title": "Joint Moments"},
}


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def alignment_loss(
    student_params: LatentParams,
    teacher_params: LatentParams,
    w_phase: float,
    w_freq: float,
    w_amp: float,
    w_offset: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Student params vs. stop-gradiented teacher params. L2 on
    frequency/amplitude/offset; wrap-safe 1-cos(2*pi*delta) on phase.

    Phase here is a FRACTION OF A CYCLE (FLD's own atan2(...)/(2*pi)
    convention -- see Models/FLD_teacher_student.py's module docstring), not
    radians. Scaling the difference by 2*pi before cos() is required for
    correct periodicity: without it, e.g. a phase difference of exactly 1.0
    (a full cycle -- truly zero error) would NOT register as zero. A plain
    L2 on the raw phase values is a bug for the same underlying reason (it
    penalises 0.99 vs. 0.01 -- adjacent in phase, off by only 2% of a cycle
    -- as if they were nearly a full cycle apart).

    Args:
        student_params: (phase, frequency, amplitude, offset), each (B, latent_dim).
        teacher_params: same shape/order. Detached internally -- gradients
            never flow from this loss back into the teacher encoder.
        w_phase, w_freq, w_amp, w_offset: per-component weights.

    Returns:
        (total_loss, components) -- components is a dict of the four
        unweighted scalar terms, for separate logging.
    """
    phase_s, freq_s, amp_s, off_s = student_params
    phase_t, freq_t, amp_t, off_t = [p.detach() for p in teacher_params]   # stop-gradient into teacher

    phase_diff_rad = 2.0 * math.pi * (phase_s - phase_t)
    loss_phase = (1.0 - torch.cos(phase_diff_rad)).mean()
    loss_freq = F.mse_loss(freq_s, freq_t)
    loss_amp = F.mse_loss(amp_s, amp_t)
    loss_offset = F.mse_loss(off_s, off_t)

    total = w_phase * loss_phase + w_freq * loss_freq + w_amp * loss_amp + w_offset * loss_offset
    components = {"phase": loss_phase, "freq": loss_freq, "amp": loss_amp, "offset": loss_offset}
    return total, components


def per_joint_mae(pred: torch.Tensor, target: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """MAE in original physical units per biomech channel, averaged over K and B.

    Args:
        pred, target: (K, B, biomech_dim) normalised.
        std: (biomech_dim,) train-split std (from norm_stats['std_biomech']).

    Returns:
        (biomech_dim,) tensor.
    """
    abs_err = (pred - target).abs()
    mae_norm = abs_err.mean(dim=(0, 1))
    return mae_norm * std.clamp(min=1e-6).to(pred.device)


def phase_alignment_error_degrees(student_phase: torch.Tensor, teacher_phase: torch.Tensor) -> float:
    """Mean absolute circular error between student and teacher phase
    (cycle-fraction convention), converted to degrees for readability.

    Args:
        student_phase, teacher_phase: (B, latent_dim).

    Returns:
        scalar float, degrees.
    """
    diff = 2.0 * math.pi * (student_phase - teacher_phase)
    circular = torch.atan2(torch.sin(diff), torch.cos(diff))
    return circular.abs().mean().item() * (180.0 / math.pi)


# ---------------------------------------------------------------------------
# Plotting -- fires once, at the final epoch only (see train()). All four
# functions below (loss curve + the three dual-branch plots) share that
# convention. The three dual-branch functions take a single TRIAL's
# TeacherStudentFLDDataset (e.g. val_dataset.datasets[0] -- ConcatDataset
# exposes its underlying per-trial datasets via .datasets), not the
# ConcatDataset itself, since they need direct access to that trial's own
# contiguous data_sensor/data_biomech tensors + norm_stats to build a
# real-time-axis plot.
# ---------------------------------------------------------------------------

def _plot_loss_curve_triple(
    fig, axes, train_history: Dict[str, List[float]], val_history: Dict[str, List[float]],
    save_path: str = None,
) -> None:
    """All three loss terms (teacher/alignment/student), train (solid) vs.
    val (dashed), one color per term -- same colors used throughout this
    script's other plots (teacher=green, student=red; alignment gets its own
    color since it has no single-branch counterpart). Adapted from
    train_fld.py's _plot_loss_curve (single combined train/val loss there;
    this needs three terms, hence not a straight reuse).

    Args:
        train_history, val_history: dict with keys 'teacher'/'alignment'/'student'
            -> list of per-epoch mean loss values (train()'s running/val_running,
            appended once per epoch).
    """
    colors = {"teacher": "seagreen", "alignment": "goldenrod", "student": "tomato"}
    ax = axes if not hasattr(axes, "__len__") else axes[0]
    ax.cla()
    n_epochs = len(train_history["teacher"])
    epochs = range(1, n_epochs + 1)
    for term, color in colors.items():
        ax.plot(epochs, train_history[term], color=color, lw=1.6, ls="-", label=f"{term} (train)")
        ax.plot(epochs, val_history[term], color=color, lw=1.6, ls="--", label=f"{term} (val)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (normalised space)")
    ax.set_title("FLD Teacher-Student Training Loss — solid = train, dashed = val")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    _show(fig, save_path)


def _plot_long_trajectory_dual(
    model: FLDTeacherStudent, dataset, device: torch.device,
    history_horizon: int, forecast_horizon: int,
    fig, axes, feature_slice: slice, group_title: str, unit: str,
    n_blocks: int = 100, block_step: int = None, sample_rate: float = 150.0,
    epoch: int = None, save_path: str = None,
) -> None:
    """Ground truth (blue) vs. TEACHER prediction (green dashed) vs. STUDENT
    prediction (red dashed), tiled across `dataset` with a fresh
    history_horizon-frame context every `block_step` frames (default =
    forecast_horizon, non-overlapping). Adapted from train_fld.py's
    _plot_long_trajectory -- that function calls a single-branch model
    directly (`model(x, k=k)`); this one calls forward_teacher()/
    forward_student()/forecast() instead, since FLDTeacherStudent has no
    single unified forward().

    Args:
        dataset: ONE trial's TeacherStudentFLDDataset (not a ConcatDataset).
        feature_slice: which BIOMECH_COLUMNS indices this figure covers
            (e.g. slice(0,10) for angles) -- see MODALITY_GROUPS.
        axes: one matplotlib Axes per feature in feature_slice.
    """
    if block_step is None:
        block_step = forecast_horizon
    H, K = history_horizon, forecast_horizon
    N = len(dataset.data_sensor)

    max_blocks = (N - H - K) // block_step + 1
    n_blocks = min(n_blocks, max(max_blocks, 0))
    if n_blocks <= 0:
        print("[FLD-TS] _plot_long_trajectory_dual: not enough rows for even one block, skipping.")
        return

    mean_b = dataset.norm_stats["mean_biomech"]
    std_b = dataset.norm_stats["std_biomech"]

    gt_end = (n_blocks - 1) * block_step + H + K
    gt_full = dataset.data_biomech[H:gt_end].detach().cpu().numpy() * std_b + mean_b   # (T, biomech_dim)
    t_gt = np.arange(H, gt_end) / sample_rate

    teacher_blocks, student_blocks = [], []
    model.eval()
    with torch.no_grad():
        for b in range(n_blocks):
            s = b * block_step
            sensor_win = dataset.data_sensor[s:s + H].T.unsqueeze(0).to(device)
            biomech_win = dataset.data_biomech[s:s + H].T.unsqueeze(0).to(device)

            teacher_params = model.forward_teacher(biomech_win)
            teacher_future = model.forecast(teacher_params, K)[:, 0, :].cpu().numpy() * std_b + mean_b

            student_params = model.forward_student(sensor_win)
            student_future = model.forecast(student_params, K)[:, 0, :].cpu().numpy() * std_b + mean_b

            t_block = np.arange(s + H, s + H + K) / sample_rate
            teacher_blocks.append((t_block, teacher_future))
            student_blocks.append((t_block, student_future))

    gt_plot = gt_full[:, feature_slice]
    col_names = BIOMECH_COLUMNS[feature_slice]

    for fi, ax in enumerate(axes):
        ax.cla()
        ax.plot(t_gt, gt_plot[:, fi], color="steelblue", lw=1.1, alpha=0.9, label="GT", zorder=10)
        for bi, (t_block, pred) in enumerate(teacher_blocks):
            ax.plot(t_block, pred[:, feature_slice][:, fi], color="seagreen", lw=1.3, alpha=0.75, ls="--",
                    label=("Teacher" if bi == 0 else None))
        for bi, (t_block, pred) in enumerate(student_blocks):
            ax.plot(t_block, pred[:, feature_slice][:, fi], color="tomato", lw=1.3, alpha=0.75, ls="--",
                    label=("Student" if bi == 0 else None))
        ax.set_ylabel(f"{col_names[fi][:20]}\n({unit})", fontsize=7, rotation=0, labelpad=95, va="center")
        ax.grid(alpha=0.15)
        ax.tick_params(labelsize=6)
        if fi == 0:
            ax.legend(fontsize=7, loc="upper right", ncol=3)

    axes[-1].set_xlabel("Time (s)")
    epoch_str = f"[Epoch {epoch + 1}]  " if epoch is not None else ""
    fig.suptitle(
        f"{epoch_str}Teacher vs Student — {group_title} ({dataset.trial_id}) — "
        f"GT (blue) vs Teacher (green dashed) vs Student (red dashed)\n"
        f"{n_blocks} blocks, {H} frames context -> {K} genuinely-future frames per block",
        fontsize=9, fontweight="bold",
    )
    _show(fig, save_path)


def _plot_phase_manifold_dual(
    model: FLDTeacherStudent, dataset, device: torch.device, history_horizon: int,
    ax, plotter: Plotter, title: str, n_windows: int = 400, start_frame: int = 0,
    save_path: str = None, fig=None, batch_size: int = 256,
) -> None:
    """Encode n_windows consecutive context windows from BOTH branches and
    overlay their Fourier phase manifolds on ONE set of PCA axes --
    Library/fld_plotter.py's Plotter.plot_pca already accepts a LIST of
    manifolds to fit one shared PCA over and plot as separate series, so
    this reuses it as-is (imported, not reimplemented) rather than writing a
    new PCA routine.

    Args:
        dataset: ONE trial's TeacherStudentFLDDataset.
    """
    H = history_horizon
    N = len(dataset.data_sensor)
    n_windows = max(1, min(n_windows, N - start_frame - H))
    starts = list(range(start_frame, start_frame + n_windows))

    teacher_phases, teacher_amps = [], []
    student_phases, student_amps = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(starts), batch_size):
            batch_starts = starts[i:i + batch_size]
            sensor_wins = torch.stack([dataset.data_sensor[s:s + H].T for s in batch_starts]).to(device)
            biomech_wins = torch.stack([dataset.data_biomech[s:s + H].T for s in batch_starts]).to(device)

            t_phase, _t_freq, t_amp, _t_off = model.forward_teacher(biomech_wins)
            s_phase, _s_freq, s_amp, _s_off = model.forward_student(sensor_wins)

            teacher_phases.append(t_phase.cpu()); teacher_amps.append(t_amp.cpu())
            student_phases.append(s_phase.cpu()); student_amps.append(s_amp.cpu())

    teacher_phase = torch.cat(teacher_phases, dim=0)
    teacher_amp = torch.cat(teacher_amps, dim=0)
    student_phase = torch.cat(student_phases, dim=0)
    student_amp = torch.cat(student_amps, dim=0)

    teacher_manifold = torch.hstack((
        teacher_amp * torch.sin(2.0 * math.pi * teacher_phase),
        teacher_amp * torch.cos(2.0 * math.pi * teacher_phase),
    ))
    student_manifold = torch.hstack((
        student_amp * torch.sin(2.0 * math.pi * student_phase),
        student_amp * torch.cos(2.0 * math.pi * student_phase),
    ))

    plotter.plot_pca(ax, [teacher_manifold, student_manifold], title=title,
                      point_color=["seagreen", "tomato"])
    # plot_pca's default legend labels each manifold "0"/"1" via `label=i` on
    # its OWN scatter call -- overriding with ax.legend(["Teacher","Student"])
    # (no explicit handles) would instead relabel whichever two artists were
    # ADDED TO THE AXES FIRST in creation order (plot_pca draws a connecting
    # line before its scatter for EACH manifold, so the first two artists are
    # both the teacher's own line+scatter) -- silently mislabeling the
    # student's points as "Teacher"'s and vice versa. Build fresh proxy
    # handles instead, tied to the same colors passed to point_color above,
    # so the legend is correct regardless of plot_pca's internal artist order.
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="seagreen", markersize=8, label="Teacher"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="tomato", markersize=8, label="Student"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    _show(fig if fig is not None else ax.figure, save_path)


def _plot_latent_histograms_dual(
    teacher_bufs: Dict[str, DistributionBuffer], student_bufs: Dict[str, DistributionBuffer],
    fig, axes, latent_dim: int, save_path: str = None,
) -> None:
    """Frequency/amplitude/offset histograms per latent channel, teacher
    (green) vs. student (red) overlaid on the same axes -- adapted from
    train_fld.py's _plot_parameter_histograms (single-branch there; this
    needs two distributions per subplot, hence not a straight reuse).

    Args:
        teacher_bufs, student_bufs: dict with keys 'freq'/'amp'/'offset' ->
            DistributionBuffer (imported from train_fld.py), accumulated
            during validation -- see train().
    """
    col_info = [("Frequency (cycles/window)", "freq"), ("Amplitude", "amp"), ("Offset", "offset")]
    for j in range(latent_dim):
        for col, (title, key) in enumerate(col_info):
            ax = axes[j, col] if latent_dim > 1 else axes[col]
            ax.cla()
            t_vals = teacher_bufs[key].get().numpy()[:, j]
            s_vals = student_bufs[key].get().numpy()[:, j]
            ax.hist(t_vals, bins=40, color="seagreen", alpha=0.5, label="Teacher", density=True)
            ax.hist(s_vals, bins=40, color="tomato", alpha=0.5, label="Student", density=True)
            ax.set_ylabel(f"ch {j}", fontsize=8, rotation=0, labelpad=28, va="center")
            ax.grid(alpha=0.2)
            if j == 0:
                ax.set_title(title, fontsize=9, fontweight="bold")
                ax.legend(fontsize=7)
    fig.suptitle("Latent Parameter Distributions — Teacher (green) vs Student (red)",
                 fontsize=9, fontweight="bold")
    _show(fig, save_path)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def _save(model: FLDTeacherStudent, optimizer: optim.Optimizer, epoch: int,
          config: TeacherStudentConfig, norm_stats: Dict[str, np.ndarray], log_dir: str) -> str:
    """Model, optimizer, config, and normalisation stats, bundled together --
    same convention as train_fld.py's _save()."""
    path = os.path.join(log_dir, f"fld_teacher_student_epoch_{epoch}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "config": asdict(config),
        "norm_stats": norm_stats,
    }, path)
    print(f"[FLD-TS] Checkpoint saved -> {path}")
    return path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: TeacherStudentConfig) -> str:
    """Full training loop. Returns the log directory the checkpoints/config
    were written to."""
    torch.manual_seed(config.seed)
    assert config.history_horizon % 2 == 1, (
        f"history_horizon must be odd (got {config.history_horizon})."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FLD-TS] Using device: {device}")
    if device.type == "cuda":
        print(f"[FLD-TS] GPU: {torch.cuda.get_device_name(device)}")

    data_device = device if device.type == "cuda" else None
    num_workers = 0 if data_device is not None else config.num_workers

    # bf16 autocast on CUDA -- same convention as train_fld.py. This matters
    # a lot here specifically: decode() materializes a (forecast_horizon+1,
    # B, biomech_dim, H) tensor through the full decoder TWICE per batch
    # (teacher branch + student branch), so halving activation memory via
    # autocast is the difference between batch_size=512 fitting in 16GB VRAM
    # and OOMing well below that.
    use_amp = config.use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16

    log_dir = os.path.join(config.log_dir, f"FLDTeacherStudent_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)

    train_ds, val_ds, norm_stats, _train_trials, val_trials = build_datasets_held_out(
        config.history_horizon, config.forecast_horizon,
        train_data_path=config.data_path, val_data_path=config.val_data_path,
        train_subjects=config.subjects, device=data_device,
    )

    loader_kwargs = dict(num_workers=num_workers, pin_memory=(data_device is None))
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                           drop_last=True, **loader_kwargs)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                         drop_last=False, **loader_kwargs)

    model = FLDTeacherStudent(
        sensor_dim=SENSOR_DIM, biomech_dim=BIOMECH_DIM,
        history_horizon=config.history_horizon, latent_dim=config.latent_dim,
        device=device, dt=1.0 / config.sample_rate,
        teacher_encoder_shape=config.teacher_encoder_shape,
        student_encoder_shape=config.student_encoder_shape,
        decoder_shape=config.decoder_shape,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    out_std = torch.tensor(norm_stats["std_biomech"], dtype=torch.float32)

    # ---- Resume (optional) -- restores model + optimizer state from a prior
    # checkpoint and picks up the epoch counter from there. config.epochs is
    # then read as "how many MORE epochs to run" (not "total epochs from
    # scratch") -- see TeacherStudentConfig.resume_from's own docstring.
    # norm_stats is NOT reloaded from the checkpoint: build_datasets_held_out()
    # above already recomputed it deterministically from the same train file/
    # subjects, so it's identical to what the checkpoint saved as long
    # as the data hasn't changed -- reusing the freshly-computed one keeps
    # this branch simple and avoids a second, easy-to-miss source of truth.
    start_epoch = 0
    if config.resume_from:
        print(f"[FLD-TS] Resuming from checkpoint: {config.resume_from}")
        resume_ckpt = torch.load(config.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        start_epoch = resume_ckpt["epoch"]
        print(f"[FLD-TS]   resumed at epoch={start_epoch} -- running {config.epochs} more "
              f"(target epoch={start_epoch + config.epochs})")
    total_epochs = start_epoch + config.epochs

    # ---- Plotting setup -- figures created once and reused via .cla() inside
    # the plot functions (same convention as train_fld.py). Fires once, at
    # the final epoch only -- see the epoch loop below. ----
    plots_dir = os.path.join(log_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plotter = Plotter()

    fig_long = {}
    ax_long = {}
    for modality, group in MODALITY_GROUPS.items():
        n_rows = len(group["indices"])
        fig_long[modality], ax_long[modality] = plt.subplots(n_rows, 1, figsize=(30, 2.0 * n_rows), sharex=True)

    fig_pca, ax_pca = plt.subplots(figsize=(8, 8))
    fig_hist, ax_hist = plt.subplots(config.latent_dim, 3, figsize=(14, 3.0 * config.latent_dim))
    fig_loss, ax_loss = plt.subplots(figsize=(9, 5))

    buf_teacher = {k: DistributionBuffer(config.latent_dim) for k in ("freq", "amp", "offset")}
    buf_student = {k: DistributionBuffer(config.latent_dim) for k in ("freq", "amp", "offset")}

    train_loss_history = {"teacher": [], "alignment": [], "student": []}
    val_loss_history = {"teacher": [], "alignment": [], "student": []}

    train_start = datetime.now()
    print(f"[FLD-TS] Training started at {train_start.strftime('%Y-%m-%d %H:%M:%S')}.")
    for epoch in range(start_epoch, total_epochs):
        epoch_start = datetime.now()
        model.train()
        warmup_active = config.teacher_warmup_enabled and epoch < config.teacher_warmup_epochs

        running = {"teacher": 0.0, "alignment": 0.0, "student": 0.0, "total": 0.0}
        n_batches = 0

        for batch in train_dl:
            sensor_win = batch["sensor_window"].to(device, non_blocking=True)       # (B, sensor_dim, H)
            biomech_win = batch["biomech_window"].to(device, non_blocking=True)     # (B, biomech_dim, H)
            future_biomech = batch["future_biomech"].to(device, non_blocking=True)  # (B, K, biomech_dim)
            target_all = future_biomech.permute(1, 0, 2)                             # (K, B, biomech_dim)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                teacher_params = model.forward_teacher(biomech_win)
                teacher_future = model.forecast(teacher_params, config.forecast_horizon)
                loss_teacher = compute_reconstruction_loss(teacher_future, target_all, gamma=config.gamma)

                student_params = model.forward_student(sensor_win)
                loss_align, _align_components = alignment_loss(
                    student_params, teacher_params,
                    config.align_weight_phase, config.align_weight_freq,
                    config.align_weight_amp, config.align_weight_offset,
                )
                student_future = model.forecast(student_params, config.forecast_horizon)
                loss_student = compute_reconstruction_loss(student_future, target_all, gamma=config.gamma)

            if warmup_active:
                total_loss = config.weight_teacher * loss_teacher
            else:
                total_loss = (
                    config.weight_teacher * loss_teacher
                    + config.weight_alignment * loss_align
                    + config.weight_student * loss_student
                )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            running["teacher"] += loss_teacher.item()
            running["alignment"] += loss_align.item()
            running["student"] += loss_student.item()
            running["total"] += total_loss.item()
            n_batches += 1

        for k in running:
            running[k] /= max(n_batches, 1)

        # ---- Validation ----
        model.eval()
        val_running = {"teacher": 0.0, "alignment": 0.0, "student": 0.0}
        mae_accum = torch.zeros(BIOMECH_DIM)
        phase_err_accum = 0.0
        n_val_batches = 0
        for buf in (*buf_teacher.values(), *buf_student.values()):
            buf.clear()
        with torch.no_grad():
            for batch in val_dl:
                sensor_win = batch["sensor_window"].to(device, non_blocking=True)
                biomech_win = batch["biomech_window"].to(device, non_blocking=True)
                future_biomech = batch["future_biomech"].to(device, non_blocking=True)
                target_all = future_biomech.permute(1, 0, 2)

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    teacher_params = model.forward_teacher(biomech_win)
                    teacher_future = model.forecast(teacher_params, config.forecast_horizon)
                    v_loss_teacher = compute_reconstruction_loss(teacher_future, target_all, gamma=config.gamma)

                    student_params = model.forward_student(sensor_win)
                    v_loss_align, _ = alignment_loss(
                        student_params, teacher_params,
                        config.align_weight_phase, config.align_weight_freq,
                        config.align_weight_amp, config.align_weight_offset,
                    )
                    student_future = model.forecast(student_params, config.forecast_horizon)
                    v_loss_student = compute_reconstruction_loss(student_future, target_all, gamma=config.gamma)

                val_running["teacher"] += v_loss_teacher.item()
                val_running["alignment"] += v_loss_align.item()
                val_running["student"] += v_loss_student.item()

                mae_accum += per_joint_mae(student_future, target_all, out_std).cpu()
                phase_err_accum += phase_alignment_error_degrees(student_params[0], teacher_params[0])
                n_val_batches += 1

                buf_teacher["freq"].insert(teacher_params[1]); buf_teacher["amp"].insert(teacher_params[2])
                buf_teacher["offset"].insert(teacher_params[3])
                buf_student["freq"].insert(student_params[1]); buf_student["amp"].insert(student_params[2])
                buf_student["offset"].insert(student_params[3])

        n_val_batches = max(n_val_batches, 1)
        for k in val_running:
            val_running[k] /= n_val_batches
        mae_accum /= n_val_batches
        phase_err_deg = phase_err_accum / n_val_batches

        for term in ("teacher", "alignment", "student"):
            train_loss_history[term].append(running[term])
            val_loss_history[term].append(val_running[term])

        warmup_str = " [teacher warmup]" if warmup_active else ""
        print(f"[FLD-TS] Epoch [{epoch + 1:>4}/{total_epochs}]{warmup_str}  "
              f"train: total={running['total']:.5f} teacher={running['teacher']:.5f} "
              f"alignment={running['alignment']:.5f} student={running['student']:.5f}  |  "
              f"val: teacher={val_running['teacher']:.5f} alignment={val_running['alignment']:.5f} "
              f"student={val_running['student']:.5f}  |  phase_err={phase_err_deg:.2f} deg")

        mae_str = "  ".join(f"{name}={mae_accum[i]:.3f}" for i, name in enumerate(BIOMECH_COLUMNS))
        print(f"          per-joint val MAE (physical units): {mae_str}")

        # All plots fire once, at the final epoch only (same convention
        # train_fld.py uses -- avoids slowing down every epoch with heavy
        # plotting). Reference trial: one HELD-OUT trial (config.plot_trial_id,
        # else the first val trial) -- val_trials holds each trial's full
        # data_sensor/data_biomech, which the ConcatDataset train_ds/val_ds
        # don't expose directly.
        if epoch == total_epochs - 1:
            _plot_loss_curve_triple(
                fig_loss, ax_loss, train_loss_history, val_loss_history,
                save_path=os.path.join(plots_dir, "loss_curve.png"),
            )
            with torch.no_grad():
                model.eval()
                ref_dataset = val_trials.get(config.plot_trial_id) or next(iter(val_trials.values()))

                for modality, group in MODALITY_GROUPS.items():
                    sl = slice(group["indices"][0], group["indices"][-1] + 1)
                    _plot_long_trajectory_dual(
                        model, ref_dataset, device, config.history_horizon, config.forecast_horizon,
                        fig_long[modality], ax_long[modality], sl, group["title"], group["unit"],
                        n_blocks=100, sample_rate=config.sample_rate, epoch=epoch,
                        save_path=os.path.join(plots_dir, f"{modality}_trajectory.png"),
                    )

                _plot_phase_manifold_dual(
                    model, ref_dataset, device, config.history_horizon, ax_pca, plotter,
                    title=f"Phase Manifold — Teacher vs Student — {ref_dataset.trial_id}",
                    n_windows=400, save_path=os.path.join(plots_dir, "phase_manifold.png"), fig=fig_pca,
                )

                _plot_latent_histograms_dual(
                    buf_teacher, buf_student, fig_hist, ax_hist, config.latent_dim,
                    save_path=os.path.join(plots_dir, "latent_histograms.png"),
                )
                model.train()

        if epoch % config.save_every == 0:
            _save(model, optimizer, epoch, config, norm_stats, log_dir)

        epoch_end = datetime.now()
        print(f"[FLD-TS] Epoch {epoch + 1} finished at {epoch_end.strftime('%Y-%m-%d %H:%M:%S')}  "
              f"(epoch took {(epoch_end - epoch_start).total_seconds():.1f}s, "
              f"elapsed {(epoch_end - train_start).total_seconds() / 60:.1f} min total)")

    _save(model, optimizer, total_epochs, config, norm_stats, log_dir)
    print(f"[FLD-TS] Training finished. Checkpoints + plots in: {log_dir}")
    plt.ioff()
    plt.show()  # keep all figures open after training completes
    return log_dir


if __name__ == "__main__":
    train(parse_args())
