"""
train_pae_teacher_student.py

Training script for Models/PAE_teacher_student.py (PAETeacherStudent) +
DataLoader/pae_teacher_student_data_loader.py. A PAE (Periodic Autoencoder)
counterpart to train_fld_teacher_student.py's FLD teacher-student setup, NOT
a modification of it -- runs side by side.

  - TEACHER encoder input: joint angles + joint angular velocities + joint
    moments (30ch, BIOMECH_COLUMNS == train_fld.OUTPUT_COLUMNS). Reconstructs
    the SAME window it was given (PAE.py is an ordinary autoencoder, it does
    not forecast future frames the way FLD does) -- "reconstruct the same",
    per the request this was built for.
  - STUDENT encoder input: body IMU accel+gyro (pelvis/thigh/shank, both
    legs) + insole accel+gyro ("feet" IMU) + insole force/COP (48ch,
    SENSOR_COLUMNS == train_fld.COLUMNS). Reconstructs the TEACHER's own
    30ch biomech window via the SHARED decoder.
  - Both branches write into the SAME sinusoidal latent parameterization
    (phase/frequency/amplitude/offset), decoded by ONE SHARED decoder (the
    teacher core's own deconv stack) -- see Models/PAE_teacher_student.py's
    module docstring for the composition-over-reimplementation design this
    mirrors from FLD_teacher_student.py.

Three loss terms, each independently weighted (same structure as
train_fld_teacher_student.py's three-term loss, just MSE-over-a-window
instead of multi-step propagation MSE, since there's no forecast horizon
here):
  a) Teacher loss   -- teacher encoder -> shared decoder vs. its own biomech
                       input window (plain window-reconstruction MSE).
  b) Alignment loss  -- student params vs. teacher params (teacher stop-
                       gradiented) -- train_fld_teacher_student.alignment_loss,
                       reused verbatim (it only operates on LatentParams
                       tuples, model-agnostic).
  c) Student loss    -- student encoder -> shared decoder vs. the SAME
                       biomech window the teacher reconstructs.

Data: defaults to the full 4-subject (S01-S04) Data/all_data_100.csv /
Data/train_val_index_100.csv (100Hz-resampled), per explicit request. The
S01-only dev/smoke-test pair (Data/all_data_S01_100.csv /
Data/train_val_index_S01_100.csv, what this script originally defaulted to)
is still available in DataLoader/pae_teacher_student_data_loader.py as
ALL_DATA_S01_100HZ_PATH / TRAIN_VAL_INDEX_S01_100HZ_PATH -- point CONFIG's
data_path/index_path at those for a quick smoke test instead.

history_horizon=101, not 100: PAE's Conv1d layers need kernel_size ==
time_range with symmetric same-size padding, which only holds for ODD
time_range (identical requirement to FLD's history_horizon%2==1 assert --
see train_fld_teacher_student_100hz.py's own "Why 101, not 100" note, same
tie-break applied here: round up to the nearest odd value from the requested
"100 time steps").

Plotting (fires once, at the final epoch only -- displayed interactively via
plt.show(), NOT saved to disk):
  - Loss curve -- all three terms, train vs. val (see _plot_loss_curve below).
  - PCA phase manifold -- ONE COMBINED PLOT across ALL trials in data_path
    (46 trials/4 subjects for the full-dataset default; not one reference
    trial, and not one subplot per trial either -- every trial's windows are
    pooled together before a SINGLE shared PCA fit), teacher vs. student
    overlaid. See _plot_pca_manifold_all_trials below.

Run:
    python3 train_pae_teacher_student.py
    # to change a setting, edit CONFIG below directly -- no CLI flags, same
    # "fully self-contained variant script" convention as
    # train_fld_teacher_student_100hz.py.
"""

import math
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

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

from Models.PAE_teacher_student import PAETeacherStudent, LatentParams
from DataLoader.pae_teacher_student_data_loader import (
    build_datasets, SENSOR_DIM, BIOMECH_DIM,
    ALL_DATA_100HZ_PATH, TRAIN_VAL_INDEX_100HZ_PATH,
)
from train_fld import _show   # reused, not reimplemented
from train_fld_teacher_student import (   # reused, not reimplemented -- both are model-agnostic (LatentParams-only)
    alignment_loss, phase_alignment_error_degrees,
)
from Library.fld_plotter import Plotter


# ---------------------------------------------------------------------------
# Config -- fully self-contained, no argparse: every setting lives here, edit
# in place (same convention train_fld_teacher_student_100hz.py uses).
# ---------------------------------------------------------------------------

@dataclass
class PAETeacherStudentConfig:
    # Data -- full 4-subject (S01-S04) dataset by default. For the smaller
    # S01-only dev/smoke-test pair instead, set data_path/index_path to
    # DataLoader.pae_teacher_student_data_loader.ALL_DATA_S01_100HZ_PATH /
    # TRAIN_VAL_INDEX_S01_100HZ_PATH.
    data_path: str = ALL_DATA_100HZ_PATH
    index_path: str = TRAIN_VAL_INDEX_100HZ_PATH
    subjects: List[str] = field(default_factory=list)   # [] = every subject in data_path (S01-S04)

    # Window / sampling
    history_horizon: int = 101      # ~1.01s @ 100Hz; must be odd -- see module docstring "history_horizon=101, not 100"
    sample_rate: float = 100.0

    # Model
    latent_dim: int = 8
    # Hidden-stage widths for the encoder/decoder conv stacks -- FLD-style
    # explicit list (Models/FLD.py's own encoder_shape/decoder_shape
    # convention), NOT derived from the branch's input width. [64, 32] here
    # matches train_fld_teacher_student_100hz.py's own
    # teacher_encoder_shape/student_encoder_shape defaults.
    teacher_intermediate_channels: List[int] = field(default_factory=lambda: [64, 32])
    student_intermediate_channels: List[int] = field(default_factory=lambda: [64, 32])

    # Loss weights
    weight_teacher: float = 1.0
    weight_alignment: float = 1.0
    weight_student: float = 1.0
    align_weight_phase: float = 1.0
    align_weight_freq: float = 1.0
    align_weight_amp: float = 1.0
    align_weight_offset: float = 1.0

    # Teacher-only warmup -- off by default (epochs defaults to 1 for the
    # smoke-test run this script was written for; a warmup phase only makes
    # sense for a longer run -- bump epochs AND teacher_warmup_epochs
    # together for a real training run).
    teacher_warmup_enabled: bool = False
    teacher_warmup_epochs: int = 8

    # Optimisation
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 5e-4
    epochs: int = 100              # smoke-test default -- bump for a real run
    num_workers: int = 4
    save_every: int = 10
    use_amp: bool = True
    seed: int = 0

    # Output
    log_dir: str = "runs"

    # PCA-manifold plot -- cap per trial BEFORE pooling into the one combined plot
    pca_max_windows_per_trial: int = 1500


CONFIG = PAETeacherStudentConfig()


# ---------------------------------------------------------------------------
# Plotting -- loss curve. NOT train_fld_teacher_student._plot_loss_curve_triple:
# that one hardcodes an "FLD Teacher-Student" title and plots lines with no
# markers, which renders as literally nothing for a 1-epoch smoke-test run
# (a single (x, y) point has no line to draw). This local variant is
# otherwise identical (same three-term/train-vs-val/color convention) but
# adds `marker="o"` and a caller-supplied title, so it stays legible at
# epochs=1 as well as at higher epoch counts. `title` is model-agnostic on
# purpose -- train_fld_teacher_student_no_future.py imports and reuses this
# function verbatim, just passing its own title, rather than duplicating it.
# ---------------------------------------------------------------------------

def _plot_loss_curve(
    fig, ax, train_history: Dict[str, List[float]], val_history: Dict[str, List[float]],
    save_path: str = None, title: str = "PAE Teacher-Student Training Loss — solid = train, dashed = val",
) -> None:
    colors = {"teacher": "seagreen", "alignment": "goldenrod", "student": "tomato"}
    ax.cla()
    n_epochs = len(train_history["teacher"])
    epochs = range(1, n_epochs + 1)
    for term, color in colors.items():
        ax.plot(epochs, train_history[term], color=color, lw=1.6, marker="o", ls="-", label=f"{term} (train)")
        ax.plot(epochs, val_history[term], color=color, lw=1.6, marker="o", ls="--", label=f"{term} (val)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (normalised space)")
    ax.set_title(title)
    ax.set_xticks(list(epochs))
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    _show(fig, save_path)


# ---------------------------------------------------------------------------
# Plotting -- PCA phase manifold, ONE COMBINED PLOT across ALL trials:
# every trial's windows are encoded and pooled together BEFORE handing them
# to Plotter.plot_pca, so the PCA is fit ONCE on the combined teacher+student
# points from every trial (not one independent PCA fit per trial) -- per
# explicit request ("I meant one combined script across all trials"),
# replacing an earlier one-subplot-per-trial grid version (see git history).
# ---------------------------------------------------------------------------

def _plot_pca_manifold_all_trials(
    model: PAETeacherStudent, trial_datasets: Dict, device: torch.device,
    history_horizon: int, plotter: Plotter, max_windows_per_trial: int = 1500,
    batch_size: int = 256, save_path: str = None,
):
    H = history_horizon
    model.eval()
    teacher_phases, teacher_amps, student_phases, student_amps = [], [], [], []
    with torch.no_grad():
        for dataset in trial_datasets.values():
            n_windows = len(dataset.data_sensor) - H + 1
            if n_windows <= 0:
                continue
            # Stride+cap PER TRIAL first (so a long trial doesn't drown out a
            # short one in the pooled point cloud), THEN pool everything.
            stride = max(1, n_windows // max_windows_per_trial)
            starts = list(range(0, n_windows, stride))[:max_windows_per_trial]

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

    fig, ax = plt.subplots(figsize=(8, 8))
    # draw_line/draw_arrow off: Plotter.plot_pca connects CONSECUTIVE points
    # by default, which reads as a single trajectory within one trial -- once
    # points from different trials are pooled and concatenated, those same
    # connecting lines/arrows would jump meaninglessly between unrelated
    # trials at every trial boundary. Pure scatter instead.
    plotter.plot_pca(ax, [teacher_manifold, student_manifold], point_color=["seagreen", "tomato"],
                      draw_line=False, draw_arrow=False)
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="seagreen", markersize=8, label="Teacher"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="tomato", markersize=8, label="Student"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10)
    ax.set_title(f"PCA Phase Manifold — Teacher vs Student — all {len(trial_datasets)} trials combined",
                 fontsize=12, fontweight="bold")
    _show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def _save(model, optimizer, epoch, config, norm_stats, log_dir):
    path = os.path.join(log_dir, f"pae_teacher_student_epoch_{epoch}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "config": asdict(config),
        "norm_stats": norm_stats,
    }, path)
    print(f"[PAE-TS] Checkpoint saved -> {path}")
    return path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: PAETeacherStudentConfig) -> str:
    torch.manual_seed(config.seed)
    assert config.history_horizon % 2 == 1, f"history_horizon must be odd (got {config.history_horizon})."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PAE-TS] Using device: {device}")
    if device.type == "cuda":
        print(f"[PAE-TS] GPU: {torch.cuda.get_device_name(device)}")

    data_device = device if device.type == "cuda" else None
    num_workers = 0 if data_device is not None else config.num_workers

    use_amp = config.use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16

    log_dir = os.path.join(config.log_dir, f"PAETeacherStudent_100Hz_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)

    train_ds, val_ds, norm_stats, trial_datasets = build_datasets(
        config.history_horizon, subjects=config.subjects,
        data_path=config.data_path, index_path=config.index_path, device=data_device,
    )

    loader_kwargs = dict(num_workers=num_workers, pin_memory=(data_device is None))
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                           drop_last=True, **loader_kwargs)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                         drop_last=False, **loader_kwargs)

    model = PAETeacherStudent(
        sensor_dim=SENSOR_DIM, biomech_dim=BIOMECH_DIM,
        history_horizon=config.history_horizon, latent_dim=config.latent_dim,
        device=device, dt=1.0 / config.sample_rate,
        teacher_intermediate_channels=config.teacher_intermediate_channels,
        student_intermediate_channels=config.student_intermediate_channels,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    plotter = Plotter()
    fig_loss, ax_loss = plt.subplots(figsize=(9, 5))

    train_loss_history = {"teacher": [], "alignment": [], "student": []}
    val_loss_history = {"teacher": [], "alignment": [], "student": []}

    train_start = datetime.now()
    print(f"[PAE-TS] Training started at {train_start.strftime('%Y-%m-%d %H:%M:%S')}.")
    for epoch in range(config.epochs):
        epoch_start = datetime.now()
        model.train()
        warmup_active = config.teacher_warmup_enabled and epoch < config.teacher_warmup_epochs

        running = {"teacher": 0.0, "alignment": 0.0, "student": 0.0, "total": 0.0}
        n_batches = 0

        for batch in train_dl:
            sensor_win = batch["sensor_window"].to(device, non_blocking=True)     # (B, sensor_dim, H)
            biomech_win = batch["biomech_window"].to(device, non_blocking=True)   # (B, biomech_dim, H) -- teacher input AND recon target

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                teacher_params = model.forward_teacher(biomech_win)
                teacher_recon = model.decode(teacher_params)
                loss_teacher = F.mse_loss(teacher_recon, biomech_win)

                student_params = model.forward_student(sensor_win)
                loss_align, _align_components = alignment_loss(
                    student_params, teacher_params,
                    config.align_weight_phase, config.align_weight_freq,
                    config.align_weight_amp, config.align_weight_offset,
                )
                student_recon = model.decode(student_params)
                loss_student = F.mse_loss(student_recon, biomech_win)

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
        phase_err_accum = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_dl:
                sensor_win = batch["sensor_window"].to(device, non_blocking=True)
                biomech_win = batch["biomech_window"].to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    teacher_params = model.forward_teacher(biomech_win)
                    teacher_recon = model.decode(teacher_params)
                    v_loss_teacher = F.mse_loss(teacher_recon, biomech_win)

                    student_params = model.forward_student(sensor_win)
                    v_loss_align, _ = alignment_loss(
                        student_params, teacher_params,
                        config.align_weight_phase, config.align_weight_freq,
                        config.align_weight_amp, config.align_weight_offset,
                    )
                    student_recon = model.decode(student_params)
                    v_loss_student = F.mse_loss(student_recon, biomech_win)

                val_running["teacher"] += v_loss_teacher.item()
                val_running["alignment"] += v_loss_align.item()
                val_running["student"] += v_loss_student.item()
                phase_err_accum += phase_alignment_error_degrees(student_params[0], teacher_params[0])
                n_val_batches += 1

        n_val_batches = max(n_val_batches, 1)
        for k in val_running:
            val_running[k] /= n_val_batches
        phase_err_deg = phase_err_accum / n_val_batches

        for term in ("teacher", "alignment", "student"):
            train_loss_history[term].append(running[term])
            val_loss_history[term].append(val_running[term])

        warmup_str = " [teacher warmup]" if warmup_active else ""
        print(f"[PAE-TS] Epoch [{epoch + 1:>4}/{config.epochs}]{warmup_str}  "
              f"train: total={running['total']:.5f} teacher={running['teacher']:.5f} "
              f"alignment={running['alignment']:.5f} student={running['student']:.5f}  |  "
              f"val: teacher={val_running['teacher']:.5f} alignment={val_running['alignment']:.5f} "
              f"student={val_running['student']:.5f}  |  phase_err={phase_err_deg:.2f} deg")

        if epoch == config.epochs - 1:
            _plot_loss_curve(fig_loss, ax_loss, train_loss_history, val_loss_history)
            _plot_pca_manifold_all_trials(
                model, trial_datasets, device, config.history_horizon, plotter,
                max_windows_per_trial=config.pca_max_windows_per_trial,
            )
            model.train()

        if epoch % config.save_every == 0 or epoch == config.epochs - 1:
            _save(model, optimizer, epoch, config, norm_stats, log_dir)

        epoch_end = datetime.now()
        print(f"[PAE-TS] Epoch {epoch + 1} finished at {epoch_end.strftime('%Y-%m-%d %H:%M:%S')}  "
              f"(epoch took {(epoch_end - epoch_start).total_seconds():.1f}s, "
              f"elapsed {(epoch_end - train_start).total_seconds() / 60:.1f} min total)")

    print(f"[PAE-TS] Training finished. Checkpoints in: {log_dir}")
    plt.ioff()
    plt.show()
    return log_dir


if __name__ == "__main__":
    train(CONFIG)
