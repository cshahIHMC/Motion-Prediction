"""
train_fld_teacher_student_no_future.py

FLD counterpart to train_pae_teacher_student.py, inspired by the "Deep
Fourier Mimic" paper's own variant of FLD: same Models/FLD_teacher_student.py
(FLDTeacherStudent) architecture as train_fld_teacher_student.py /
train_fld_teacher_student_100hz.py, but with the forecast/future-prediction
capability set to 0 -- i.e. FLD used purely as a per-window AUTOENCODER
(reconstruct the SAME window it encoded), exactly the training objective
train_pae_teacher_student.py already uses for PAE. This is NOT a new model
file -- FLDTeacherStudent's own decode() already supports this: calling
decode(params, horizon_steps=1) and taking index [0] gives the
NO-phase-shift ("step 0") reconstruction of the CURRENT window (phase_dynamics
at step 0 is just `phase + frequency*dt*0 == phase`, i.e. no time-roll at
all) -- forecast() is simply never called, since forecast() is specifically
the "drop step 0, keep only genuinely-future steps" helper and step 0 is
exactly the reconstruction we want here. See Models/FLD_teacher_student.py's
decode()/forecast() docstrings for the k/horizon_steps convention this
reuses verbatim.

Reuses train_pae_teacher_student.py's DATA loader, loss/plotting machinery
almost entirely -- the actual "no future" behaviour lives entirely in how
THIS script calls FLDTeacherStudent.decode(), nowhere else:
  - DataLoader/pae_teacher_student_data_loader.py's build_datasets/
    SENSOR_DIM/BIOMECH_DIM/ALL_DATA_100HZ_PATH/TRAIN_VAL_INDEX_100HZ_PATH --
    reused AS-IS, unmodified. Its `window_size == history_horizon` (no
    `+forecast_horizon`) and `biomech_window` doubling as both the teacher's
    context input AND the reconstruction target are exactly what "no future"
    needs -- this loader was already built for that, nothing FLD-specific
    about it (SENSOR_COLUMNS/BIOMECH_COLUMNS are plain column lists).
  - train_fld_teacher_student.alignment_loss / phase_alignment_error_degrees
    -- reused verbatim (LatentParams-only, model-agnostic; FLDTeacherStudent
    and PAETeacherStudent both produce the exact same
    (phase, frequency, amplitude, offset) 4-tuple convention).
  - train_pae_teacher_student._plot_loss_curve (now takes a `title` kwarg
    precisely so this script can reuse it with its own title instead of
    duplicating it) and _plot_pca_manifold_all_trials (calls only
    model.forward_teacher()/forward_student(), never decode() -- so it's
    already fully model-agnostic, no changes needed to reuse it here).

Three loss terms, same structure/weights as train_pae_teacher_student.py:
  a) Teacher loss   -- teacher encoder -> shared decoder, horizon_steps=1,
                       vs. its own biomech input window (window MSE).
  b) Alignment loss  -- student params vs. teacher params (teacher stop-
                       gradiented).
  c) Student loss    -- student encoder -> shared decoder, horizon_steps=1,
                       vs. the SAME biomech window the teacher reconstructs.

Model shapes: teacher_encoder_shape/student_encoder_shape/decoder_shape =
[64, 32]/[64, 32]/[32, 64] -- the SAME values train_fld_teacher_student_100hz.py
already defaults to, and structurally equivalent to PAE's current [64, 32]
cascade (30 -> 64 -> 32 -> 8 -> 32 -> 64 -> 30) EXCEPT FLD's decoder_shape is
an independently-settable list (unlike PAE's, which is forced to mirror
whichever branch built the shared decoder, reversed -- see the "where do we
set the decoder shape" discussion this script's docstring assumes context
from).

Data: full 4-subject (S01-S04) Data/all_data_100.csv / Data/train_val_index_100.csv
by default, same as train_pae_teacher_student.py's current CONFIG.

Run:
    python3 train_fld_teacher_student_no_future.py
    # to change a setting, edit CONFIG below directly -- no CLI flags, same
    # "fully self-contained variant script" convention as
    # train_fld_teacher_student_100hz.py / train_pae_teacher_student.py.
"""

import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import List

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

sys.path.insert(0, os.path.dirname(__file__))

from Models.FLD_teacher_student import FLDTeacherStudent
from DataLoader.pae_teacher_student_data_loader import (   # reused AS-IS -- see module docstring
    build_datasets, SENSOR_DIM, BIOMECH_DIM,
    ALL_DATA_100HZ_PATH, TRAIN_VAL_INDEX_100HZ_PATH,
)
from train_fld_teacher_student import alignment_loss, phase_alignment_error_degrees   # reused, not reimplemented
from train_pae_teacher_student import _plot_loss_curve, _plot_pca_manifold_all_trials   # reused, not reimplemented
from Library.fld_plotter import Plotter


# ---------------------------------------------------------------------------
# Config -- fully self-contained, no argparse: every setting lives here, edit
# in place. Mirrors train_pae_teacher_student.py's PAETeacherStudentConfig
# field-for-field where the two overlap; teacher_encoder_shape/
# student_encoder_shape/decoder_shape replace PAE's single
# teacher_intermediate_channels/student_intermediate_channels pair, since
# FLD keeps the decoder shape independently settable (see module docstring).
# ---------------------------------------------------------------------------

@dataclass
class FLDNoFutureConfig:
    # Data -- full 4-subject (S01-S04) dataset by default.
    data_path: str = ALL_DATA_100HZ_PATH
    index_path: str = TRAIN_VAL_INDEX_100HZ_PATH
    subjects: List[str] = field(default_factory=list)   # [] = every subject in data_path (S01-S04)

    # Window / sampling
    history_horizon: int = 101      # ~1.01s @ 100Hz; must be odd -- FLD's Conv1d same-padding requirement
    sample_rate: float = 100.0

    # Model
    latent_dim: int = 8
    teacher_encoder_shape: List[int] = field(default_factory=lambda: [128, 64])
    student_encoder_shape: List[int] = field(default_factory=lambda: [128, 64])
    decoder_shape: List[int] = field(default_factory=lambda: [64, 128])

    # Loss weights
    weight_teacher: float = 1.0
    weight_alignment: float = 1.0
    weight_student: float = 1.0
    align_weight_phase: float = 1.0
    align_weight_freq: float = 1.0
    align_weight_amp: float = 1.0
    align_weight_offset: float = 1.0

    # Teacher-only warmup -- off by default, same reasoning as
    # train_pae_teacher_student.py (only worth enabling for epochs >> 1).
    teacher_warmup_enabled: bool = False
    teacher_warmup_epochs: int = 8

    # Optimisation
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 5e-4
    epochs: int = 100
    num_workers: int = 4
    save_every: int = 10
    use_amp: bool = True
    seed: int = 0

    # Output
    log_dir: str = "runs"

    # PCA-manifold plot -- cap per trial BEFORE pooling into the one combined plot
    pca_max_windows_per_trial: int = 1500


CONFIG = FLDNoFutureConfig()


# ---------------------------------------------------------------------------
# Checkpoint -- distinct filename stem ("fld_teacher_student_no_future") so
# runs never collide with train_fld_teacher_student.py's own
# "fld_teacher_student" or the 100Hz variant's "fld_teacher_student_100hz"
# stems -- same precedent the GRF variant already set for itself.
# ---------------------------------------------------------------------------

def _save(model, optimizer, epoch, config, norm_stats, log_dir):
    path = os.path.join(log_dir, f"fld_teacher_student_no_future_epoch_{epoch}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "config": asdict(config),
        "norm_stats": norm_stats,
    }, path)
    print(f"[FLD-TS-NoFuture] Checkpoint saved -> {path}")
    return path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: FLDNoFutureConfig) -> str:
    torch.manual_seed(config.seed)
    assert config.history_horizon % 2 == 1, f"history_horizon must be odd (got {config.history_horizon})."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FLD-TS-NoFuture] Using device: {device}")
    if device.type == "cuda":
        print(f"[FLD-TS-NoFuture] GPU: {torch.cuda.get_device_name(device)}")

    data_device = device if device.type == "cuda" else None
    num_workers = 0 if data_device is not None else config.num_workers

    use_amp = config.use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16

    log_dir = os.path.join(config.log_dir, f"FLDTeacherStudentNoFuture_100Hz_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
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

    model = FLDTeacherStudent(
        sensor_dim=SENSOR_DIM, biomech_dim=BIOMECH_DIM,
        history_horizon=config.history_horizon, latent_dim=config.latent_dim,
        device=device, dt=1.0 / config.sample_rate,
        teacher_encoder_shape=config.teacher_encoder_shape,
        student_encoder_shape=config.student_encoder_shape,
        decoder_shape=config.decoder_shape,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    plotter = Plotter()
    fig_loss, ax_loss = plt.subplots(figsize=(9, 5))

    train_loss_history = {"teacher": [], "alignment": [], "student": []}
    val_loss_history = {"teacher": [], "alignment": [], "student": []}

    train_start = datetime.now()
    print(f"[FLD-TS-NoFuture] Training started at {train_start.strftime('%Y-%m-%d %H:%M:%S')}.")
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
                # horizon_steps=1 -> step 0 -> NO phase shift -> reconstructs
                # the CURRENT window, exactly PAE's own reconstruction
                # target -- forecast() is deliberately never called (it
                # would drop this exact step). See module docstring.
                teacher_recon = model.decode(teacher_params, horizon_steps=1)[0]
                loss_teacher = F.mse_loss(teacher_recon, biomech_win)

                student_params = model.forward_student(sensor_win)
                loss_align, _align_components = alignment_loss(
                    student_params, teacher_params,
                    config.align_weight_phase, config.align_weight_freq,
                    config.align_weight_amp, config.align_weight_offset,
                )
                student_recon = model.decode(student_params, horizon_steps=1)[0]
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
                    teacher_recon = model.decode(teacher_params, horizon_steps=1)[0]
                    v_loss_teacher = F.mse_loss(teacher_recon, biomech_win)

                    student_params = model.forward_student(sensor_win)
                    v_loss_align, _ = alignment_loss(
                        student_params, teacher_params,
                        config.align_weight_phase, config.align_weight_freq,
                        config.align_weight_amp, config.align_weight_offset,
                    )
                    student_recon = model.decode(student_params, horizon_steps=1)[0]
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
        print(f"[FLD-TS-NoFuture] Epoch [{epoch + 1:>4}/{config.epochs}]{warmup_str}  "
              f"train: total={running['total']:.5f} teacher={running['teacher']:.5f} "
              f"alignment={running['alignment']:.5f} student={running['student']:.5f}  |  "
              f"val: teacher={val_running['teacher']:.5f} alignment={val_running['alignment']:.5f} "
              f"student={val_running['student']:.5f}  |  phase_err={phase_err_deg:.2f} deg")

        if epoch == config.epochs - 1:
            _plot_loss_curve(
                fig_loss, ax_loss, train_loss_history, val_loss_history,
                title="FLD Teacher-Student (no future) Training Loss — solid = train, dashed = val",
            )
            _plot_pca_manifold_all_trials(
                model, trial_datasets, device, config.history_horizon, plotter,
                max_windows_per_trial=config.pca_max_windows_per_trial,
            )
            model.train()

        if epoch % config.save_every == 0 or epoch == config.epochs - 1:
            _save(model, optimizer, epoch, config, norm_stats, log_dir)

        epoch_end = datetime.now()
        print(f"[FLD-TS-NoFuture] Epoch {epoch + 1} finished at {epoch_end.strftime('%Y-%m-%d %H:%M:%S')}  "
              f"(epoch took {(epoch_end - epoch_start).total_seconds():.1f}s, "
              f"elapsed {(epoch_end - train_start).total_seconds() / 60:.1f} min total)")

    print(f"[FLD-TS-NoFuture] Training finished. Checkpoints in: {log_dir}")
    plt.ioff()
    plt.show()
    return log_dir


if __name__ == "__main__":
    train(CONFIG)
