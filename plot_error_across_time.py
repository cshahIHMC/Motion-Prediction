"""
plot_error_across_time.py

Compares how prediction error grows across the forecast horizon (first
EVAL_STEPS = 20 steps, x-axis 0-20) for the LATEST TCNN checkpoint (TCNN_RUN
below, blank = auto-pick the most recent runs/TCNN_* run) and the pinned
70-epoch FLD-TeacherStudent (student branch) checkpoint (FLD_TS_RUN). The old
FLD-151 comparison is commented out in model_loaders (see there). Evaluated on
the HELD-OUT subject S05 (EVAL_HELD_OUT below -- every window of every S05 trial in
Data/all_data_S05_150_corrected.csv, a subject neither model trained on; no index CSV
involved). With EVAL_HELD_OUT = False it instead evaluates the validation windows
recorded in Data/train_val_index_150_corrected.csv
(built by build_train_val_index.py): ~15% of each trial's windows, spread
across a few segments through the trial rather than one chronological tail,
with a window-sized gap enforced at every train/val boundary so no evaluated
window overlaps anything the checkpoint could have trained on.
(PLOT_TRAIN_SET below can additionally evaluate each model on its training
windows too, for an overfitting check.)

The FLD-75 (history_horizon=75, shorter-context ablation) comparison is
commented out below in model_loaders -- not deleted, just disabled for now.
Uncomment that line (and set FLD75_RUN to a pinned runs/FLD_* directory, same
convention as FLD151_RUN) to bring it back.

Each checkpoint's own history_horizon/forecast_horizon determines its
window_size; _load_split_index() checks that against the index's own
window_size (see build_train_val_index.py) before evaluating, so a
checkpoint trained with a larger window than the persisted index was built
for fails loudly instead of silently reading windows too close to train data.

For every forecast step k = 1..forecast_horizon, RMSE is computed in
original physical units, pooled across every validation window from every
task, separately for the joint-angle (deg) and joint-moment (N.m/kg)
modality groups (mixing units into one number would be meaningless).
Produces one figure with two subplots (angles, moments), each with one curve
per model, so you can see directly whether FLD or TCNN degrades faster/
slower as the prediction horizon extends.

Run:
    python3 plot_error_across_time.py
"""

import os
import sys
import glob
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from DataLoader.data_loader import FLDDataset
from DataLoader.tcnn_data_loader import TCNNDataset
from Models.FLD import FLD
from Models.TCNN import TCNModel_Forecast
from Models.FLD_teacher_student import FLDTeacherStudent
from DataLoader.teacher_student_data_loader import SENSOR_COLUMNS as TS_SENSOR_COLUMNS, SENSOR_DIM as TS_SENSOR_DIM

# ---------------------------------------------------------------------------
# Paths & column definitions — same as train_fld.py / train_tcnn.py
# ---------------------------------------------------------------------------

DATA_DIR      = os.path.join(os.path.dirname(__file__), "Data")
TRAIN_DATA_PATH    = os.path.join(DATA_DIR, "all_data_150_corrected.csv")          # built by combine_data.py (150 Hz, corrected)
HELD_OUT_DATA_PATH = os.path.join(DATA_DIR, "all_data_S05_150_corrected.csv")      # held-out subject S05, never trained on
INDEX_PATH         = os.path.join(DATA_DIR, "train_val_index_150_corrected.csv")   # built by build_train_val_index.py

# True (default): leave-one-subject-out evaluation -- score every window (stride 1) of every trial in
#                 HELD_OUT_DATA_PATH; no train_val_index is read, and there is no "train" split to plot.
# False:          the original within-trial train/val split from INDEX_PATH on TRAIN_DATA_PATH.
EVAL_HELD_OUT = True
ALL_DATA_PATH = HELD_OUT_DATA_PATH if EVAL_HELD_OUT else TRAIN_DATA_PATH
RUNS_DIR      = os.path.join(os.path.dirname(__file__), "runs")

# ---------------------------------------------------------------------------
# Model selection — pin a specific run by its runs/ directory name (e.g.
# "FLD_20260722_181921"), or leave blank ("") to auto-pick that model's most
# recent checkpoint. If a field is blank and no run of that type exists yet
# (or a pinned run/checkpoint can't be found), that model is skipped and the
# script just plots whichever model(s) ARE available.
#
# FLD151_RUN and FLD75_RUN both load via the same _load_fld() / FLD class --
# they're just two checkpoints with different history_horizon baked into
# their own saved config (151 vs 75 frames of context). FLD75_RUN is unused
# while its model_loaders entry below is commented out (see that comment);
# left defined here so uncommenting is a one-line change, not a re-add.
# ---------------------------------------------------------------------------
FLD151_RUN = ""   # history_horizon=151 -- blank = latest runs/FLD_* run
FLD75_RUN  = ""   # history_horizon=75  -- blank = latest runs/FLD_* run (currently unused, see model_loaders)
TCNN_RUN   = ""                      # blank = latest runs/TCNN_* run
FLD_TS_RUN = "FLDTeacherStudent_20260921_163758"   # pinned -- 70 epochs, history 151 / forecast 20, see _load_fld_teacher_student

# Only the first EVAL_STEPS forecast steps are evaluated/plotted (x-axis 0-EVAL_STEPS), so
# models trained with different forecast horizons (e.g. TCNN at 30, FLD-TeacherStudent at 20)
# are compared over the SAME 20 steps. A model with a shorter horizon just plots what it has.
EVAL_STEPS = 20

# False (default): one pooled figure — RMSE per forecast step summed across
#                  every trial's validation windows, then sqrt'd once.
# True:            one figure PER trial — RMSE computed independently within
#                  each trial's own validation windows, so you can see which
#                  specific tasks drive the pooled curve vs. which are flat.
SEPARATE_TRIALS = False

# False: evaluate on validation windows only.
# True (now default): ALSO evaluate on training windows and add them as extra
#                  curves ("<Model> (train)" alongside "<Model> (val)"), same
#                  color per model, dashed = train / solid = val -- lets you
#                  see the train/val gap directly (overfitting check). Costs
#                  real time: there are ~7x more train windows than val
#                  windows per trial (the whole point of an ~85/15 split), so
#                  this roughly doubles the model's total eval time.
PLOT_TRAIN_SET = True   # ignored (treated as False) when EVAL_HELD_OUT -- a held-out subject has no train windows

COLUMNS = [
    'pelvis_acc_x',  'pelvis_acc_y',  'pelvis_acc_z',
    'thigh_r_acc_x', 'thigh_r_acc_y', 'thigh_r_acc_z',
    'shank_r_acc_x', 'shank_r_acc_y', 'shank_r_acc_z',
    'thigh_l_acc_x', 'thigh_l_acc_y', 'thigh_l_acc_z',
    'shank_l_acc_x', 'shank_l_acc_y', 'shank_l_acc_z',
    'pelvis_gyro_x',  'pelvis_gyro_y',  'pelvis_gyro_z',
    'thigh_r_gyro_x', 'thigh_r_gyro_y', 'thigh_r_gyro_z',
    'shank_r_gyro_x', 'shank_r_gyro_y', 'shank_r_gyro_z',
    'thigh_l_gyro_x', 'thigh_l_gyro_y', 'thigh_l_gyro_z',
    'shank_l_gyro_x', 'shank_l_gyro_y', 'shank_l_gyro_z',
    'L_insole_accel_x', 'L_insole_accel_y', 'L_insole_accel_z',
    'R_insole_accel_x', 'R_insole_accel_y', 'R_insole_accel_z',
    'L_insole_gyro_x', 'L_insole_gyro_y', 'L_insole_gyro_z',
    'R_insole_gyro_x', 'R_insole_gyro_y', 'R_insole_gyro_z',
    'L_insole_force', 'L_insole_COPx', 'L_insole_COPz',
    'R_insole_force', 'R_insole_COPx', 'R_insole_COPz',
]
SUBJECT_COLUMNS = ['subject_weight', 'subject_height']
INPUT_COLUMNS = COLUMNS + SUBJECT_COLUMNS

# Current convention (matches train_fld.py exactly) -- 10 joint angles + 10
# joint angular velocities + 10 joint moments = 30 features. FLD checkpoints
# trained since velocities were added have output_dim=30 to match; TCNN was
# never updated to predict velocities (train_tcnn.py's own OUTPUT_COLUMNS is
# still just angle+moment = 20) -- see TCNN_OUTPUT_COLUMNS below, which is
# this same 30-column list with the velocity columns removed, matching
# TCNN's actual trained shape. Every checkpoint's own output_dim is read
# from ckpt["out_mean"].shape[0] at load time (see _load_fld/_load_tcnn),
# not assumed from these constants -- these just have to match whichever of
# the two conventions is actually in use so task_dfs (built once, shared by
# every model) carries the right columns for whichever model reads it.
OUTPUT_COLUMNS = [
    'hip_flexion_r',    'hip_flexion_l',
    'hip_adduction_r',  'hip_adduction_l',
    'hip_rotation_r',   'hip_rotation_l',
    'knee_angle_r',     'knee_angle_l',
    'ankle_angle_r',    'ankle_angle_l',
    'hip_flexion_r_velocity',    'hip_flexion_l_velocity',
    'hip_adduction_r_velocity',  'hip_adduction_l_velocity',
    'hip_rotation_r_velocity',   'hip_rotation_l_velocity',
    'knee_angle_r_velocity',     'knee_angle_l_velocity',
    'ankle_angle_r_velocity',    'ankle_angle_l_velocity',
    'hip_flexion_r_moment',   'hip_flexion_l_moment',
    'hip_adduction_r_moment', 'hip_adduction_l_moment',
    'hip_rotation_r_moment',  'hip_rotation_l_moment',
    'knee_angle_r_moment',    'knee_angle_l_moment',
    'ankle_angle_r_moment',   'ankle_angle_l_moment',
]
MOMENT_COLUMNS = OUTPUT_COLUMNS[20:]   # the 10 *_moment columns -- normalised by subject weight

# TCNN's own output columns (train_tcnn.py never gained a velocity output) --
# angle + moment only, same 20 columns the old (pre-velocity) FLD convention
# used too. build_dataset() for TCNN slices task_dfs down to exactly this
# set before constructing TCNNDataset -- see _load_tcnn.
TCNN_OUTPUT_COLUMNS = OUTPUT_COLUMNS[0:10] + OUTPUT_COLUMNS[20:30]
TCNN_OUTPUT_DIM = len(TCNN_OUTPUT_COLUMNS)

INPUT_DIM  = len(INPUT_COLUMNS)
OUTPUT_DIM = len(OUTPUT_COLUMNS)


def _modality_groups_for_output_dim(output_dim):
    """angle/moment index groups for a SPECIFIC model's own output_dim -- 20
    (angle+moment only, e.g. TCNN or an old pre-velocity FLD checkpoint) or
    30 (angle+velocity+moment, current FLD convention -- velocity is
    deliberately skipped here since this script only ever compares angle vs.
    moment, the two modalities every model it can load actually shares)."""
    if output_dim == 20:
        return {
            "angle":  {"indices": list(range(0, 10)),  "unit": "deg",    "label": "Joint Angle RMSE (deg)"},
            "moment": {"indices": list(range(10, 20)), "unit": "N·m/kg", "label": "Joint Moment RMSE (N·m/kg)"},
        }
    if output_dim == 30:
        return {
            "angle":  {"indices": list(range(0, 10)),  "unit": "deg",    "label": "Joint Angle RMSE (deg)"},
            "moment": {"indices": list(range(20, 30)), "unit": "N·m/kg", "label": "Joint Moment RMSE (N·m/kg)"},
        }
    raise ValueError(
        f"Unrecognised output_dim={output_dim} -- expected 20 (angle+moment) or 30 "
        f"(angle+velocity+moment, velocity skipped in this angle/moment-only comparison)."
    )


# ---------------------------------------------------------------------------
# Data loading — Data/all_data_150_corrected.csv (combine_data.py) + Data/train_val_index_150_corrected.csv
# (build_train_val_index.py). Both must exist; run those scripts first if
# either is missing.
# ---------------------------------------------------------------------------

def _load_all_data():
    """Read Data/all_data_150_corrected.csv and normalise joint moments to N·m/kg by
    dividing each row's moment columns by that row's own subject_weight (kg)
    -- so torque is reported per unit body mass instead of raw N·m, and is
    comparable across subjects of different weight."""
    if not os.path.exists(ALL_DATA_PATH):
        raise FileNotFoundError(f"{ALL_DATA_PATH} not found -- run combine_data.py first.")
    usecols = list(dict.fromkeys(['trial_id', 'subject_name', 'task'] + INPUT_COLUMNS + OUTPUT_COLUMNS))
    df = pd.read_csv(ALL_DATA_PATH, usecols=usecols)
    df[MOMENT_COLUMNS] = df[MOMENT_COLUMNS].div(df['subject_weight'], axis=0)
    return df


def _discover_tasks(all_df):
    """trial_id -> {'label'}, read from each trial's first row."""
    tasks = {}
    for trial_id, g in all_df.groupby('trial_id', sort=False):
        tasks[trial_id] = {'label': str(g.iloc[0]['task'])}
    return tasks


def _load_split_index(window_size):
    """trial_id -> {'train': [...], 'val': [local_start_idx...]}, from
    Data/train_val_index_150_corrected.csv. Verifies the index was built for a window at
    least as large as window_size -- see train_fld.py's _load_split_index
    for why a smaller-window index isn't safe to reuse here."""
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"{INDEX_PATH} not found -- run build_train_val_index.py first.")
    idx_df = pd.read_csv(INDEX_PATH, usecols=['trial_id', 'split', 'local_start_idx', 'window_size'])
    index_window_size = int(idx_df['window_size'].min())
    if index_window_size < window_size:
        raise ValueError(
            f"{INDEX_PATH} was built for window_size={index_window_size}, smaller than this "
            f"checkpoint's window_size={window_size} -- rerun build_train_val_index.py with "
            f"HISTORY_HORIZON/FORECAST_HORIZON matching (or exceeding) this checkpoint's config."
        )
    splits = {}
    for trial_id, g in idx_df.groupby('trial_id', sort=False):
        splits[trial_id] = {
            'train': g.loc[g['split'] == 'train', 'local_start_idx'].tolist(),
            'val':   g.loc[g['split'] == 'val',   'local_start_idx'].tolist(),
        }
    return splits


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def _find_checkpoint(run_prefix, checkpoint_stem, pinned_run=""):
    """Return (run_dir, checkpoint_path) for a model's checkpoint.

    If pinned_run is given, look only inside runs/<pinned_run> for its
    highest-epoch {checkpoint_stem}_epoch_*.pt. Otherwise search every
    runs/{run_prefix}_* directory, newest first, and use the highest-epoch
    checkpoint in the first one that actually has any."""
    pattern = re.compile(rf"^{re.escape(checkpoint_stem)}_epoch_(\d+)\.pt$")

    if pinned_run:
        run_dirs = [os.path.join(RUNS_DIR, pinned_run)]
    else:
        run_dirs = sorted(glob.glob(os.path.join(RUNS_DIR, f"{run_prefix}_*")), reverse=True)

    for run_dir in run_dirs:
        ckpts = []
        for path in glob.glob(os.path.join(run_dir, f"{checkpoint_stem}_epoch_*.pt")):
            m = pattern.match(os.path.basename(path))
            if m:
                ckpts.append((int(m.group(1)), path))
        if ckpts:
            ckpts.sort(key=lambda t: t[0])
            return run_dir, ckpts[-1][1]

    if pinned_run:
        raise FileNotFoundError(
            f"No '{checkpoint_stem}_epoch_*.pt' checkpoints found in {os.path.join(RUNS_DIR, pinned_run)}"
        )
    raise FileNotFoundError(
        f"No '{checkpoint_stem}_epoch_*.pt' checkpoints found under {RUNS_DIR}/{run_prefix}_* "
        f"— train that model first."
    )


def _load_state_dict_flexible(model, state_dict):
    """torch.compile occasionally prefixes state_dict keys with '_orig_mod.'
    — strip it if a plain load fails."""
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        stripped = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in state_dict.items()
        }
        model.load_state_dict(stripped)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_fld(device, pinned_run=""):
    run_dir, ckpt_path = _find_checkpoint("FLD", "fld_model", pinned_run)
    print(f"[ErrorPlot] FLD checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    # Read output_dim from the checkpoint itself, not the module-level
    # OUTPUT_DIM constant -- a checkpoint trained before/after a column-count
    # change (e.g. velocities added to OUTPUT_COLUMNS) must be built with ITS
    # OWN shape or state_dict loading fails with a size-mismatch. See
    # train_fld.py's module docstring for the exact history of this.
    output_dim = ckpt["out_mean"].shape[0]
    model = FLD(
        observation_dim=INPUT_DIM,
        output_dim=output_dim,
        history_horizon=config["history_horizon"],
        latent_channel=config["latent_dim"],
        device=device,
        dt=1.0 / 150.0,
        encoder_shape=config["encoder_shape"],
        decoder_shape=config["decoder_shape"],
    )
    _load_state_dict_flexible(model, ckpt["fld_state_dict"])
    model.eval()

    norm_stats = {
        'mean_in':  ckpt["in_mean"].cpu().numpy(),  'std_in':  ckpt["in_std"].cpu().numpy(),
        'mean_out': ckpt["out_mean"].cpu().numpy(), 'std_out': ckpt["out_std"].cpu().numpy(),
    }
    out_std = ckpt["out_std"].cpu().numpy()
    modality_groups = _modality_groups_for_output_dim(output_dim)

    def build_dataset(df):
        # task_dfs (built once in main(), shared across models) carries the
        # FULL current OUTPUT_COLUMNS (30-wide) -- if this checkpoint's own
        # output_dim is smaller (an old 20-wide checkpoint), slice down to
        # just its own angle+moment columns before constructing FLDDataset,
        # same reasoning as _load_tcnn's TCNN_OUTPUT_COLUMNS slice below.
        cols = INPUT_COLUMNS + (OUTPUT_COLUMNS if output_dim == OUTPUT_DIM else TCNN_OUTPUT_COLUMNS)
        return FLDDataset(
            df[cols], history_horizon=config["history_horizon"],
            forecast_horizon=config["forecast_horizon"],
            feature_set='cross', norm_stats=norm_stats,
            device=device, input_dim=INPUT_DIM,
        )

    def forecast(context, forecast_horizon):
        pred_dynamics, _, _, _ = model(context, k=forecast_horizon + 1)
        return pred_dynamics[1:, :, :, -1]   # (K, B, out_dim)

    window_size = config["history_horizon"] + config["forecast_horizon"]
    return model, window_size, config["forecast_horizon"], out_std, build_dataset, forecast, modality_groups


def _load_tcnn(device, pinned_run=""):
    run_dir, ckpt_path = _find_checkpoint("TCNN", "tcnn_model", pinned_run)
    print(f"[ErrorPlot] TCNN checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    # Read output_dim from the checkpoint itself -- see _load_fld's comment.
    # TCNN has never had a velocity output (train_tcnn.py's own
    # OUTPUT_COLUMNS is still angle+moment=20), so this should always be 20,
    # but it's still read from the checkpoint rather than assumed.
    output_dim = ckpt["out_mean"].shape[0]
    model = TCNModel_Forecast(
        input_size=INPUT_DIM,
        output_size=output_dim,
        horizon=config["forecast_horizon"],
        num_channels=config["num_channels"],
        kernel_size=config["kernel_size"],
        dropout=config["dropout"],
    )
    _load_state_dict_flexible(model, ckpt["tcnn_state_dict"])
    model.to(device)
    model.eval()

    norm_stats = {
        'mean_in':  ckpt["in_mean"].cpu().numpy(),  'std_in':  ckpt["in_std"].cpu().numpy(),
        'mean_out': ckpt["out_mean"].cpu().numpy(), 'std_out': ckpt["out_std"].cpu().numpy(),
    }
    out_std = ckpt["out_std"].cpu().numpy()
    modality_groups = _modality_groups_for_output_dim(output_dim)

    def build_dataset(df):
        # task_dfs carries the FULL current OUTPUT_COLUMNS (30-wide, includes
        # velocity) -- TCNN was never trained to predict velocity, so slice
        # its own angle+moment columns out before constructing TCNNDataset
        # (whose norm_stats/output head are sized for 20, not 30).
        cols = INPUT_COLUMNS + (TCNN_OUTPUT_COLUMNS if output_dim == TCNN_OUTPUT_DIM else OUTPUT_COLUMNS)
        return TCNNDataset(
            df[cols], history_horizon=config["history_horizon"],
            forecast_horizon=config["forecast_horizon"],
            norm_stats=norm_stats, device=device, input_dim=INPUT_DIM,
        )

    def forecast(context, forecast_horizon):
        pred = model(context)                # (B, out_dim, K)
        return pred.permute(2, 0, 1)          # (K, B, out_dim)

    window_size = config["history_horizon"] + config["forecast_horizon"]
    return model, window_size, config["forecast_horizon"], out_std, build_dataset, forecast, modality_groups


def _load_fld_teacher_student(device, pinned_run=""):
    """Loads the STUDENT branch only (sensor_window -> shared decoder) --
    the teacher branch needs privileged biomechanics input this comparison
    can't give it at "inference" time, and the student is what would
    actually be deployed, making it the fair, apples-to-apples comparison
    against FLD-151/TCNN (both of which also only ever see sensor input).

    TS_SENSOR_COLUMNS (36, from DataLoader/teacher_student_data_loader.py)
    is a strict subset of this script's own INPUT_COLUMNS (48 wearable +
    subject_weight/height) -- no insole accel/gyro, no subject conditioning
    -- so task_dfs (built from INPUT_COLUMNS + OUTPUT_COLUMNS) already
    carries everything needed; build_dataset() just selects the smaller
    column subset. BIOMECH_COLUMNS (that loader's own output convention) is
    identical to this script's own 30-column OUTPUT_COLUMNS already, so no
    output-side remapping is needed.
    """
    run_dir, ckpt_path = _find_checkpoint("FLDTeacherStudent", "fld_teacher_student", pinned_run)
    print(f"[ErrorPlot] FLD-TeacherStudent checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = FLDTeacherStudent(
        sensor_dim=TS_SENSOR_DIM, biomech_dim=OUTPUT_DIM,
        history_horizon=config["history_horizon"], latent_dim=config["latent_dim"],
        device=device, dt=1.0 / 150.0,
        teacher_encoder_shape=config["teacher_encoder_shape"],
        student_encoder_shape=config["student_encoder_shape"],
        decoder_shape=config["decoder_shape"],
    )
    _load_state_dict_flexible(model, ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    ts_norm = ckpt["norm_stats"]   # keys: mean_sensor/std_sensor/mean_biomech/std_biomech, plain np.ndarray
    norm_stats = {
        'mean_in': ts_norm['mean_sensor'], 'std_in': ts_norm['std_sensor'],
        'mean_out': ts_norm['mean_biomech'], 'std_out': ts_norm['std_biomech'],
    }
    out_std = ts_norm['std_biomech']
    modality_groups = _modality_groups_for_output_dim(OUTPUT_DIM)   # 30 -- same convention as FLD-151

    def build_dataset(df):
        cols = TS_SENSOR_COLUMNS + OUTPUT_COLUMNS
        return FLDDataset(
            df[cols], history_horizon=config["history_horizon"],
            forecast_horizon=config["forecast_horizon"],
            feature_set='cross', norm_stats=norm_stats,
            device=device, input_dim=TS_SENSOR_DIM,
        )

    def forecast(context, forecast_horizon):
        with torch.no_grad():
            student_params = model.forward_student(context)
        return model.forecast(student_params, forecast_horizon)   # (K, B, out_dim)

    window_size = config["history_horizon"] + config["forecast_horizon"]
    return model, window_size, config["forecast_horizon"], out_std, build_dataset, forecast, modality_groups


# ---------------------------------------------------------------------------
# Evaluation — squared-error sums per trial, per forecast step. Both the
# pooled view and the per-trial view are derived from these same sums, so
# RMSE is always sqrt(sum of squared error / n) computed at the right
# granularity — never an average of already-square-rooted RMSEs.
# ---------------------------------------------------------------------------

def _sq_err_by_trial(build_dataset, forecast, forecast_horizon, task_dfs, split_index,
                      device, batch_size=256, split='val'):
    """Returns {trial_id: (sq_sum (K, out_dim) float64, n_windows)}. out_dim
    is inferred from the model's own prediction shape on the first batch,
    NOT a shared module-level constant -- FLD (30-wide) and TCNN (20-wide)
    are evaluated by the same function here and must each get their own
    width. Window starts come from split_index (Data/train_val_index_150_corrected.csv,
    see _load_split_index) -- a scattered set of local_start_idx per trial,
    not a single chronological cutoff. split: 'val' (default) or 'train' --
    which half of split_index to evaluate on."""
    per_trial = {}

    with torch.no_grad():
        for trial_id, df in task_dfs.items():
            dataset = build_dataset(df)
            if split_index is None:   # held-out subject: every window, no index CSV
                idx = range(len(dataset)) if split == 'val' else []
            else:
                idx = split_index.get(trial_id, {}).get(split, [])
            if not idx:
                print(f"[ErrorPlot]   {trial_id}: no {split} windows, skipping.")
                continue
            subset = Subset(dataset, idx)
            dl = DataLoader(subset, batch_size=batch_size, shuffle=False, drop_last=False)

            sq_sum    = None
            n_windows = 0
            for batch_x, batch_y in dl:
                batch_x = batch_x.to(device, non_blocking=True)   # (B, in_dim, H)
                batch_y = batch_y.to(device, non_blocking=True)   # (B, K, out_dim)
                pred = forecast(batch_x, forecast_horizon).float()   # (K, B, out_dim)
                tgt  = batch_y.permute(1, 0, 2).float()               # (K, B, out_dim)
                if sq_sum is None:
                    sq_sum = torch.zeros(forecast_horizon, pred.shape[-1], dtype=torch.float64, device=device)
                sq_sum += (pred - tgt).pow(2).sum(dim=1).double()     # (K, out_dim)
                n_windows += batch_x.shape[0]

            if sq_sum is None:
                print(f"[ErrorPlot]   {trial_id}: no {split} batches produced, skipping.")
                continue
            print(f"[ErrorPlot]   {trial_id}: {n_windows:,} {split} windows")
            per_trial[trial_id] = (sq_sum, n_windows)

    return per_trial


def _rmse_from_sq_err(sq_sum, n_windows, out_std):
    """(K, out_dim) squared-error sum + count -> (min(K, EVAL_STEPS), out_dim) RMSE, physical units."""
    rmse_norm = (sq_sum[:EVAL_STEPS] / max(n_windows, 1)).sqrt().cpu().numpy()   # (<=EVAL_STEPS, out_dim)
    return rmse_norm * out_std.reshape(1, -1)


def _rmse_pooled(per_trial, out_std):
    """Pool every trial's sq_sum/n together first, then sqrt once — matches
    the original all-tasks-combined RMSE curve. Shape (forecast_horizon,
    out_dim) is taken from the sq_sum tensors themselves, not a shared
    module-level constant -- see _sq_err_by_trial."""
    sample_sq_sum = next(iter(per_trial.values()))[0]
    sq_sum_total = torch.zeros_like(sample_sq_sum)
    n_total = 0
    for sq_sum, n in per_trial.values():
        sq_sum_total += sq_sum
        n_total += n
    return _rmse_from_sq_err(sq_sum_total, n_total, out_std), n_total


def _rmse_per_trial(per_trial, out_std):
    """{trial_id: (sq_sum, n)} -> {trial_id: (rmse (K, OUTPUT_DIM), n)}."""
    return {
        trial_id: (_rmse_from_sq_err(sq_sum, n, out_std), n)
        for trial_id, (sq_sum, n) in per_trial.items()
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_MODEL_COLORS = {
    "FLD-151": "tab:blue", "FLD-75": "tab:green", "TCNN": "tab:orange",
    "FLD-TeacherStudent(student)": "tab:purple",
}


def _style_for(label):
    """label is either a bare model name ("FLD") when PLOT_TRAIN_SET is off,
    or "FLD (val)"/"FLD (train)" when it's on. Same color per model either
    way; dashed line for train so a model's train/val curves read as a pair
    but stay visually distinguishable."""
    base = label.split(" (")[0]
    linestyle = "--" if label.endswith("(train)") else "-"
    return _MODEL_COLORS.get(base), linestyle


def _plot(results, save_path, title="RMSE vs. Forecast Horizon — pooled across all tasks"):
    """results: dict {model_label: (forecast_horizon, rmse_by_step (K, out_dim), modality_groups)}
    -- modality_groups is THIS model's own (see _modality_groups_for_output_dim):
    FLD's "moment" indices differ from TCNN's since FLD's rmse_by_step is
    30-wide (velocity in the middle) and TCNN's is 20-wide. Always plots the
    same two subplots (angle, moment) -- every model loadable here supports
    both, even though their column widths differ."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, group_key in zip(axes, ("angle", "moment")):
        for model_name, (forecast_horizon, rmse_by_step, modality_groups) in results.items():
            group = modality_groups[group_key]
            color, linestyle = _style_for(model_name)
            group_rmse = rmse_by_step[:, group["indices"]].mean(axis=1)   # (K,)
            steps = np.arange(1, forecast_horizon + 1)
            ax.plot(steps, group_rmse, label=model_name, color=color, ls=linestyle,
                    lw=1.8, marker='o', markersize=2.5)
        sample_group = next(iter(results.values()))[2][group_key]
        ax.set_xlabel("Forecast step (frames ahead, 150 Hz)")
        ax.set_xlim(0, EVAL_STEPS)
        ax.set_xticks(np.arange(0, EVAL_STEPS + 1, 1))   # resolution: one tick per forecast step
        ax.tick_params(axis='x', labelsize=7)
        ax.set_ylabel(f"RMSE ({sample_group['unit']})")
        ax.set_title(sample_group["label"])
        ax.grid(alpha=0.25)
        ax.legend()

    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f"[ErrorPlot] Saved figure -> {save_path}")


def _plot_per_trial(results_per_trial, task_labels, out_dir):
    """results_per_trial: dict {model_label: (modality_groups, {trial_id: (rmse (K, out_dim), n)})}

    Saves one figure per trial (angle + moment subplots, one curve per model)
    to out_dir — no plt.show(), since blocking on N trial windows one at a
    time isn't practical."""
    forecast_horizon = None
    for _modality_groups, per_trial in results_per_trial.values():
        for rmse, _ in per_trial.values():
            forecast_horizon = rmse.shape[0]
            break
        if forecast_horizon is not None:
            break

    trial_ids = sorted({tid for _mg, per_trial in results_per_trial.values() for tid in per_trial})
    saved_paths = []

    for trial_id in trial_ids:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, group_key in zip(axes, ("angle", "moment")):
            for model_name, (modality_groups, per_trial) in results_per_trial.items():
                if trial_id not in per_trial:
                    continue
                group = modality_groups[group_key]
                color, linestyle = _style_for(model_name)
                rmse, n = per_trial[trial_id]
                group_rmse = rmse[:, group["indices"]].mean(axis=1)   # (K,)
                steps = np.arange(1, forecast_horizon + 1)
                ax.plot(steps, group_rmse, label=f"{model_name} (n={n})",
                        color=color, ls=linestyle, lw=1.8, marker='o', markersize=2.5)
            ax.set_xlabel("Forecast step (frames ahead, 150 Hz)")
            ax.set_xlim(0, EVAL_STEPS)
            ax.set_xticks(np.arange(0, EVAL_STEPS + 1, 1))   # resolution: one tick per forecast step
            ax.tick_params(axis='x', labelsize=7)
            ax.set_ylabel(f"RMSE ({group['unit']})")
            ax.set_title(group["label"])
            ax.grid(alpha=0.25)
            ax.legend()

        label = task_labels.get(trial_id, trial_id)
        fig.suptitle(f"RMSE vs. Forecast Horizon — {trial_id} ({label})",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()

        save_path = os.path.join(out_dir, f"error_across_time_{trial_id}.png")
        # fig.savefig(save_path, dpi=150)
        # plt.close(fig)
        # saved_paths.append(save_path)

    for path in saved_paths:
        print(f"[ErrorPlot] Saved figure -> {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ErrorPlot] Using device: {device}")

    all_df = _load_all_data()
    discovered = _discover_tasks(all_df)
    task_labels = {tid: info['label'] for tid, info in discovered.items()}
    task_dfs = {
        trial_id: g[INPUT_COLUMNS + OUTPUT_COLUMNS].reset_index(drop=True)
        for trial_id, g in all_df.groupby('trial_id', sort=False)
    }
    print(f"[ErrorPlot] Discovered {len(task_dfs)} tasks in {ALL_DATA_PATH}")

    results = {}          # pooled:    {model_label: (K, rmse (K, out_dim), modality_groups)}
    results_per_trial = {}  # per-trial: {model_label: (modality_groups, {trial_id: (rmse (K, out_dim), n)})}
    model_loaders = (
        # ("FLD-151", FLD151_RUN, _load_fld,  _sq_err_by_trial),   # disabled -- comparing TCNN vs. FLD-TeacherStudent only
        # ("FLD-75",  FLD75_RUN,  _load_fld,  _sq_err_by_trial),   # disabled for now -- see module docstring
        ("TCNN",    TCNN_RUN,   _load_tcnn, _sq_err_by_trial),
        ("FLD-TeacherStudent(student)", FLD_TS_RUN, _load_fld_teacher_student, _sq_err_by_trial),
    )
    plot_train_set = PLOT_TRAIN_SET and not EVAL_HELD_OUT
    splits_to_eval = ('val', 'train') if plot_train_set else ('val',)

    for name, pinned_run, loader, eval_fn in model_loaders:
        pin_str = f"pinned: {pinned_run}" if pinned_run else "auto: latest"
        print(f"[ErrorPlot] Loading {name} ({pin_str})...")
        try:
            model, window_size, K, out_std, build_dataset, forecast, modality_groups = loader(device, pinned_run)
        except FileNotFoundError as e:
            print(f"[ErrorPlot] Skipping {name}: {e}")
            continue

        split_index = None if EVAL_HELD_OUT else _load_split_index(window_size)

        for split in splits_to_eval:
            # bare name when only evaluating val (today's behaviour, unchanged
            # labels); "<name> (val)"/"<name> (train)" once both are on.
            label = f"{name} ({split})" if plot_train_set else (f"{name} (held-out)" if EVAL_HELD_OUT else name)

            print(f"[ErrorPlot] Evaluating {name} on {split} windows...")
            per_trial_sq = eval_fn(build_dataset, forecast, K, task_dfs, split_index, device, split=split)
            if not per_trial_sq:
                print(f"[ErrorPlot] Skipping {label}: no {split} windows in any task.")
                continue

            rmse, n = _rmse_pooled(per_trial_sq, out_std)
            print(f"[ErrorPlot] {label}: {n:,} total {split} windows")
            results[label] = (rmse.shape[0], rmse, modality_groups)   # steps actually plotted (<= EVAL_STEPS)
            results_per_trial[label] = (modality_groups, _rmse_per_trial(per_trial_sq, out_std))

    if not results:
        raise FileNotFoundError(
            "No FLD-151 or TCNN checkpoints could be resolved — nothing to plot. "
            "Check FLD151_RUN/TCNN_RUN at the top of this script, or train a model first."
        )

    out_dir = os.path.join(RUNS_DIR, f"ErrorAcrossTime_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    if SEPARATE_TRIALS:
        _plot_per_trial(results_per_trial, task_labels, out_dir)
    else:
        _plot(results, os.path.join(out_dir, "error_across_time.png"))

    # Console summary at a few representative horizons (always pooled, regardless of SEPARATE_TRIALS)
    print("\n[ErrorPlot] RMSE summary (physical units, averaged within modality group):")
    name_w = max((len(n) for n in results), default=5)
    for model_name, (K, rmse, modality_groups) in results.items():
        checkpoints = [k for k in (1, 10, 25, K) if k <= K]
        for group_key, group in modality_groups.items():
            row = "  ".join(
                f"k{k}={rmse[k-1, group['indices']].mean():.3f}{group['unit']}"
                for k in checkpoints
            )
            print(f"  {model_name:<{name_w}s} {group_key:7s}: {row}")

    plt.show()

if __name__ == "__main__":
    main()
