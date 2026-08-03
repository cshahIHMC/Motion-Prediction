"""
train_tcnn.py

Training script for a baseline Temporal Convolutional Network (TCN) direct
motion predictor, trained jointly across multiple recording sessions
("tasks"/trials). Mirrors train_fld.py's data pipeline (task discovery from
Data/all_data.csv, train/val window-start index from
Data/train_val_index.csv, pooled global normalisation, same input/output
columns) but swaps the FLD phase-based model for TCNModel_Forecast, a plain
causal-dilated-conv backbone with a linear head that predicts the whole
forecast horizon in one forward pass.

Deliberately independent of train_fld.py and DataLoader/data_loader.py: this
script uses its own DataLoader/tcnn_data_loader.py (TCNNDataset) and
duplicates the small amount of task-loading/column-definition code it needs,
so edits to one script's data pipeline can never break the other's.

Input  = body acc (5x3=15) + body gyro (5x3=15) + insole acc (2x3=6) + insole gyro (2x3=6)
       + insole force/COP (L/R x 3 = 6) + subject weight + subject height  ->  50 features total
Output = 10 joint angles (deg) + 10 joint moments (N.m/kg, each divided by
         that row's own subject_weight -- see _load_all_data) = 20 features

At 150 Hz, each training sample is a clean, disjoint split (history_horizon
and forecast_horizon are config values -- H=150/K=50 shown below as an
example, current config may differ, see main()):
  context window : rows [s, s+H)     -- H frames of context, fed to the TCN
  future target   : rows [s+H, s+H+K) -- K genuinely-future frames, predicted
                     in one shot by the linear head (see TCNModel_Forecast).

Each trial's rows are its own contiguous recording. Context/future pairs are
built per-trial and never cross a trial boundary. Train/val window starts
come from Data/train_val_index.csv (built by build_train_val_index.py):
each trial's rows are split ~85/15, with val spread across a few segments
through the trial (not just the tail) and a window_size gap enforced around
every train/val boundary so no window pair leaks across the split. A global
normalisation is fit on the pooled training rows across all tasks (excluding
any row a validation window reads -- see _norm_stats_mask) so every task
shares one input/output scale. Unlike train_fld.py, no latent-space/
phase-manifold diagnostics are produced here (there's no latent space to
inspect) -- only the loss curve and the per-task future-prediction
trajectory plots are saved, drawn from each task's largest validation
segment.

Run:
    python3 combine_data.py            # once, or after adding trials/subjects
    python3 build_train_val_index.py   # once, or after the above changes
    python3 train_tcnn.py
"""

import os
import sys
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset, ConcatDataset
import pandas as pd
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from DataLoader.tcnn_data_loader import TCNNDataset
from Models.TCNN import TCNModel_Forecast

# ---------------------------------------------------------------------------
# Paths & column definitions
# ---------------------------------------------------------------------------

DATA_DIR      = os.path.join(os.path.dirname(__file__), "Data")
ALL_DATA_PATH = os.path.join(DATA_DIR, "all_data.csv")           # built by combine_data.py
INDEX_PATH    = os.path.join(DATA_DIR, "train_val_index.csv")    # built by build_train_val_index.py

# Input features — 48 wearable sensor channels
COLUMNS = [
    # Body segment accelerometers — 5 locations × 3 axes = 15 features
    'pelvis_acc_x',  'pelvis_acc_y',  'pelvis_acc_z',
    'thigh_r_acc_x', 'thigh_r_acc_y', 'thigh_r_acc_z',
    'shank_r_acc_x', 'shank_r_acc_y', 'shank_r_acc_z',
    'thigh_l_acc_x', 'thigh_l_acc_y', 'thigh_l_acc_z',
    'shank_l_acc_x', 'shank_l_acc_y', 'shank_l_acc_z',
    # Body segment gyroscopes — 5 locations × 3 axes = 15 features
    'pelvis_gyro_x',  'pelvis_gyro_y',  'pelvis_gyro_z',
    'thigh_r_gyro_x', 'thigh_r_gyro_y', 'thigh_r_gyro_z',
    'shank_r_gyro_x', 'shank_r_gyro_y', 'shank_r_gyro_z',
    'thigh_l_gyro_x', 'thigh_l_gyro_y', 'thigh_l_gyro_z',
    'shank_l_gyro_x', 'shank_l_gyro_y', 'shank_l_gyro_z',
    # Insole accelerometers — 2 × 3 = 6 features
    'L_insole_accel_x', 'L_insole_accel_y', 'L_insole_accel_z',
    'R_insole_accel_x', 'R_insole_accel_y', 'R_insole_accel_z',
    # Insole gyroscopes — 2 × 3 = 6 features
    'L_insole_gyro_x', 'L_insole_gyro_y', 'L_insole_gyro_z',
    'R_insole_gyro_x', 'R_insole_gyro_y', 'R_insole_gyro_z',
    # Insole force + COP — 2 × 3 = 6 features
    'L_insole_force', 'L_insole_COPx', 'L_insole_COPz',
    'R_insole_force', 'R_insole_COPx', 'R_insole_COPz',
    # Total: 48 features
]

# Subject-specific constants — same value repeated on every row of a given
# trial's CSV. Now that Data/all_data.csv pools multiple subjects (S01, S02,
# ...) with genuinely different weight/height, these carry real signal —
# they let the TCN condition its output on who's wearing the sensors.
SUBJECT_COLUMNS = ['subject_weight', 'subject_height']

# What actually gets fed to the TCN: wearable sensors + subject constants
INPUT_COLUMNS = COLUMNS + SUBJECT_COLUMNS

# Output features — 10 joint angles (°) + 10 joint moments (N·m/kg) = 20 features
OUTPUT_COLUMNS = [
    # Joint angles — degrees
    'hip_flexion_r',    'hip_flexion_l',
    'hip_adduction_r',  'hip_adduction_l',
    'hip_rotation_r',   'hip_rotation_l',
    'knee_angle_r',     'knee_angle_l',
    'ankle_angle_r',    'ankle_angle_l',
    # Joint moments — N·m/kg (raw N·m divided by subject_weight in _load_all_data,
    # so torque is reported per unit body mass and comparable across subjects)
    'hip_flexion_r_moment',   'hip_flexion_l_moment',
    'hip_adduction_r_moment', 'hip_adduction_l_moment',
    'hip_rotation_r_moment',  'hip_rotation_l_moment',
    'knee_angle_r_moment',    'knee_angle_l_moment',
    'ankle_angle_r_moment',   'ankle_angle_l_moment',
    # Total: 20 features
]
MOMENT_COLUMNS = OUTPUT_COLUMNS[10:]   # the 10 *_moment columns -- normalised by subject weight

INPUT_DIM  = len(INPUT_COLUMNS)
OUTPUT_DIM = len(OUTPUT_COLUMNS)

# Output modality groups — for per-group RMSE logging and per-modality loss
# weighting (config["weight_angle"] / config["weight_moment"]).
MODALITY_GROUPS = {
    "angle":  {"indices": list(range(0, 10)),  "unit": "°",      "label": "Joint Angle RMSE (°)"},
    "moment": {"indices": list(range(10, 20)), "unit": "N·m/kg", "label": "Joint Moment RMSE (N·m/kg)"},
}


# ---------------------------------------------------------------------------
# Data loading — Data/all_data.csv (combine_data.py) + Data/train_val_index.csv
# (build_train_val_index.py). Both must exist; run those scripts first if
# either is missing.
# ---------------------------------------------------------------------------

def _load_all_data():
    """Read Data/all_data.csv and normalise joint moments to N·m/kg by
    dividing each row's moment columns by that row's own subject_weight (kg)
    -- so torque is reported per unit body mass instead of raw N·m, and is
    comparable across subjects of different weight."""
    if not os.path.exists(ALL_DATA_PATH):
        raise FileNotFoundError(f"{ALL_DATA_PATH} not found -- run combine_data.py first.")
    usecols = list(dict.fromkeys(['trial_id', 'task'] + INPUT_COLUMNS + OUTPUT_COLUMNS))
    df = pd.read_csv(ALL_DATA_PATH, usecols=usecols)
    df[MOMENT_COLUMNS] = df[MOMENT_COLUMNS].div(df['subject_weight'], axis=0)
    return df


def _discover_tasks(all_df):
    """trial_id -> {'label', 'weight', 'height'}, read from each trial's
    first row (all_df must have trial_id/task/subject_weight/subject_height)."""
    tasks = {}
    for trial_id, g in all_df.groupby('trial_id', sort=False):
        first = g.iloc[0]
        tasks[trial_id] = {
            'label':  str(first['task']),
            'weight': float(first['subject_weight']),
            'height': float(first['subject_height']),
        }
    return tasks


def _load_split_index(window_size):
    """trial_id -> {'train': [local_start_idx...], 'val': [local_start_idx...]},
    from Data/train_val_index.csv (build_train_val_index.py). Verifies the
    index was built for a window at least as large as window_size -- a CSV
    built for a bigger window is safe to reuse (its gaps/trial-boundary
    margins are only more conservative), but one built for a smaller window
    could let a window here spill past a trial boundary or the leakage gap."""
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"{INDEX_PATH} not found -- run build_train_val_index.py first.")
    idx_df = pd.read_csv(INDEX_PATH, usecols=['trial_id', 'split', 'local_start_idx', 'window_size'])
    index_window_size = int(idx_df['window_size'].min())
    if index_window_size < window_size:
        raise ValueError(
            f"{INDEX_PATH} was built for window_size={index_window_size}, smaller than this "
            f"script's window_size={window_size} -- rerun build_train_val_index.py with "
            f"HISTORY_HORIZON/FORECAST_HORIZON matching (or exceeding) this config first."
        )
    splits = {}
    for trial_id, g in idx_df.groupby('trial_id', sort=False):
        splits[trial_id] = {
            'train': g.loc[g['split'] == 'train', 'local_start_idx'].tolist(),
            'val':   g.loc[g['split'] == 'val',   'local_start_idx'].tolist(),
        }
    return splits


def _val_runs(val_local_starts, window_size):
    """Group a trial's val local_start_idx values into contiguous stride-1
    runs -- one run per val segment placed by build_train_val_index.py.
    Returns list of (run_start, raw_row_end) tuples, raw_row_end being one
    past the last raw row any window in that run reads."""
    if not val_local_starts:
        return []
    starts = np.sort(np.asarray(val_local_starts))
    breaks = np.flatnonzero(np.diff(starts) > 1)
    run_starts = np.r_[starts[0], starts[breaks + 1]]
    run_ends   = np.r_[starts[breaks], starts[-1]]
    return [(int(rs), int(re) + window_size) for rs, re in zip(run_starts, run_ends)]


def _norm_stats_mask(trial_length, val_local_starts, window_size):
    """Boolean mask over a trial's raw rows: True for rows usable when
    fitting normalisation statistics. Excludes exactly the rows any
    validation window reads (context or future), so val never leaks into the
    fitted train normalisation -- see _val_runs."""
    mask = np.ones(trial_length, dtype=bool)
    for rs, re in _val_runs(val_local_starts, window_size):
        mask[rs:re] = False
    return mask


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_forecast_loss(pred_all, target_all, gamma=1.0, feature_weights=None):
    """
    pred_all, target_all : (K, B, features) normalised tensors — one genuinely
    future point per forecast step.

    - Loss computed in NORMALISED space — no denormalisation, so mixed
      physical units (deg, N·m) don't inflate one modality.
    - AVERAGED over K steps (not summed), so magnitude is comparable across
      datasets with different forecast horizons.
    - Optional gamma discounting: step i is weighted by gamma**i so
      near-term accuracy is emphasised more (default 1.0 = off).
    - Optional feature_weights (features,): multiply squared error
      element-wise to up/down-weight modality groups.
    """
    K      = pred_all.shape[0]
    total  = torch.zeros(1, device=pred_all.device)
    weight = 1.0

    for i in range(K):
        sq_err = (pred_all[i] - target_all[i]).pow(2)   # (B, features)
        if feature_weights is not None:
            sq_err = sq_err * feature_weights.view(1, -1)
        total  = total + sq_err.mean() * (gamma ** i)
        weight = weight + (gamma ** i)

    return total / weight


def per_feature_rmse(pred_all, target_all, std):
    """RMSE in original physical units per feature, averaged over K and B."""
    sq_err    = (pred_all - target_all).pow(2)       # (K, B, features)
    mse_norm  = sq_err.mean(dim=(0, 1))               # (features,)
    rmse_norm = mse_norm.sqrt()
    return rmse_norm * std.clamp(min=1e-6).to(pred_all.device)


def rmse_by_horizon(pred_all, target_all):
    """RMSE in normalised space per forecast step. Returns (K,) tensor."""
    sq_err = (pred_all - target_all).pow(2)   # (K, B, features)
    mse_k  = sq_err.mean(dim=(1, 2))           # (K,)
    return mse_k.sqrt()


def _forecast(model, context, forecast_horizon):
    """Run the TCN model on a context window and return genuinely-future
    predictions in the same (K, B, output_dim) layout used by the loss and
    plotting code.

    model(context) returns (B, output_dim, K) in one shot (TCNModel_Forecast
    predicts the whole horizon from a single forward pass — no autoregressive
    unrolling needed).
    """
    pred = model(context)                       # (B, output_dim, K)
    future_pred = pred.permute(2, 0, 1)          # (K, B, output_dim)
    return future_pred


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: dict):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TCNN] Using device: {device}")
    if device.type == "cuda":
        print(f"[TCNN] GPU: {torch.cuda.get_device_name(device)}")

    use_amp   = config.get("use_amp", True) and device.type == "cuda"
    amp_dtype = torch.bfloat16

    # The normalised dataset is small — keep it resident on the GPU so
    # __getitem__ slices straight out of device memory (see train_fld.py for
    # the same pattern).
    data_device = device if device.type == "cuda" else None
    num_workers = 0 if data_device is not None else config["num_workers"]

    log_dir   = os.path.join("runs", f"TCNN_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    plots_dir = os.path.join(log_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    history_horizon  = config["history_horizon"]
    forecast_horizon = config["forecast_horizon"]
    window_size = history_horizon + forecast_horizon

    # ---- Discover every task from Data/all_data.csv, load the pre-built ------
    # ---- train/val window-start index -----------------------------------------
    all_df      = _load_all_data()
    discovered  = _discover_tasks(all_df)
    task_labels = {tid: info['label'] for tid, info in discovered.items()}
    split_index = _load_split_index(window_size)
    print(f"[TCNN] Discovered {len(task_labels)} tasks in {ALL_DATA_PATH}: {task_labels}")

    task_dfs = {}
    train_raw_in, train_raw_out = [], []
    for trial_id, g in all_df.groupby('trial_id', sort=False):
        df = g[INPUT_COLUMNS + OUTPUT_COLUMNS].reset_index(drop=True)
        task_dfs[trial_id] = df
        if trial_id not in split_index:
            raise ValueError(f"{trial_id}: not present in {INDEX_PATH} -- rerun build_train_val_index.py")
        val_local = split_index[trial_id]['val']
        mask = _norm_stats_mask(len(df), val_local, window_size)
        train_raw_in.append(df.loc[mask, INPUT_COLUMNS].values.astype('float32'))
        train_raw_out.append(df.loc[mask, OUTPUT_COLUMNS].values.astype('float32'))
        info = discovered[trial_id]
        n_train_win = len(split_index[trial_id]['train'])
        n_val_win   = len(split_index[trial_id]['val'])
        print(f"[TCNN] {trial_id} ({task_labels[trial_id]}): {len(df):,} rows "
              f"({len(df)/150:.1f}s @150Hz)  train_windows={n_train_win:,}  val_windows={n_val_win:,}  "
              f"subject={info['weight']:.1f}kg/{info['height']:.0f}cm")

    # Global normalisation fit on TRAIN rows only, pooled across all tasks.
    raw_in_all  = np.concatenate(train_raw_in,  axis=0)
    raw_out_all = np.concatenate(train_raw_out, axis=0)
    mean_in,  std_in  = raw_in_all.mean(axis=0),  raw_in_all.std(axis=0)
    mean_out, std_out = raw_out_all.mean(axis=0), raw_out_all.std(axis=0)
    std_in[std_in == 0]   = 1.0
    std_out[std_out == 0] = 1.0
    norm_stats  = {'mean_in': mean_in, 'std_in': std_in, 'mean_out': mean_out, 'std_out': std_out}
    out_std_cpu = torch.tensor(std_out, dtype=torch.float32)

    # ---- Per-task TCNNDataset — windows are built within one file only -------
    task_info = {}
    train_subsets, val_subsets = [], []
    for trial_id, label in task_labels.items():
        df = task_dfs[trial_id]   # already ordered INPUT_COLUMNS + OUTPUT_COLUMNS
        dataset = TCNNDataset(
            df,
            history_horizon=history_horizon,
            forecast_horizon=forecast_horizon,
            norm_stats=norm_stats,
            device=data_device,
            input_dim=INPUT_DIM,
        )
        train_idx = split_index[trial_id]['train']
        val_idx   = split_index[trial_id]['val']
        if not train_idx or not val_idx:
            raise ValueError(
                f"{trial_id}: missing train or val windows in {INDEX_PATH} "
                f"(train={len(train_idx)}, val={len(val_idx)})"
            )

        train_subset = Subset(dataset, train_idx)
        val_subset   = Subset(dataset, val_idx)
        train_subsets.append(train_subset)
        val_subsets.append(val_subset)

        task_info[trial_id] = {
            "label": label,
            "dataset": dataset,
            # scattered val window starts, replacing the old single val_start
            # (val is no longer one contiguous tail region) -- long-trajectory
            # plots below use the largest contiguous run of these.
            "val_runs": _val_runs(val_idx, window_size),
        }
        print(f"[TCNN] {trial_id} windows -> train={len(train_subset):,}  val={len(val_subset):,}")

    task_ids = list(task_labels.keys())

    train_dataset = ConcatDataset(train_subsets)
    val_dataset   = ConcatDataset(val_subsets)
    print(f"[TCNN] Combined: {len(train_dataset):,} train windows, {len(val_dataset):,} val windows "
          f"across {len(task_ids)} tasks")

    loader_kwargs = dict(num_workers=num_workers, pin_memory=(data_device is None))
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_dl = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                          drop_last=True, **loader_kwargs)
    val_dl = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False,
                        drop_last=False, **loader_kwargs)

    # ---- Model ---------------------------------------------------------------
    model = TCNModel_Forecast(
        input_size=INPUT_DIM,
        output_size=OUTPUT_DIM,
        horizon=forecast_horizon,
        num_channels=config["num_channels"],
        kernel_size=config["kernel_size"],
        dropout=config["dropout"],
    )
    model.to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=config["lr"],
                           weight_decay=config["weight_decay"])

    if config.get("compile_model", False):
        try:
            model = torch.compile(model)
            print("[TCNN] torch.compile enabled.")
        except Exception as e:
            print(f"[TCNN] torch.compile failed, continuing eager: {e}")

    noise_level = config["noise_level"]
    gamma       = config.get("gamma", 1.0)
    n_features  = OUTPUT_DIM

    # Build per-feature weight tensor from config modality weights
    fw = torch.ones(n_features, device=device)
    for modality, group in MODALITY_GROUPS.items():
        w = config.get(f"weight_{modality}", 1.0)   # e.g. "weight_angle" / "weight_moment"
        for idx in group["indices"]:
            if idx < n_features:
                fw[idx] = w
    feature_weights = fw

    fig_loss, ax_loss = plt.subplots(figsize=(8, 4))
    # Long trajectory — split into angles and moments, one figure reused per task
    _n_angles  = 10
    _n_moments = 10
    fig_long_angle,  ax_long_angle  = plt.subplots(_n_angles,  1, figsize=(30, 2.0 * _n_angles),  sharex=True)
    fig_long_moment, ax_long_moment = plt.subplots(_n_moments, 1, figsize=(30, 2.0 * _n_moments), sharex=True)

    train_losses, val_losses = [], []
    diag_k = [9, 24, forecast_horizon - 1]   # horizon steps to log (0-indexed)

    # ---- Training loop -------------------------------------------------------
    train_start = datetime.now()
    print(f"[TCNN] Training started at {train_start.strftime('%Y-%m-%d %H:%M:%S')}.")
    for epoch in range(config["epochs"]):
        epoch_start = datetime.now()
        model.train()
        running_loss = 0.0

        for batch_x, batch_y in train_dl:
            batch_x = batch_x.to(device, non_blocking=True)   # (B, input_dim, H) — context
            batch_y = batch_y.to(device, non_blocking=True)   # (B, K, output_dim) — future

            if noise_level > 0.0:
                batch_x = batch_x + torch.randn_like(batch_x) * noise_level

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                future_pred = _forecast(model, batch_x, forecast_horizon)
                tgt_all = batch_y.permute(1, 0, 2)   # (K, B, output_dim)
                loss = compute_forecast_loss(future_pred, tgt_all, gamma, feature_weights)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss_noisy = running_loss / len(train_dl)

        # ---- Clean train loss (eval-mode, no noise, first 8 batches) ---------
        model.eval()
        clean_running = 0.0
        n_clean = 0
        with torch.no_grad():
            for batch_x, batch_y in train_dl:
                if n_clean >= 8:
                    break
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    future_pred = _forecast(model, batch_x, forecast_horizon)
                    tgt_all = batch_y.permute(1, 0, 2)
                    clean_running += compute_forecast_loss(
                        future_pred, tgt_all, gamma, feature_weights
                    ).item()
                n_clean += 1
        train_loss = clean_running / max(n_clean, 1)

        # ---- Validation + diagnostics (combined across all tasks) ------------
        model.eval()
        val_running = 0.0
        all_pred, all_tgt = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_dl:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    future_pred = _forecast(model, batch_x, forecast_horizon)
                    tgt_all = batch_y.permute(1, 0, 2)
                    val_running += compute_forecast_loss(
                        future_pred, tgt_all, gamma, feature_weights
                    ).item()
                if len(all_pred) < 8:
                    all_pred.append(future_pred.float().cpu())
                    all_tgt.append(tgt_all.float().cpu())

        val_loss = val_running / len(val_dl)

        # --- Diagnostic computations on accumulated val batches ---------------
        pred_cat = torch.cat(all_pred, dim=1)   # (K, B_acc, out_dim)
        tgt_cat  = torch.cat(all_tgt,  dim=1)   # (K, B_acc, out_dim)

        horiz_rmse  = rmse_by_horizon(pred_cat, tgt_cat)   # (K,)
        diag_k_vals = [horiz_rmse[k].item() for k in diag_k]

        feat_rmse = per_feature_rmse(pred_cat, tgt_cat, out_std_cpu)  # (out_features,)
        mod_rmse  = {}
        for modality, group in MODALITY_GROUPS.items():
            valid = [i for i in group["indices"] if i < n_features]
            mod_rmse[modality] = feat_rmse[valid].mean().item() if valid else 0.0

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        hz_str = "  ".join(f"k{k+1}={v:.4f}" for k, v in zip(diag_k, diag_k_vals))
        print(f"[TCNN] Epoch [{epoch + 1:>4}/{config['epochs']}]  "
              f"train(clean)={train_loss:.5f}  train(noisy)={train_loss_noisy:.5f}  val={val_loss:.5f}  "
              f"| horizon RMSE: {hz_str}")

        mod_str = "  ".join(f"{k}={v:.3f}" for k, v in mod_rmse.items())
        print(f"          modality RMSE (orig units): {mod_str}")

        # All plots fire once at the end of training only
        if epoch == config["epochs"] - 1:
            with torch.no_grad():
                model.eval()
                _plot_loss_curve(fig_loss, ax_loss, train_losses, val_losses,
                                 save_path=os.path.join(log_dir, "loss_curve.png"))

                # --- Per-task future-prediction trajectory plots -----------------
                # Val is now spread across a few segments per trial (see
                # build_train_val_index.py), not one contiguous tail -- this
                # plot needs a single contiguous stretch, so use the largest
                # val run and bound reads to its own extent (max_frame).
                for trial_id in task_ids:
                    info        = task_info[trial_id]
                    task_ds     = info["dataset"]
                    run_start, run_end = max(info["val_runs"], key=lambda r: r[1] - r[0])
                    label_safe  = info["label"].replace("/", "-").replace(" ", "_")
                    prefix      = f"{trial_id}_{label_safe}"

                    _plot_long_trajectory(
                        model, task_ds, device,
                        history_horizon, fig_long_angle, ax_long_angle,
                        n_blocks=225, forecast_horizon=forecast_horizon, block_step=20,
                        sample_rate=150.0, epoch=epoch,
                        start_frame=run_start, max_frame=run_end, feature_slice=slice(0, 10),
                        task_label=f"{trial_id} ({info['label']})",
                        save_path=os.path.join(plots_dir, f"{prefix}_future_pred_angles.png"),
                    )
                    _plot_long_trajectory(
                        model, task_ds, device,
                        history_horizon, fig_long_moment, ax_long_moment,
                        n_blocks=225, forecast_horizon=forecast_horizon, block_step=20,
                        sample_rate=150.0, epoch=epoch,
                        start_frame=run_start, max_frame=run_end, feature_slice=slice(10, 20),
                        task_label=f"{trial_id} ({info['label']})",
                        save_path=os.path.join(plots_dir, f"{prefix}_future_pred_moments.png"),
                    )
                model.train()

        # Checkpoint
        if epoch % config["save_every"] == 0:
            _save(model, optimizer, epoch, config, norm_stats, log_dir)

        epoch_end = datetime.now()
        print(f"[TCNN] Epoch {epoch + 1} finished at {epoch_end.strftime('%Y-%m-%d %H:%M:%S')}  "
              f"(epoch took {(epoch_end - epoch_start).total_seconds():.1f}s, "
              f"elapsed {(epoch_end - train_start).total_seconds() / 60:.1f} min total)")

    _save(model, optimizer, config["epochs"], config, norm_stats, log_dir)
    print(f"[TCNN] Training finished. Checkpoints + plots in: {log_dir}")
    plt.ioff()
    plt.show()  # keep all figures open after training completes
    return log_dir


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_loss_curve(fig, ax, train_losses, val_losses, save_path=None):
    ax.cla()
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train")
    ax.plot(epochs, val_losses,   label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("TCNN Training Loss")
    ax.legend()
    ax.grid(True)
    _show(fig, save_path)


def _show(fig, save_path=None):
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    fig.canvas.draw()
    plt.pause(0.05)


# ---------------------------------------------------------------------------
# Long trajectory — GT vs prediction stitched over many windows
# ---------------------------------------------------------------------------

def _plot_long_trajectory(model, dataset, device,
                           history_horizon, fig, axes,
                           n_blocks=200, forecast_horizon=50,
                           sample_rate=150.0, epoch=None,
                           batch_size=64, start_frame=0, max_frame=None,
                           feature_slice=None, task_label=None, save_path=None,
                           block_step=None):
    """
    Feed a fresh history_horizon-frame context to the model every `block_step`
    frames, starting at start_frame (pass a validation region's start to plot
    validation data only), and plot the K genuinely-future frames that follow
    each context. Everything is drawn on a real-time x-axis: the ground truth
    is one continuous curve, and each block's prediction is layered on top at
    its true time position.

    max_frame: exclusive upper bound on how far blocks may read (defaults to
    the whole dataset). Since validation windows are now scattered across a
    few segments per trial rather than one contiguous tail, pass the end of
    the specific validation run being plotted so this never wanders past it
    into train rows.

    block_step defaults to forecast_horizon, which reproduces the original
    behaviour: predicted segments tile the timeline with no overlap. Pass a
    smaller block_step (e.g. 20) to sample contexts more densely — predicted
    segments will then overlap in real time, so you can see several blocks'
    predictions for the same stretch layered on top of each other.

    feature_slice: slice object selecting which output features to plot,
                   e.g. slice(0,10) for angles, slice(10,20) for moments.
    Blue = ground truth (drawn once).  Red dashed = one prediction per block.
    """
    if block_step is None:
        block_step = forecast_horizon

    data_in  = dataset.data_in           # (N, in_dim) normalised — fed to the TCN
    data_out = dataset.data_out          # (N, out_dim) normalised — GT
    std_np   = dataset.out_std_tensor.detach().cpu().numpy()
    mean_np  = dataset.out_mean_tensor.detach().cpu().numpy()
    N        = data_in.shape[0] if max_frame is None else min(max_frame, data_in.shape[0])
    H        = history_horizon
    K        = forecast_horizon

    max_blocks = (N - start_frame - H - K) // block_step + 1
    n_blocks   = min(n_blocks, max(max_blocks, 0))
    if n_blocks <= 0:
        print(f"[TCNN] _plot_long_trajectory: not enough rows for even one block, skipping.")
        return

    gt_start = start_frame + H
    gt_end   = min(start_frame + H + (n_blocks - 1) * block_step + K, N)
    gt_full  = data_out[gt_start:gt_end].detach().cpu().numpy() * std_np + mean_np   # (T, out_dim)
    t_gt     = np.arange(gt_start, gt_end) / sample_rate

    pred_blocks = []   # list of (t_block, pred_block_orig) — one per block
    model.eval()
    with torch.no_grad():
        for b_start in range(0, n_blocks, batch_size):
            b_end      = min(b_start + batch_size, n_blocks)
            ctx_starts = [start_frame + i * block_step for i in range(b_start, b_end)]
            wins = torch.stack(
                [data_in[s : s + H].T for s in ctx_starts]
            ).to(device)                                      # (B, in_dim, H)

            future_pred = _forecast(model, wins, K)            # (K, B, D) — genuinely future
            pred_block = future_pred.permute(1, 0, 2).cpu().numpy()   # (B, K, D)

            for bi, s in enumerate(ctx_starts):
                pred_orig = pred_block[bi] * std_np + mean_np           # (K, out_dim)
                t_block   = np.arange(s + H, s + H + K) / sample_rate
                pred_blocks.append((t_block, pred_orig))

    fs = feature_slice if feature_slice is not None else slice(None)
    gt_plot   = gt_full[:, fs]
    col_names = OUTPUT_COLUMNS[fs] if feature_slice is not None else OUTPUT_COLUMNS
    all_units = ([MODALITY_GROUPS["angle"]["unit"]] * 10
                + [MODALITY_GROUPS["moment"]["unit"]] * 10)
    units_plot = (all_units[fs] if isinstance(fs, slice)
                  else [all_units[i] for i in fs])
    group_label = ("Joint Angles (°)" if feature_slice == slice(0, 10)
                   else f"Joint Moments ({MODALITY_GROUPS['moment']['unit']})" if feature_slice == slice(10, 20)
                   else "Output Features")

    overlapping = block_step < K
    pred_alpha  = 0.45 if overlapping else 0.85

    for fi, ax in enumerate(axes):
        ax.cla()
        ax.plot(t_gt, gt_plot[:, fi], color="steelblue", lw=1.1, alpha=0.95, label="GT", zorder=10)
        for bi, (t_block, pred_orig) in enumerate(pred_blocks):
            ax.plot(t_block, pred_orig[:, fs][:, fi],
                    color="tomato", lw=1.4, alpha=pred_alpha, ls="--",
                    label=("Pred" if bi == 0 else None))
        ax.set_ylabel(f"{col_names[fi][:20]}\n({units_plot[fi]})",
                      fontsize=7, rotation=0, labelpad=95, va="center")
        ax.grid(alpha=0.15)
        ax.tick_params(labelsize=6)
        if fi == 0:
            ax.legend(fontsize=7, loc="upper right", ncol=2)

    axes[-1].set_xlabel("Time (s)")
    epoch_str = f"[Epoch {epoch+1}]  " if epoch is not None else ""
    task_str  = f"{task_label} — " if task_label else ""
    overlap_str = (f"step={block_step} frames < {K}-frame horizon -> predictions overlap in real time"
                   if overlapping else f"step={block_step} frames = horizon -> predictions tile with no overlap")
    fig.suptitle(
        f"{epoch_str}{task_str}Future Prediction — {group_label} — GT (blue) vs Prediction (red dashed)\n"
        f"{n_blocks} blocks, {H} frames of context each -> {K} genuinely-future frames ({K/sample_rate*1000:.0f}ms) "
        f"per block  |  {overlap_str}  |  physical units",
        fontsize=9, fontweight="bold",
    )
    _show(fig, save_path)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(model, optimizer, epoch, config, norm_stats, log_dir):
    path = os.path.join(log_dir, f"tcnn_model_epoch_{epoch}.pt")
    torch.save(
        {
            "tcnn_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "in_mean":  torch.tensor(norm_stats["mean_in"]),
            "in_std":   torch.tensor(norm_stats["std_in"]),
            "out_mean": torch.tensor(norm_stats["mean_out"]),
            "out_std":  torch.tensor(norm_stats["std_out"]),
        },
        path,
    )
    print(f"[TCNN] Checkpoint saved → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    config = {
        # Data: every Data/trial_*_all_data.csv, each split 80/20 chronologically
        "history_horizon": 150,      # ~1.0 s of context at 150 Hz
        "forecast_horizon": 50,      # ~0.5 s ahead at 150 Hz
        "num_channels": [80, 80, 80, 80, 80],   # TCN hidden channels per level
        "kernel_size": 6,
        "dropout": 0.2,
        # Loss settings
        "gamma":        0.95,        # horizon discount
        "weight_angle":  1.0,        # relative weight for joint angles (°)
        "weight_moment": 1.0,        # relative weight for joint moments (N·m/kg)
        "batch_size": 256,           # dataset fits on GPU — large batches keep it fed
        "num_workers": 4,            # only used when data can't live on GPU (CPU fallback)
        "lr": 1e-4,
        # 0.0, not FLD's 5e-4: every TCN layer is weight_norm-parametrized
        # (weight = g * v/||v||), and the loss is invariant to ||v||'s raw
        # magnitude -- so L2 weight_decay is the ONLY force acting on ||v||,
        # slowly shrinking it every step with nothing pushing back. After a
        # few thousand steps ||v|| gets small enough that v/||v|| amplifies
        # float rounding noise and the whole network blows up to NaN in a
        # single step (confirmed by reproducing training with instrumentation:
        # loss/grad-norm traces were smooth and gradient clipping never
        # engaged right up to the NaN step, and weight_decay=0.0 trains
        # through 16k+ steps with ||v|| steadily *growing* instead of
        # shrinking). FLD has no weight_norm anywhere so it doesn't hit this.
        "weight_decay": 0.0,
        "noise_level": 0.05,
        "epochs": 1,
        "save_every": 10,
        "use_amp": True,             # bf16 autocast on CUDA
        "compile_model": True,       # torch.compile, falls back to eager on failure
    }

    train(config)


if __name__ == "__main__":
    main()
