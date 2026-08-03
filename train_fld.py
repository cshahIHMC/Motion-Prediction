"""
train_fld.py

Training script for the FLD (Fourier Latent Dynamics) model applied to
biomechanical sensor data, trained jointly across multiple recording
sessions ("tasks"/trials). Tasks are discovered from Data/all_data.csv
(built by combine_data.py -- every subject's trial CSVs stacked with a
trial_id/subject_name/task column) -- no separate mapping file is used or
needed.

Input  = body acc (5x3=15) + body gyro (5x3=15) + insole acc (2x3=6) + insole gyro (2x3=6)
       + insole force/COP (L/R x 3 = 6) + subject weight + subject height  ->  50 features total
Output = 10 joint angles (deg) + 10 joint moments (N.m/kg, each divided by
         that row's own subject_weight -- see _load_all_data) = 20 features

At 150 Hz, each training sample is a clean, disjoint split (history_horizon
and forecast_horizon are config values -- H=151/K=50 shown below as an
example, current config may differ, see main()):
  context window : rows [s, s+H)     -- H frames of context, fed to the encoder
  future target   : rows [s+H, s+H+K) -- K genuinely-future frames, never seen by the encoder

The model's forward pass returns phase-shifted reconstructions; only the last
frame of each shifted reconstruction is used, which lands exactly one step
past the previous one -- so the K compared predictions correspond 1:1 with
the K genuinely-future target rows. See _forecast() for the exact mechanism.
No windowed/overlapping comparisons anywhere -- input in, output out.

Each trial's rows are its own contiguous recording. Context/future pairs are
built per-trial and never cross a trial boundary. Train/val window starts
come from Data/train_val_index.csv (built by build_train_val_index.py):
each trial's rows are split ~85/15, with val spread across a few segments
through the trial (not just the tail) and a window_size gap enforced around
every train/val boundary so no window pair leaks across the split. A global
normalisation is fit on the pooled training rows across all tasks (excluding
any row a validation window reads -- see _norm_stats_mask) so every task
shares one input/output scale. Long-trajectory ("future prediction") and
phase-manifold (PCA) plots are generated once per task, drawn from that
task's largest validation segment; the remaining diagnostic plots summarise
the combined model/dataset.

Run:
    python3 combine_data.py            # once, or after adding trials/subjects
    python3 build_train_val_index.py   # once, or after the above changes
    python3 train_fld.py
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

from DataLoader.data_loader import FLDDataset
from Models.FLD import FLD
from Library.fld_plotter import Plotter

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
# they let the encoder condition its output on who's wearing the sensors.
SUBJECT_COLUMNS = ['subject_weight', 'subject_height']

# What actually gets fed to the encoder: wearable sensors + subject constants
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
    usecols = list(dict.fromkeys(['trial_id', 'subject_name', 'task'] + INPUT_COLUMNS + OUTPUT_COLUMNS))
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
# Distribution buffer
# ---------------------------------------------------------------------------

class DistributionBuffer:
    def __init__(self, latent_dim: int, max_size: int = 5000):
        self._buf = []
        self._latent_dim = latent_dim
        self._max_size = max_size

    def insert(self, data: torch.Tensor):
        self._buf.append(data.detach().cpu())
        if len(self._buf) > self._max_size:
            self._buf = self._buf[-self._max_size:]

    def get(self) -> torch.Tensor:
        if not self._buf:
            return torch.zeros(1, self._latent_dim)
        return torch.cat(self._buf, dim=0)

    def clear(self):
        self._buf = []


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_reconstruction_loss(pred_all, target_all,
                                 gamma=1.0, feature_weights=None):
    """
    pred_all, target_all : (K, B, features)  normalised tensors — one genuinely
    future point per forecast step (see _forecast() below), not a window.

    - Loss computed in NORMALISED space — no denormalisation, so mixed
      physical units (deg/s, m/s², N, m) don't inflate one modality.
    - AVERAGED over K steps (not summed), so magnitude is comparable
      across datasets with different forecast horizons.
    - Optional gamma discounting: step i is weighted by gamma**i so
      near-term accuracy is emphasised more (default 1.0 = off).
    - Optional feature_weights (features,): multiply squared error
      element-wise to up/down-weight modality groups.
    - std clamped ≥ 1e-6 upstream — no division by zero risk here.
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

    return total / weight                # normalised by effective step count


def per_feature_rmse(pred_all, target_all, std):
    """
    RMSE in original physical units per feature, averaged over K and B.

    pred_all, target_all : (K, B, features) normalised
    std                  : (features,) — training dataset std per feature

    Returns (features,) tensor — e.g. deg/s for gyro, N for force.
    """
    sq_err   = (pred_all - target_all).pow(2)       # (K, B, features)
    mse_norm = sq_err.mean(dim=(0, 1))                # (features,)  mean over K,B
    rmse_norm = mse_norm.sqrt()                       # normalised scale
    return rmse_norm * std.clamp(min=1e-6).to(pred_all.device)


def rmse_by_horizon(pred_all, target_all):
    """
    RMSE in normalised space per forecast step.

    pred_all, target_all : (K, B, features) normalised

    Returns (K,) tensor — rising curve = accuracy degrades with horizon.
    """
    sq_err   = (pred_all - target_all).pow(2)        # (K, B, features)
    mse_k    = sq_err.mean(dim=(1, 2))                # (K,)
    return mse_k.sqrt()


def phase_propagation_error(model, window_t, window_tk, dt, k):
    """
    Circular error between FLD's linearly-propagated phase prediction and
    the phase obtained by independently re-encoding the actual future window.

    window_t, window_tk : (B, features, H) normalised — anchor and future
    dt                  : 1 / sample_rate (seconds per frame)
    k                   : number of steps between the two windows

    Returns mean absolute circular error in radians (scalar).

    GOOD: small error → linear phase dynamics are accurate at horizon k.
    BAD:  large error → non-stationarity or aperiodic motion at horizon k.
    """
    with torch.no_grad():
        _, _, _, params_t  = model(window_t)
        _, _, _, params_tk = model(window_tk)

    phi0       = params_t[0]                                   # (B, latent)
    freq       = params_t[1]                                   # (B, latent)
    phi_pred   = phi0 + freq * dt * k                          # (B, latent)
    phi_actual = params_tk[0]                                  # (B, latent)

    diff = phi_pred - phi_actual
    # circular difference wraps to (−π, π]
    circular = torch.atan2(
        torch.sin(2.0 * torch.pi * diff),
        torch.cos(2.0 * torch.pi * diff),
    )
    return circular.abs().mean().item()


def compute_diversity_loss(frequency):
    """
    Penalise latent channels for having the same frequency.

    frequency : (B, latent_dim)

    Computes mean pairwise absolute distance between channel frequencies
    and returns the negative — minimising total loss maximises spread.

    With homogeneous data (single walking speed) the model naturally
    collapses all channels to one frequency. This loss forces channels
    toward different harmonics or timescales even in homogeneous data.
    """
    B, L = frequency.shape
    freq_i = frequency.unsqueeze(2)
    freq_j = frequency.unsqueeze(1)
    pairwise_dist = (freq_i - freq_j).abs()   # (B, L, L)
    mask = (1 - torch.eye(L, device=frequency.device)).unsqueeze(0)
    mean_dist = (pairwise_dist * mask).sum() / (B * L * (L - 1))
    return -mean_dist


def _forecast(model, context, forecast_horizon):
    """Run the FLD model on a context window and return only genuinely-future
    predictions — no windowed/overlapping comparison anywhere.

    model(context, k) returns k phase-shifted reconstructions, each a full
    history_horizon-length window; index 0 is the step-0 reconstruction of the
    context itself (not future). So we ask for one extra step and drop it:
    the LAST frame of shifted window step (1..forecast_horizon) lands exactly
    `step` frames past the end of the context — i.e. rows context_end+1 ..
    context_end+forecast_horizon, matching FLDDataset's `future` target 1:1.

    Returns (future_pred, latent, signal, params):
      future_pred : (forecast_horizon, B, output_dim)
    """
    pred_dynamics, latent, signal, params = model(context, k=forecast_horizon + 1)
    future_pred = pred_dynamics[1:, :, :, -1]   # (K, B, output_dim) — drop step 0, last frame of each
    return future_pred, latent, signal, params


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: dict):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FLD] Using device: {device}")
    if device.type == "cuda":
        print(f"[FLD] GPU: {torch.cuda.get_device_name(device)}")

    use_amp  = config.get("use_amp", True) and device.type == "cuda"
    amp_dtype = torch.bfloat16

    # The normalised dataset is small (tens of MB) — keep it resident on the
    # GPU so __getitem__ slices straight out of device memory. This avoids
    # per-batch host->device copies and lets the DataLoader run with
    # num_workers=0 (CUDA tensors can't be shared across worker processes).
    data_device = device if device.type == "cuda" else None
    num_workers = 0 if data_device is not None else config["num_workers"]

    log_dir   = os.path.join("runs", f"FLD_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
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
    print(f"[FLD] Discovered {len(task_labels)} tasks in {ALL_DATA_PATH}: {task_labels}")

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
        print(f"[FLD] {trial_id} ({task_labels[trial_id]}): {len(df):,} rows "
              f"({len(df)/150:.1f}s @150Hz)  train_windows={n_train_win:,}  val_windows={n_val_win:,}  "
              f"subject={info['weight']:.1f}kg/{info['height']:.0f}cm")

    # Global normalisation fit on TRAIN rows only, pooled across all tasks —
    # every task shares one input/output scale so the model isn't juggling
    # per-task statistics.
    raw_in_all  = np.concatenate(train_raw_in,  axis=0)
    raw_out_all = np.concatenate(train_raw_out, axis=0)
    if config.get("normalize_input", True):
        mean_in, std_in = raw_in_all.mean(axis=0), raw_in_all.std(axis=0)
        std_in[std_in == 0] = 1.0
    else:
        # mean=0/std=1 is a no-op transform ((x-0)/1 == x) -- feeds raw
        # physical-unit sensor values straight to the encoder. Input
        # channels span wildly different scales (insole force in the
        # hundreds-to-thousands vs. accel/gyro likely single-to-double
        # digits), so expect the first Conv1d layer to be numerically
        # dominated by whichever channel has the largest raw magnitude --
        # this is an ablation, not expected to train as well as normalized.
        print("[FLD] normalize_input=False: encoder sees RAW (unnormalised) sensor values.")
        mean_in = np.zeros(raw_in_all.shape[1], dtype=np.float32)
        std_in  = np.ones(raw_in_all.shape[1], dtype=np.float32)
    mean_out, std_out = raw_out_all.mean(axis=0), raw_out_all.std(axis=0)
    std_out[std_out == 0] = 1.0
    norm_stats = {'mean_in': mean_in, 'std_in': std_in, 'mean_out': mean_out, 'std_out': std_out}
    out_std_cpu = torch.tensor(std_out, dtype=torch.float32)   # for physical-unit diagnostics below

    # ---- Per-task FLDDataset — windows are built within one file only --------
    task_info = {}
    train_subsets, val_subsets = [], []
    for trial_id, label in task_labels.items():
        df = task_dfs[trial_id]   # already ordered INPUT_COLUMNS + OUTPUT_COLUMNS
        dataset = FLDDataset(
            df,
            history_horizon=history_horizon,
            forecast_horizon=forecast_horizon,
            feature_set=config["feature_set"],
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
            # (val is no longer one contiguous tail region) -- long-trajectory/
            # PCA-manifold plots below use the largest contiguous run of these.
            "val_runs": _val_runs(val_idx, window_size),
        }
        print(f"[FLD] {trial_id} windows -> train={len(train_subset):,}  val={len(val_subset):,}")

    task_ids   = list(task_labels.keys())
    ref_task_id = task_ids[0]
    ref_dataset = task_info[ref_task_id]["dataset"]   # representative dataset for whole-model diagnostics

    train_dataset = ConcatDataset(train_subsets)
    val_dataset   = ConcatDataset(val_subsets)
    print(f"[FLD] Combined: {len(train_dataset):,} train windows, {len(val_dataset):,} val windows "
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
    model = FLD(
        observation_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        history_horizon=history_horizon,
        latent_channel=config["latent_dim"],
        device=device,
        dt=1.0 / 150.0,               # every task is recorded at ~150 Hz
        encoder_shape=config["encoder_shape"],
        decoder_shape=config["decoder_shape"],
    )
    model.to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=config["lr"],
                           weight_decay=config["weight_decay"])

    if config.get("compile_model", False):
        try:
            model = torch.compile(model)
            print("[FLD] torch.compile enabled.")
        except Exception as e:
            print(f"[FLD] torch.compile failed, continuing eager: {e}")

    plotter     = Plotter()
    latent_dim  = config["latent_dim"]
    noise_level = config["noise_level"]

    # Persistent figures (cleared each plot cycle)
    fig_dist,   ax_dist   = plt.subplots(1, 3, figsize=(15, 4))
    fig_recon,  ax_recon  = plt.subplots(6, 1, figsize=(10, 18))
    fig_params, ax_params = plt.subplots(latent_dim, 5, figsize=(20, latent_dim * 3))
    fig_pca,    ax_pca    = plt.subplots(figsize=(8, 8))
    fig_loss,   ax_loss   = plt.subplots(figsize=(8, 4))
    # Per-channel latent parameter histograms
    fig_hist,   ax_hist   = plt.subplots(latent_dim, 4, figsize=(18, 3.5 * latent_dim))
    # Long trajectory — split into angles and moments, one figure reused per task
    _n_angles  = 10
    _n_moments = 10
    fig_long_angle,  ax_long_angle  = plt.subplots(_n_angles,  1, figsize=(30, 2.0 * _n_angles),  sharex=True)
    fig_long_moment, ax_long_moment = plt.subplots(_n_moments, 1, figsize=(30, 2.0 * _n_moments), sharex=True)
    # Per-task PCA phase manifold — one figure reused per task
    fig_task_pca, ax_task_pca = plt.subplots(figsize=(8, 8))
    # Latent sine curves — one row per latent channel, illustrative (first task)
    fig_latent, ax_latent = plt.subplots(latent_dim, 1, figsize=(30, 3.5 * latent_dim), sharex=True)

    buf_freq = DistributionBuffer(latent_dim)
    buf_amp  = DistributionBuffer(latent_dim)
    buf_off  = DistributionBuffer(latent_dim)

    train_losses, val_losses = [], []
    gamma       = config.get("gamma", 1.0)
    n_features  = OUTPUT_DIM

    # Build per-feature weight tensor from config modality weights
    fw = torch.ones(n_features, device=device)
    for modality, group in MODALITY_GROUPS.items():
        w = config.get(f"weight_{modality}", 1.0)   # e.g. "weight_angle" / "weight_moment"
        for idx in group["indices"]:
            if idx < n_features:
                fw[idx] = w
    feature_weights = fw  # (n_output_features,)

    diag_k = [9, 24, forecast_horizon - 1]   # horizon steps to log (0-indexed)

    # ---- Training loop -------------------------------------------------------
    train_start = datetime.now()
    print(f"[FLD] Training started at {train_start.strftime('%Y-%m-%d %H:%M:%S')}.")
    for epoch in range(config["epochs"]):
        epoch_start = datetime.now()
        model.train()
        buf_freq.clear(); buf_amp.clear(); buf_off.clear()
        running_loss = 0.0

        for batch_x, batch_y in train_dl:
            batch_x = batch_x.to(device, non_blocking=True)   # (B, input_dim, H) — context
            batch_y = batch_y.to(device, non_blocking=True)   # (B, K, output_dim) — future

            if noise_level > 0.0:
                batch_x = batch_x + torch.randn_like(batch_x) * noise_level

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                future_pred, latent, signal, params = _forecast(model, batch_x, forecast_horizon)
                tgt_all = batch_y.permute(1, 0, 2)   # (K, B, output_dim)
                loss = compute_reconstruction_loss(future_pred, tgt_all, gamma, feature_weights)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            phase, frequency, amplitude, offset = params
            buf_freq.insert(frequency)
            buf_amp.insert(amplitude)
            buf_off.insert(offset)

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
                    future_pred, _, _, _ = _forecast(model, batch_x, forecast_horizon)
                    tgt_all = batch_y.permute(1, 0, 2)
                    clean_running += compute_reconstruction_loss(
                        future_pred, tgt_all, gamma, feature_weights
                    ).item()
                n_clean += 1
        train_loss = clean_running / max(n_clean, 1)

        # ---- Validation + diagnostics (combined across all tasks) ------------
        model.eval()
        val_running = 0.0
        # Accumulate one full pass for diagnostics
        all_pred, all_tgt, all_inp = [], [], []
        with torch.no_grad():
            for batch_x, batch_y in val_dl:
                batch_x = batch_x.to(device, non_blocking=True)   # (B, input_dim, H) — context
                batch_y = batch_y.to(device, non_blocking=True)   # (B, K, output_dim) — future
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    future_pred, _, _, _ = _forecast(model, batch_x, forecast_horizon)
                    tgt_all = batch_y.permute(1, 0, 2)
                    val_running += compute_reconstruction_loss(
                        future_pred, tgt_all, gamma, feature_weights
                    ).item()
                # collect first 8 batches for diagnostics (avoid OOM)
                if len(all_pred) < 8:
                    all_pred.append(future_pred.float().cpu())
                    all_tgt.append(tgt_all.float().cpu())
                    all_inp.append(batch_x.float().cpu())

        val_loss = val_running / len(val_dl)

        # --- Diagnostic computations on accumulated val batches ---------------
        pred_cat = torch.cat([p for p in all_pred], dim=1)   # (K, B_acc, out_dim)
        tgt_cat  = torch.cat([t for t in all_tgt],  dim=1)  # (K, B_acc, out_dim)
        inp_cat  = torch.cat([i for i in all_inp],  dim=0)  # (B_acc, in_dim, H)

        # RMSE by horizon (normalised space)
        horiz_rmse = rmse_by_horizon(pred_cat, tgt_cat)   # (K,)
        diag_k_vals = [horiz_rmse[k].item() for k in diag_k]

        # Phase propagation error at each diagnostic horizon
        phase_errs = []
        for k_step in diag_k:
            # Phase propagation must use INPUT windows (48 sensor channels), not output targets
            window_t  = inp_cat.to(device)                               # (B_acc, in_dim, H)
            window_tk = inp_cat.to(device)                               # same window — approximate
            phase_errs.append(
                phase_propagation_error(model, window_t, window_tk, 1.0/150.0, k_step+1)
            )

        # Per-feature RMSE in original units (averaged over all K)
        feat_rmse = per_feature_rmse(pred_cat, tgt_cat, out_std_cpu)  # (out_features,)
        mod_rmse  = {}
        for modality, group in MODALITY_GROUPS.items():
            valid = [i for i in group["indices"] if i < n_features]
            mod_rmse[modality] = feat_rmse[valid].mean().item() if valid else 0.0

        # Cumulative per-feature RMSE, in original units: _feat_rmse_k(k_idx)
        # averages squared error over every step from frame 1 through frame
        # k_idx+1 (not just the single frame at k_idx+1), so e.g. the 500ms
        # column is the average error across the WHOLE 0-500ms horizon, not
        # just the error exactly at 500ms.
        std_t = out_std_cpu.clamp(min=1e-6)
        def _feat_rmse_k(k_idx):
            k_idx = min(k_idx, pred_cat.shape[0] - 1)
            err = (pred_cat[:k_idx + 1] - tgt_cat[:k_idx + 1]).pow(2).mean(dim=(0, 1)).sqrt()  # (F,)
            return (err * std_t).tolist()

        feat_units = ([MODALITY_GROUPS["angle"]["unit"]] * 10
                      + [MODALITY_GROUPS["moment"]["unit"]] * 10)   # 10 joint angles, 10 joint moments
        # Nearest step (frame 1, ~7ms), then every 100ms out to forecast_horizon --
        # adapts automatically if forecast_horizon changes, rather than always
        # showing only the first few frames regardless of how far ahead we predict.
        frames_per_100ms = round(150.0 * 0.1)   # 15 at 150 Hz
        n_100ms_points   = forecast_horizon // frames_per_100ms
        K_table    = [0] + [i * frames_per_100ms - 1 for i in range(1, n_100ms_points + 1)]
        K_table    = [k for k in K_table if k < forecast_horizon]
        ms_labels  = [f"0-{(k+1)/150*1000:.0f}ms avg" for k in K_table]
        rmse_table = [_feat_rmse_k(k) for k in K_table]

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        hz_str = "  ".join(f"k{k+1}={v:.4f}" for k, v in zip(diag_k, diag_k_vals))
        ph_str = "  ".join(f"k{k+1}={v:.4f}rad" for k, v in zip(diag_k, phase_errs))
        print(f"[FLD] Epoch [{epoch + 1:>4}/{config['epochs']}]  "
              f"train(clean)={train_loss:.5f}  train(noisy)={train_loss_noisy:.5f}  val={val_loss:.5f}  "
              f"| horizon RMSE: {hz_str}  "
              f"| phase err: {ph_str}")

        # Modality RMSE summary
        mod_str = "  ".join(f"{k}={v:.3f}" for k, v in mod_rmse.items())
        print(f"          modality RMSE (orig units): {mod_str}")

        # Per-feature RMSE table
        col_w  = 22
        cell_w = 14
        header_row = f"          {'Feature':<{col_w}}" + "".join(f" {lbl:>{cell_w}}" for lbl in ms_labels)
        sep    = f"          {'-'*col_w}" + "".join(f" {'-'*cell_w}" for _ in ms_labels)
        print(header_row)
        print(sep)
        for fi, (name, unit) in enumerate(zip(OUTPUT_COLUMNS, feat_units)):
            row_str = f"          {name[:col_w]:<{col_w}}"
            for rmse_k in rmse_table:
                row_str += f" {rmse_k[fi]:>{cell_w-2}.3f}{unit} "
            print(row_str)

        # All plots fire once at the end of training only
        if epoch == config["epochs"] - 1:
            with torch.no_grad():
                model.eval()
                _plot_loss_curve(fig_loss, ax_loss, train_losses, val_losses,
                                 save_path=os.path.join(log_dir, "loss_curve.png"))
                _plot_epoch(
                    plotter, model, ref_dataset, device,
                    fig_dist, ax_dist,
                    fig_recon, ax_recon,
                    fig_params, ax_params,
                    fig_pca, ax_pca,
                    buf_freq, buf_amp, buf_off,
                    latent_dim, history_horizon, forecast_horizon,
                    epoch, INPUT_DIM, OUTPUT_DIM,
                    save_dir=plots_dir,
                )
                _plot_parameter_histograms(
                    fig_hist, ax_hist,
                    buf_freq, buf_amp, buf_off,
                    latent_dim, epoch, history_horizon,
                    save_path=os.path.join(plots_dir, "whole_param_histograms.png"),
                )

                # --- Per-task plots: future-prediction trajectory + PCA manifold ---
                # Val is now spread across a few segments per trial (see
                # build_train_val_index.py), not one contiguous tail -- these
                # plots need a single contiguous stretch, so use the largest
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
                    _plot_pca_manifold(
                        model, task_ds, device, history_horizon,
                        ax_task_pca, plotter,
                        title=f"Phase Manifold — {trial_id} ({info['label']}) — validation region",
                        start_frame=run_start,
                        manifold_windows=min(500, run_end - run_start - history_horizon),
                        save_path=os.path.join(plots_dir, f"{prefix}_pca_manifold.png"),
                    )

                ref_run_start, ref_run_end = max(
                    task_info[ref_task_id]["val_runs"], key=lambda r: r[1] - r[0]
                )
                _plot_latent_sine_trajectory(
                    model, ref_dataset, device,
                    history_horizon, fig_latent, ax_latent,
                    n_blocks=90, forecast_horizon=forecast_horizon,
                    sample_rate=150.0, epoch=epoch,
                    start_frame=ref_run_start, max_frame=ref_run_end,
                    save_path=os.path.join(plots_dir, "whole_latent_sine.png"),
                )
                model.train()

        # Checkpoint
        if epoch % config["save_every"] == 0:
            _save(model, optimizer, epoch, config, norm_stats, log_dir)

        epoch_end = datetime.now()
        print(f"[FLD] Epoch {epoch + 1} finished at {epoch_end.strftime('%Y-%m-%d %H:%M:%S')}  "
              f"(epoch took {(epoch_end - epoch_start).total_seconds():.1f}s, "
              f"elapsed {(epoch_end - train_start).total_seconds() / 60:.1f} min total)")

    _save(model, optimizer, config["epochs"], config, norm_stats, log_dir)
    print(f"[FLD] Training finished. Checkpoints + plots in: {log_dir}")
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
    ax.set_title("FLD Training Loss")
    ax.legend()
    ax.grid(True)
    _show(fig, save_path)


def _show(fig, save_path=None):
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    fig.canvas.draw()
    plt.pause(0.05)


def _plot_epoch(plotter, model, dataset, device,
                fig_dist, ax_dist,
                fig_recon, ax_recon,
                fig_params, ax_params,
                fig_pca, ax_pca,
                buf_freq, buf_amp, buf_off,
                latent_dim, history_horizon, forecast_horizon,
                epoch, input_dim, output_dim, save_dir=None):

    def _p(name):
        return os.path.join(save_dir, name) if save_dir else None

    # --- Distribution plots --------------------------------------------------
    plotter.plot_distribution(ax_dist[0], buf_freq.get(), title="Frequency Distribution")
    plotter.plot_distribution(ax_dist[1], buf_amp.get(),  title="Amplitude Distribution")
    plotter.plot_distribution(ax_dist[2], buf_off.get(),  title="Offset Distribution")
    _show(fig_dist, _p("whole_distributions.png"))

    # --- Encode the first context window in the dataset ----------------------
    eval_x, eval_y = dataset[0]
    eval_input  = eval_x.unsqueeze(0).to(device)      # (1, input_dim, H) — context
    eval_target = eval_y.unsqueeze(0).to(device)      # (1, K, output_dim) — future

    future_pred, latent, signal, params = _forecast(model, eval_input, forecast_horizon)
    pred_future = future_pred[:, 0, :].T              # (output_dim, K) — predicted future trajectory
    gt_future   = eval_target[0].T                    # (output_dim, K) — actual future trajectory

    # --- Reconstruction overview (6 rows) ------------------------------------
    plotter.plot_curves(ax_recon[0], eval_input[0], -1.0, 1.0, -5.0, 5.0,
                        title=f"Encoder Input ({input_dim} channels: sensors + subject × {history_horizon} steps)", show_axes=False)
    plotter.plot_curves(ax_recon[1], latent[0], -1.0, 1.0, -2.0, 2.0,
                        title=f"Latent Embedding ({latent_dim}×{history_horizon})", show_axes=False)
    plotter.plot_circles(ax_recon[2], params[0][0], params[2][0],
                         title=f"Learned Phase Timing ({latent_dim} channels)", show_axes=False)
    plotter.plot_curves(ax_recon[3], signal[0], -1.0, 1.0, -2.0, 2.0,
                        title=f"Parametrised Fourier Signal ({latent_dim}×{history_horizon})", show_axes=False)
    plotter.plot_curves(ax_recon[4], pred_future, -1.0, 1.0, -5.0, 5.0,
                        title=f"Predicted Future Output ({output_dim} channels × {forecast_horizon} steps ahead)", show_axes=False)
    plotter.plot_curves(
        ax_recon[5],
        torch.vstack((gt_future.flatten(0, 1).unsqueeze(0),
                      pred_future.flatten(0, 1).unsqueeze(0))),
        -1.0, 1.0, -5.0, 5.0,
        title="Target vs Predicted Future (flattened)", show_axes=False,
    )
    _show(fig_recon, _p("whole_reconstruction.png"))

    # --- Per-channel Fourier parameter plots ---------------------------------
    phase_all     = params[0][0]   # (latent_dim,)
    frequency_all = params[1][0]
    amplitude_all = params[2][0]
    offset_all    = params[3][0]

    for j in range(latent_dim):
        plotter.plot_phase_1d(ax_params[j, 0], phase_all[j].unsqueeze(0), amplitude_all[j].unsqueeze(0),
                              title=("1D Phase" if j == 0 else None), show_axes=False)
        plotter.plot_phase_2d(ax_params[j, 1], phase_all[j].unsqueeze(0), amplitude_all[j].unsqueeze(0),
                              title=("2D Phase" if j == 0 else None), show_axes=False)
        plotter.plot_curves(ax_params[j, 2], frequency_all[j].view(1, 1),
                            -1.0, 1.0, 0.0, 4.0, title=("Frequency" if j == 0 else None), show_axes=False)
        plotter.plot_curves(ax_params[j, 3], amplitude_all[j].view(1, 1),
                            -1.0, 1.0, 0.0, 1.0, title=("Amplitude" if j == 0 else None), show_axes=False)
        plotter.plot_curves(ax_params[j, 4], offset_all[j].view(1, 1),
                            -1.0, 1.0, -1.0, 1.0, title=("Offset" if j == 0 else None), show_axes=False)
    _show(fig_params, _p("whole_latent_params.png"))

    # --- PCA phase manifold — one clean gait cycle ---------------------------
    _plot_pca_manifold(
        model, dataset, device, history_horizon, ax_pca, plotter,
        title="Phase Manifold — 500 windows (~3.3 s, training data)",
        start_frame=500, manifold_windows=500,
        save_path=_p("whole_pca_manifold.png"), fig=fig_pca,
    )


def _plot_pca_manifold(model, dataset, device, history_horizon, ax, plotter,
                        title, start_frame=0, manifold_windows=500,
                        save_path=None, fig=None):
    """Encode `manifold_windows` consecutive context windows starting at
    start_frame and plot their Fourier phase manifold via PCA."""
    data = dataset.data_in
    N = data.shape[0]
    manifold_windows = max(1, min(manifold_windows, N - start_frame - history_horizon))
    manifold_data = data[start_frame: start_frame + history_horizon + manifold_windows].to(device)
    seqs = torch.stack([
        manifold_data[i: i + history_horizon].T for i in range(manifold_windows)
    ])   # (manifold_windows, in_dim, H)
    _, _, _, m_params = model(seqs)
    phase_m     = m_params[0]   # (manifold_windows, latent_dim)
    amplitude_m = m_params[2]
    manifold = torch.hstack((
        amplitude_m * torch.sin(2.0 * torch.pi * phase_m),
        amplitude_m * torch.cos(2.0 * torch.pi * phase_m),
    ))
    plotter.plot_pca(ax, [manifold.cpu()], title=title)
    _show(fig if fig is not None else ax.figure, save_path)


# ---------------------------------------------------------------------------
# Per-channel latent parameter histograms
# ---------------------------------------------------------------------------

def _plot_parameter_histograms(fig, axes, buf_freq, buf_amp, buf_off,
                                latent_dim, epoch, history_horizon, save_path=None):
    """
    Histograms of frequency, amplitude, offset, and phase per latent channel,
    collected across every batch in the epoch.

    Layout: latent_dim rows × 4 columns
      Col 0 — Frequency   (how fast each channel oscillates)
      Col 1 — Amplitude   (oscillation strength per channel)
      Col 2 — Offset      (DC bias per channel)
      Col 3 — Phase       (position in cycle, −0.5 to 0.5)

    Key diagnostic: if all frequency histograms peak at the same value
    → channel collapse (all channels encoding the same frequency).
    Diverse peaks → healthy latent space.
    """
    freq_data  = buf_freq.get().numpy()   # (N, latent_dim)
    amp_data   = buf_amp.get().numpy()
    off_data   = buf_off.get().numpy()

    cmap = plt.cm.tab10
    # 1 cycle/window in Hz depends on the window length, not a fixed constant
    freq_1hz = 150.0 / history_horizon   # ≈ 1 Hz in cycles/window units
    freq_2hz = 2.0 * freq_1hz            # ≈ 2 Hz (step frequency)

    col_info = [
        ("Frequency  (cycles/window)", freq_data,
         "GOOD: channels peak at DIFFERENT values.\nBAD: all at same value = collapse.\n"
         "Dashed lines: 1 Hz (stride) and 2 Hz (step) reference."),
        ("Amplitude  (oscillation strength)", amp_data,
         "GOOD: peaks above zero = active channels.\nBAD: peak near zero = dead channel."),
        ("Offset  (DC bias)", off_data,
         "GOOD: near zero (data normalised).\nBAD: large = encoding mean, not oscillation."),
    ]

    for j in range(latent_dim):
        for col, (title, data, note) in enumerate(col_info):
            ax = axes[j, col] if latent_dim > 1 else axes[col]
            ax.cla()
            vals   = data[:, j]
            mean_v = vals.mean()
            std_v  = vals.std()
            ax.hist(vals, bins=50, color=cmap(j), alpha=0.75, edgecolor="none")
            ax.axvline(mean_v, color="k", lw=1.5, ls="--",
                       label=f"μ={mean_v:.3f}  σ={std_v:.3f}")
            # Add 1 Hz and 2 Hz reference lines on frequency column only
            if col == 0:
                ax.axvline(freq_1hz, color="tomato",    lw=1.2, ls=":", alpha=0.8,
                           label=f"1 Hz ({freq_1hz:.2f} c/w)")
                ax.axvline(freq_2hz, color="steelblue", lw=1.2, ls=":", alpha=0.8,
                           label=f"2 Hz ({freq_2hz:.2f} c/w)")
            ax.legend(fontsize=7, loc="upper right")
            ax.set_ylabel(f"ch {j}", fontsize=8, rotation=0, labelpad=28, va="center")
            ax.grid(alpha=0.25)
            if j == 0:
                ax.set_title(title, fontsize=9, fontweight="bold")
                ax.text(0.02, 0.97, note, transform=ax.transAxes, fontsize=6.5,
                        va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fffde7",
                                  alpha=0.85, edgecolor="#cccc66"))

        # Phase col (col 3) — not in buf, derive from frequency sign/wrapping proxy
        # We don't store phase in buffer so show amplitude as a 2D scatter instead
        ax_p = axes[j, 3] if latent_dim > 1 else axes[3]
        ax_p.cla()
        amp_j  = amp_data[:, j]
        freq_j = freq_data[:, j]
        ax_p.scatter(freq_j, amp_j, s=2, alpha=0.15, color=cmap(j), edgecolors="none")
        ax_p.set_xlabel("Frequency", fontsize=7)
        ax_p.set_ylabel("Amplitude", fontsize=7)
        ax_p.grid(alpha=0.2)
        if j == 0:
            ax_p.set_title("Freq vs Amp scatter", fontsize=9, fontweight="bold")

    fig.suptitle(
        f"[Epoch {epoch+1}]  Latent Parameter Distributions  —  "
        f"{freq_data.shape[0]:,} windows\n"
        "Rows = channels  |  Col 0–2: histograms  |  Col 3: freq vs amp scatter",
        fontsize=9, fontweight="bold",
    )
    _show(fig, save_path)


# ---------------------------------------------------------------------------
# Latent sine curves over the long trial
# ---------------------------------------------------------------------------

def _plot_latent_sine_trajectory(model, dataset, device,
                                  history_horizon, fig, axes,
                                  n_blocks=200, forecast_horizon=50,
                                  sample_rate=150.0, epoch=None,
                                  batch_size=64, start_frame=0, max_frame=None,
                                  save_path=None):
    """
    For each block encode the context window and compute the latent sine curves
    for the K genuinely-future steps beyond it (rows context_end+1 .. context_end+K
    — same convention as _forecast()). Pass start_frame set to a validation
    region's start to plot validation data only; max_frame bounds how far
    blocks may read (defaults to the whole dataset) so this doesn't wander
    past that region into train rows -- see _plot_long_trajectory.
    """
    data = dataset.data_in
    N, D = data.shape
    if max_frame is not None:
        N = min(max_frame, N)
    H    = history_horizon
    K    = forecast_horizon
    dt   = 1.0 / sample_rate

    max_blocks = (N - start_frame - H) // K
    n_blocks   = min(n_blocks, max_blocks)
    latent_dim = model.latent_channel

    sine_blocks = []   # each entry: (K, latent_dim)

    model.eval()
    with torch.no_grad():
        for b_start in range(0, n_blocks, batch_size):
            b_end      = min(b_start + batch_size, n_blocks)
            ctx_starts = [start_frame + i * K for i in range(b_start, b_end)]
            wins = torch.stack(
                [data[s : s + H].T for s in ctx_starts]
            ).to(device)                                    # (B, in_dim, H)

            _, _, _, params = model(wins, k=1)
            phase  = params[0].cpu()   # (B, latent_dim)
            freq   = params[1].cpu()   # (B, latent_dim)
            amp    = params[2].cpu()   # (B, latent_dim)
            offset = params[3].cpu()   # (B, latent_dim)

            # `phase` is referenced to the CENTER of the context window (args=0).
            # Shift to the last context position, then step 1..K forward — same
            # rows _forecast() predicts (context_end+1 .. context_end+K).
            edge_offset = freq * (H - 1) * dt / 2                          # (B, latent_dim)
            k_steps = torch.arange(1, K + 1, dtype=torch.float32)          # (K,) -- 1-indexed, future only
            for bi in range(b_end - b_start):
                # phase at each of the K future steps: (K, latent_dim)
                phase_k = (phase[bi] + edge_offset[bi]).unsqueeze(0) \
                    + freq[bi].unsqueeze(0) * dt * k_steps.unsqueeze(1)
                sine_k  = amp[bi] * torch.sin(2 * torch.pi * phase_k) + offset[bi]
                sine_blocks.append(sine_k.numpy())          # (K, latent_dim)

    sine_arr = np.concatenate(sine_blocks, axis=0)          # (n_blocks*K, latent_dim)
    n_frames = sine_arr.shape[0]
    t        = np.arange(n_frames) / sample_rate
    cmap     = plt.cm.tab10

    ax_list = np.atleast_1d(axes)
    for ch, ax in enumerate(ax_list):
        ax.cla()
        ax.plot(t, sine_arr[:, ch], color=cmap(ch % 10), lw=0.9, alpha=0.85)
        for blk in range(1, n_blocks):
            ax.axvline(blk * K / sample_rate, color="grey", lw=0.4, ls=":", alpha=0.35)
        ax.set_ylabel(f"ch {ch}", fontsize=9, rotation=0, labelpad=32, va="center")
        ax.grid(alpha=0.15)
        ax.tick_params(labelsize=7)

    ax_list[-1].set_xlabel("Time (s)")
    epoch_str = f"[Epoch {epoch+1}]  " if epoch is not None else ""
    fig.suptitle(
        f"{epoch_str}Latent Sine Curves (genuinely-future steps) — A·sin(2π·(φ + f·dt·k)) + b per channel\n"
        f"{latent_dim} channels  |  {n_blocks} blocks × {K} frames past each block's context ({n_blocks*K/sample_rate:.1f}s)  |  "
        f"vertical lines = block boundaries",
        fontsize=9, fontweight="bold",
    )
    _show(fig, save_path)


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
    each context — see _forecast(). Everything is drawn on a real-time
    x-axis: the ground truth is one continuous curve, and each block's
    prediction is layered on top at its true time position.

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

    data_in  = dataset.data_in           # (N, in_dim) normalised — fed to encoder
    data_out = dataset.data_out          # (N, out_dim) normalised — GT for decoder
    std_np   = dataset.out_std_tensor.detach().cpu().numpy()
    mean_np  = dataset.out_mean_tensor.detach().cpu().numpy()
    N        = data_in.shape[0] if max_frame is None else min(max_frame, data_in.shape[0])
    H        = history_horizon
    K        = forecast_horizon

    # Maximum blocks available from start_frame onward
    # (last block needs: start_frame + (n_blocks-1)*block_step + H + K <= N)
    max_blocks = (N - start_frame - H - K) // block_step + 1
    n_blocks   = min(n_blocks, max(max_blocks, 0))
    if n_blocks <= 0:
        print(f"[FLD] _plot_long_trajectory: not enough rows for even one block, skipping.")
        return

    # One continuous ground-truth stretch spanning every block's future range
    gt_start = start_frame + H
    gt_end   = min(start_frame + H + (n_blocks - 1) * block_step + K, N)
    gt_full  = data_out[gt_start:gt_end].detach().cpu().numpy() * std_np + mean_np   # (T, out_dim)
    t_gt     = np.arange(gt_start, gt_end) / sample_rate

    pred_blocks = []   # list of (t_block, pred_block_orig) — one per block
    model.eval()
    with torch.no_grad():
        for b_start in range(0, n_blocks, batch_size):
            b_end   = min(b_start + batch_size, n_blocks)
            ctx_starts = [start_frame + i * block_step for i in range(b_start, b_end)]
            wins = torch.stack(
                [data_in[s : s + H].T for s in ctx_starts]
            ).to(device)                                      # (B, in_dim, H)

            future_pred, _, _, _ = _forecast(model, wins, K)   # (K, B, D) — genuinely future
            pred_block = future_pred.permute(1, 0, 2).cpu().numpy()   # (B, K, D)

            for bi, s in enumerate(ctx_starts):
                pred_orig = pred_block[bi] * std_np + mean_np           # (K, out_dim)
                t_block   = np.arange(s + H, s + H + K) / sample_rate
                pred_blocks.append((t_block, pred_orig))

    # Apply feature slice for this subplot group
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
    path = os.path.join(log_dir, f"fld_model_epoch_{epoch}.pt")
    torch.save(
        {
            "fld_state_dict": model.state_dict(),
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
    print(f"[FLD] Checkpoint saved → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # NOTE: history_horizon must be ODD for FLD's Conv1d same-size padding.
    # 151 is the nearest odd integer to 150 (~1.0s of context at 150Hz).
    config = {
        # Data: every Data/trial_*_all_data.csv, each split 80/20 chronologically
        "history_horizon": 151,     # ~1.0 s of context at 150 Hz (must be odd for FLD Conv1d)
        "forecast_horizon": 50,     # ~0.5 s ahead at 150 Hz
        "latent_dim": 8,             # Fourier latent channels — increased for cross-modal task
        "encoder_shape": [128, 64],   # larger encoder to handle 48→8 cross-modal compression
        "decoder_shape": [64, 128],   # larger decoder to expand 8→20 joint outputs
        "feature_set": "cross",      # input: 48 IMU/insole cols → output: 20 joint angle+moment cols
        # Loss settings
        "gamma":        0.95,        # horizon discount
        "weight_angle":  1.0,        # relative weight for joint angles (°)
        "weight_moment": 1.0,        # relative weight for joint moments (N·m/kg)
        "batch_size": 256,           # dataset fits on GPU — large batches keep it fed
        "num_workers": 4,            # only used when data can't live on GPU (CPU fallback)
        "lr": 1e-4,
        "weight_decay": 5e-4,
        "noise_level": 0.05,
        "epochs": 80,
        "save_every": 10,
        "use_amp": True,             # bf16 autocast on CUDA
        "compile_model": True,       # torch.compile, falls back to eager on failure
    }

    train(config)


if __name__ == "__main__":
    main()
