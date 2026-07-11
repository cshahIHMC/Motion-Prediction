"""
train_fld.py

Training script for the FLD (Fourier Latent Dynamics) model applied to
biomechanical sensor data from Trial 2.

Data: Trial_2_all_data_filtered_ID.csv  — all walking, ~150 Hz, ~11.6 min
Input  = body acc (5×3=15) + body gyro (5×3=15) + insole acc (2×3=6) + insole gyro (2×3=6)
       + insole force/COP (L/R × 3 = 6)  →  48 features total
Output = same 48 features, self-supervised against 50 future shifted windows

At 150 Hz:
  history_horizon = 151 frames = 1.007 s of context
  forecast_horizon = 50 frames = 0.333 s ahead

Run:
    python3 train_fld.py
"""

import os
import sys
import csv
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
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

DATA_PATH    = os.path.join(os.path.dirname(__file__), "Data", "Combined_file.csv")
TRIAL5_PATH  = os.path.join(os.path.dirname(__file__), "Data", "Trial_5_all_data_filtered_ID.csv")
TRIAL5_INCLINE_7MIN_ROWS  = 7 * 60 * 150  # last 7 min → incline walking data
TRIAL5_INCLINE_VAL_30S    = 30 * 150      # last 30s   → validation

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

# Output features — 10 joint angles (°) + 10 joint moments (N·m) = 20 features
OUTPUT_COLUMNS = [
    # Joint angles — degrees
    'hip_flexion_r',    'hip_flexion_l',
    'hip_adduction_r',  'hip_adduction_l',
    'hip_rotation_r',   'hip_rotation_l',
    'knee_angle_r',     'knee_angle_l',
    'ankle_angle_r',    'ankle_angle_l',
    # Joint moments — N·m
    'hip_flexion_r_moment',   'hip_flexion_l_moment',
    'hip_adduction_r_moment', 'hip_adduction_l_moment',
    'hip_rotation_r_moment',  'hip_rotation_l_moment',
    'knee_angle_r_moment',    'knee_angle_l_moment',
    'ankle_angle_r_moment',   'ankle_angle_l_moment',
    # Total: 20 features
]

# Output modality groups — for per-group RMSE logging
MODALITY_GROUPS = {
    "angle (°)  ": list(range(0, 10)),    # 10 joint angles
    "moment(N·m)": list(range(10, 20)),   # 10 joint moments
}

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
    pred_all, target_all : (K, B, features, H)  normalised tensors.

    Changes vs original:
      - Loss computed in NORMALISED space — no denormalisation, so mixed
        physical units (deg/s, m/s², N, m) don't inflate one modality.
      - AVERAGED over K windows (not summed), so magnitude is comparable
        across datasets with different forecast horizons.
      - Optional gamma discounting: window i is weighted by gamma**i so
        near-term accuracy is emphasised more (default 1.0 = off).
      - Optional feature_weights (features,): multiply squared error
        element-wise to up/down-weight modality groups.
      - std clamped ≥ 1e-6 upstream — no division by zero risk here.
    """
    K      = pred_all.shape[0]
    total  = torch.zeros(1, device=pred_all.device)
    weight = 1.0

    for i in range(K):
        sq_err = (pred_all[i] - target_all[i]).pow(2)   # (B, features, H)
        if feature_weights is not None:
            sq_err = sq_err * feature_weights.view(1, -1, 1)
        total  = total + sq_err.mean() * (gamma ** i)
        weight = weight + (gamma ** i)

    return total / weight                # normalised by effective window count


def per_feature_rmse(pred_all, target_all, std):
    """
    RMSE in original physical units per feature, averaged over K, B, H.

    pred_all, target_all : (K, B, features, H) normalised
    std                  : (features,) — training dataset std per feature

    Returns (features,) tensor — e.g. deg/s for gyro, N for force.
    """
    sq_err   = (pred_all - target_all).pow(2)       # (K, B, features, H)
    mse_norm = sq_err.mean(dim=(0, 1, 3))            # (features,)  mean over K,B,H
    rmse_norm = mse_norm.sqrt()                       # normalised scale
    return rmse_norm * std.clamp(min=1e-6).to(pred_all.device)


def rmse_by_horizon(pred_all, target_all):
    """
    RMSE in normalised space per forecast step.

    pred_all, target_all : (K, B, features, H) normalised

    Returns (K,) tensor — rising curve = accuracy degrades with horizon.
    """
    sq_err   = (pred_all - target_all).pow(2)        # (K, B, features, H)
    mse_k    = sq_err.mean(dim=(1, 2, 3))             # (K,)
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FLD] Using device: {device}")

    log_dir = os.path.join("runs", f"FLD_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)

    # ---- Data — Trial 5 incline walking (last 7 min) -------------------------
    print("[FLD] Loading Trial 5 incline walking data …")
    if not os.path.exists(TRIAL5_PATH):
        raise FileNotFoundError(f"Trial 5 file not found: {TRIAL5_PATH}")
    df5 = pd.read_csv(TRIAL5_PATH)
    print(f"[FLD] Trial 5: {len(df5):,} rows ({len(df5)/150:.1f}s at 150 Hz)")

    # Last 7 minutes = incline walking
    df5_incline = df5.iloc[-TRIAL5_INCLINE_7MIN_ROWS:].reset_index(drop=True)
    print(f"[FLD] Using last {TRIAL5_INCLINE_7MIN_ROWS:,} rows ({TRIAL5_INCLINE_7MIN_ROWS/150:.0f}s = 7 min) incline walking")
    df_combined = pd.concat([df5_incline[COLUMNS], df5_incline[OUTPUT_COLUMNS]], axis=1)

    dataset = FLDDataset(
        df_combined,
        history_horizon=config["history_horizon"],
        forecast_horizon=config["forecast_horizon"],
        feature_set=config["feature_set"],
    )
    print(f"[FLD] Dataset: {len(dataset):,} windows, input_dim={dataset.input_dim}, output_dim={dataset.output_dim}")

    # Val = last 30s, Train = everything before it (with window-size gap)
    val_start_row = TRIAL5_INCLINE_7MIN_ROWS - TRIAL5_INCLINE_VAL_30S  # row where val begins
    train_end_win = val_start_row - dataset.window_size                 # last train window index
    val_start_win = val_start_row                                       # first val window index
    val_start     = val_start_row                                       # frame offset for trajectory plots

    train_dataset = Subset(dataset, list(range(train_end_win)))
    val_dataset   = Subset(dataset, list(range(val_start_win, len(dataset))))
    print(f"[FLD] Train: {len(train_dataset):,} windows (first 6.5 min)  |  "
          f"Val: {len(val_dataset):,} windows (last 30s)")

    train_dl = DataLoader(train_dataset, batch_size=config["batch_size"],
                          shuffle=True, num_workers=config["num_workers"],
                          pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_dataset, batch_size=config["batch_size"],
                        shuffle=False, num_workers=config["num_workers"],
                        pin_memory=True, drop_last=False)

    in_mean  = dataset.mean_tensor.to(device)
    in_std   = dataset.std_tensor.to(device)
    out_mean = dataset.out_mean_tensor.to(device)
    out_std  = dataset.out_std_tensor.to(device)

    # ---- Model ---------------------------------------------------------------
    model = FLD(
        observation_dim=dataset.input_dim,
        output_dim=dataset.output_dim,
        history_horizon=config["history_horizon"],
        latent_channel=config["latent_dim"],
        device=device,
        dt=1.0 / 150.0,              # Trial 2 is recorded at ~150 Hz
        encoder_shape=config["encoder_shape"],
        decoder_shape=config["decoder_shape"],
    )
    model.to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=config["lr"],
                           weight_decay=config["weight_decay"])

    plotter     = Plotter()
    latent_dim  = config["latent_dim"]
    history_horizon  = config["history_horizon"]
    forecast_horizon = config["forecast_horizon"]
    noise_level = config["noise_level"]

    # Persistent figures (cleared each plot cycle)
    fig_dist,   ax_dist   = plt.subplots(1, 3, figsize=(15, 4))
    fig_recon,  ax_recon  = plt.subplots(6, 1, figsize=(10, 18))
    fig_params, ax_params = plt.subplots(latent_dim, 5, figsize=(20, latent_dim * 3))
    fig_pca,    ax_pca    = plt.subplots(figsize=(8, 8))
    fig_loss,   ax_loss   = plt.subplots(figsize=(8, 4))
    # Per-channel latent parameter histograms
    fig_hist,   ax_hist   = plt.subplots(latent_dim, 4, figsize=(18, 3.5 * latent_dim))
    # Long trajectory — split into angles and moments, one figure per condition
    _n_angles  = 10
    _n_moments = 10
    fig_long_angle,  ax_long_angle  = plt.subplots(_n_angles,  1, figsize=(30, 2.0 * _n_angles),  sharex=True)
    fig_long_moment, ax_long_moment = plt.subplots(_n_moments, 1, figsize=(30, 2.0 * _n_moments), sharex=True)
    # Latent sine curves — one row per latent channel over the same long trial
    fig_latent, ax_latent = plt.subplots(latent_dim, 1, figsize=(30, 3.5 * latent_dim), sharex=True)

    buf_freq = DistributionBuffer(latent_dim)
    buf_amp  = DistributionBuffer(latent_dim)
    buf_off  = DistributionBuffer(latent_dim)

    train_losses, val_losses = [], []
    gamma       = config.get("gamma", 1.0)
    n_features  = dataset.output_dim

    # Build per-feature weight tensor from config modality weights
    fw = torch.ones(n_features, device=device)
    for modality, indices in MODALITY_GROUPS.items():
        key = "weight_" + modality.split()[0].rstrip("(")   # e.g. "weight_angle"
        w   = config.get(key, 1.0)
        for idx in indices:
            if idx < n_features:
                fw[idx] = w
    feature_weights = fw  # (n_output_features,)

    # ---- CSV loss log --------------------------------------------------------
    csv_path = os.path.join(log_dir, "loss_log.csv")
    diag_k   = [9, 24, forecast_horizon - 1]   # horizon steps to log (0-indexed)
    diag_labels = [f"k{k+1}_rmse" for k in diag_k]
    ph_labels   = [f"phase_err_k{k+1}" for k in diag_k]
    modality_labels = [f"rmse_{m.split()[0]}" for m in MODALITY_GROUPS]
    header = (["epoch", "train_loss_clean", "train_loss_noisy", "val_loss"]
              + diag_labels + ph_labels + modality_labels)
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(header)

    # ---- Training loop -------------------------------------------------------
    print("[FLD] Training started.")
    for epoch in range(config["epochs"]):
        model.train()
        buf_freq.clear(); buf_amp.clear(); buf_off.clear()
        running_loss = 0.0

        for batch_x, batch_y in train_dl:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            win_x = batch_x.unfold(1, history_horizon, 1)   # (B, K+1, features, H)
            win_y = batch_y.unfold(1, history_horizon, 1)

            batch_input = win_x[:, 0, :, :]
            if noise_level > 0.0:
                batch_input = batch_input + torch.randn_like(batch_input) * noise_level

            pred_dynamics, latent, signal, params = model(batch_input, k=forecast_horizon)

            # Stack all K targets: (K, B, features, H)
            tgt_all  = torch.stack([win_y[:, i, :, :] for i in range(forecast_horizon)])
            pred_all = pred_dynamics

            loss = compute_reconstruction_loss(pred_all, tgt_all, gamma, feature_weights)

            optimizer.zero_grad()
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
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                win_x   = batch_x.unfold(1, history_horizon, 1)
                win_y   = batch_y.unfold(1, history_horizon, 1)
                pred_dyn, _, _, _ = model(win_x[:, 0, :, :], k=forecast_horizon)
                tgt_b = torch.stack([win_y[:, i, :, :] for i in range(forecast_horizon)])
                clean_running += compute_reconstruction_loss(
                    pred_dyn, tgt_b, gamma, feature_weights
                ).item()
                n_clean += 1
        train_loss = clean_running / max(n_clean, 1)

        # ---- Validation + diagnostics ----------------------------------------
        model.eval()
        val_running = 0.0
        # Accumulate one full pass for diagnostics
        all_pred, all_tgt, all_inp = [], [], []
        with torch.no_grad():
            for batch_x, batch_y in val_dl:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                win_x   = batch_x.unfold(1, history_horizon, 1)
                win_y   = batch_y.unfold(1, history_horizon, 1)
                inp_b   = win_x[:, 0, :, :]   # (B, input_dim, H) — 48 sensor channels
                pred_dyn, _, _, _ = model(inp_b, k=forecast_horizon)
                tgt_b   = torch.stack([win_y[:, i, :, :] for i in range(forecast_horizon)])
                val_running += compute_reconstruction_loss(
                    pred_dyn, tgt_b, gamma, feature_weights
                ).item()
                # collect first 8 batches for diagnostics (avoid OOM)
                if len(all_pred) < 8:
                    all_pred.append(pred_dyn.cpu())
                    all_tgt.append(tgt_b.cpu())
                    all_inp.append(inp_b.cpu())

        val_loss = val_running / len(val_dl)

        # --- Diagnostic computations on accumulated val batches ---------------
        pred_cat = torch.cat([p for p in all_pred], dim=1)   # (K, B_acc, out_dim, H)
        tgt_cat  = torch.cat([t for t in all_tgt],  dim=1)  # (K, B_acc, out_dim, H)
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
        feat_rmse = per_feature_rmse(pred_cat, tgt_cat, dataset.out_std_tensor)  # (out_features,)
        mod_rmse  = {}
        for modality, indices in MODALITY_GROUPS.items():
            valid = [i for i in indices if i < n_features]
            key   = modality.split()[0].rstrip("(")
            mod_rmse[key] = feat_rmse[valid].mean().item() if valid else 0.0

        # Per-feature RMSE at selected horizons in original units
        # Uses only frame 0 of each predicted window so phase errors are visible
        # (full-window averaging hides phase offsets in periodic signals)
        std_t = dataset.out_std_tensor.clamp(min=1e-6)
        def _feat_rmse_k(k_idx):
            k_idx = min(k_idx, pred_cat.shape[0] - 1)
            err = (pred_cat[k_idx, :, :, 0] - tgt_cat[k_idx, :, :, 0]).pow(2).mean(dim=0).sqrt()  # (F,)
            return (err * std_t).tolist()

        feat_units = ["°"] * 10 + ["N·m"] * 10   # 10 joint angles, 10 joint moments
        K_table   = [0, 1, 2, 3, 4, 5]
        ms_labels = [f"k={k} ({k/150*1000:.0f}ms)" for k in K_table]
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
        header = f"          {'Feature':<{col_w}}" + "".join(f" {lbl:>{cell_w}}" for lbl in ms_labels)
        sep    = f"          {'-'*col_w}" + "".join(f" {'-'*cell_w}" for _ in ms_labels)
        print(header)
        print(sep)
        for fi, (name, unit) in enumerate(zip(OUTPUT_COLUMNS, feat_units)):
            row_str = f"          {name[:col_w]:<{col_w}}"
            for rmse_k in rmse_table:
                row_str += f" {rmse_k[fi]:>{cell_w-2}.3f}{unit} "
            print(row_str)

        # Append to CSV  (train_loss = clean eval pass; train_loss_noisy = noisy train pass)
        row = ([epoch + 1, train_loss, train_loss_noisy, val_loss]
               + diag_k_vals + phase_errs
               + [mod_rmse[m.split()[0].rstrip("(")] for m in MODALITY_GROUPS])
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        # All plots fire once at the end of training only
        if epoch == config["epochs"] - 1:
            with torch.no_grad():
                model.eval()
                _plot_loss_curve(fig_loss, ax_loss, train_losses, val_losses)
                _plot_epoch(
                    plotter, model, dataset, device,
                    fig_dist, ax_dist,
                    fig_recon, ax_recon,
                    fig_params, ax_params,
                    fig_pca, ax_pca,
                    buf_freq, buf_amp, buf_off,
                    latent_dim, history_horizon, forecast_horizon,
                    epoch, dataset.input_dim, dataset.output_dim,
                )
                _plot_parameter_histograms(
                    fig_hist, ax_hist,
                    buf_freq, buf_amp, buf_off,
                    latent_dim, epoch,
                )
                # Incline walking validation (last 30s of Trial 5)
                _plot_long_trajectory(
                    model, dataset, device,
                    history_horizon, fig_long_angle, ax_long_angle,
                    n_blocks=90, forecast_horizon=forecast_horizon,
                    sample_rate=150.0, epoch=epoch,
                    start_frame=val_start, feature_slice=slice(0, 10),
                )
                _plot_long_trajectory(
                    model, dataset, device,
                    history_horizon, fig_long_moment, ax_long_moment,
                    n_blocks=90, forecast_horizon=forecast_horizon,
                    sample_rate=150.0, epoch=epoch,
                    start_frame=val_start, feature_slice=slice(10, 20),
                )
                _plot_latent_sine_trajectory(
                    model, dataset, device,
                    history_horizon, fig_latent, ax_latent,
                    n_blocks=90, forecast_horizon=forecast_horizon,
                    sample_rate=150.0, epoch=epoch,
                    start_frame=val_start,
                )
                model.train()

        # Checkpoint
        if epoch % config["save_every"] == 0:
            _save(model, optimizer, epoch, config, dataset, log_dir)

    _save(model, optimizer, config["epochs"], config, dataset, log_dir)
    print(f"[FLD] Training finished. Checkpoints in: {log_dir}")
    plt.ioff()
    plt.show()  # keep all figures open after training completes


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_loss_curve(fig, ax, train_losses, val_losses):
    ax.cla()
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train")
    ax.plot(epochs, val_losses,   label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("FLD Training Loss")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.canvas.draw()
    plt.pause(0.01)


def _show(fig):
    fig.tight_layout()
    fig.canvas.draw()
    plt.pause(0.05)


def _plot_epoch(plotter, model, dataset, device,
                fig_dist, ax_dist,
                fig_recon, ax_recon,
                fig_params, ax_params,
                fig_pca, ax_pca,
                buf_freq, buf_amp, buf_off,
                latent_dim, history_horizon, forecast_horizon,
                epoch, input_dim, output_dim):

    # --- Distribution plots --------------------------------------------------
    plotter.plot_distribution(ax_dist[0], buf_freq.get(), title="Frequency Distribution")
    plotter.plot_distribution(ax_dist[1], buf_amp.get(),  title="Amplitude Distribution")
    plotter.plot_distribution(ax_dist[2], buf_off.get(),  title="Offset Distribution")
    _show(fig_dist)

    # --- Encode the first window in the dataset ------------------------------
    eval_x, eval_y = dataset[0]
    eval_x = eval_x.unsqueeze(0).to(device)           # (1, W, input_dim)
    eval_y = eval_y.unsqueeze(0).to(device)           # (1, W, output_dim)
    win_x  = eval_x.unfold(1, history_horizon, 1)     # (1, F+1, input_dim, H)
    win_y  = eval_y.unfold(1, history_horizon, 1)     # (1, F+1, output_dim, H)
    eval_input  = win_x[:, 0, :, :]                   # (1, input_dim, H)
    eval_target = win_y[:, 0, :, :]                   # (1, output_dim, H)

    pred_dynamics, latent, signal, params = model(eval_input, k=forecast_horizon)

    # --- Reconstruction overview (6 rows) ------------------------------------
    plotter.plot_curves(ax_recon[0], eval_input[0], -1.0, 1.0, -5.0, 5.0,
                        title=f"Sensor Input ({input_dim} channels × {history_horizon} steps)", show_axes=False)
    plotter.plot_curves(ax_recon[1], latent[0], -1.0, 1.0, -2.0, 2.0,
                        title=f"Latent Embedding ({latent_dim}×{history_horizon})", show_axes=False)
    plotter.plot_circles(ax_recon[2], params[0][0], params[2][0],
                         title=f"Learned Phase Timing ({latent_dim} channels)", show_axes=False)
    plotter.plot_curves(ax_recon[3], signal[0], -1.0, 1.0, -2.0, 2.0,
                        title=f"Parametrised Fourier Signal ({latent_dim}×{history_horizon})", show_axes=False)
    plotter.plot_curves(ax_recon[4], pred_dynamics[0][0], -1.0, 1.0, -5.0, 5.0,
                        title=f"Reconstructed Output ({output_dim} channels × {history_horizon} steps)", show_axes=False)
    plotter.plot_curves(
        ax_recon[5],
        torch.vstack((eval_target[0].flatten(0, 1).unsqueeze(0),
                      pred_dynamics[0][0].flatten(0, 1).unsqueeze(0))),
        -1.0, 1.0, -5.0, 5.0,
        title="Target vs Reconstruction (flattened)", show_axes=False,
    )
    _show(fig_recon)

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
    _show(fig_params)

    # --- PCA phase manifold — one clean gait cycle ---------------------------
    # At 150 Hz, one gait cycle ≈ 150 frames → 150 consecutive windows.
    # Start at frame 500 to skip any trial startup transient.
    manifold_start   = 500
    manifold_windows = 500   # ~3.3 s covering several gait cycles
    manifold_data = dataset.data_in[
        manifold_start : manifold_start + history_horizon + manifold_windows
    ].to(device)
    seqs = torch.stack([manifold_data[i : i + history_horizon].T
                        for i in range(manifold_windows)])   # (150, in_dim, H)
    _, _, _, m_params = model(seqs)
    phase_m     = m_params[0]   # (150, latent_dim)
    amplitude_m = m_params[2]
    manifold = torch.hstack((
        amplitude_m * torch.sin(2.0 * torch.pi * phase_m),
        amplitude_m * torch.cos(2.0 * torch.pi * phase_m),
    ))
    plotter.plot_pca(ax_pca, [manifold.cpu()],
                     title="Phase Manifold — 500 windows (~3.3 s, training data)")
    _show(fig_pca)



# ---------------------------------------------------------------------------
# Per-channel latent parameter histograms
# ---------------------------------------------------------------------------

def _plot_parameter_histograms(fig, axes, buf_freq, buf_amp, buf_off,
                                latent_dim, epoch):
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
    import numpy as np
    freq_data  = buf_freq.get().numpy()   # (N, latent_dim)
    amp_data   = buf_amp.get().numpy()
    off_data   = buf_off.get().numpy()

    cmap = plt.cm.tab10
    # At 150 Hz, window=151: 1 cycle/window ≈ 0.99 Hz, 2 cycles/window ≈ 1.99 Hz
    freq_1hz = 150.0 / 151.0   # ≈ 1 Hz in cycles/window units
    freq_2hz = 2.0 * freq_1hz  # ≈ 2 Hz (step frequency)

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
    _show(fig)


# ---------------------------------------------------------------------------
# Latent sine curves over the long trial
# ---------------------------------------------------------------------------

def _plot_latent_sine_trajectory(model, dataset, device,
                                  history_horizon, fig, axes,
                                  n_blocks=200, forecast_horizon=50,
                                  sample_rate=150.0, epoch=None,
                                  batch_size=64, start_frame=0):
    """
    For each block encode the context window and compute the latent sine curves.
    Pass start_frame=val_start to plot over validation data only.
    """
    data = dataset.data_in
    N, D = data.shape
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

            k_steps = torch.arange(K, dtype=torch.float32)              # (K,)
            for bi in range(b_end - b_start):
                # phase at each of the K future steps: (K, latent_dim)
                phase_k = phase[bi].unsqueeze(0) + freq[bi].unsqueeze(0) * dt * k_steps.unsqueeze(1)
                sine_k  = amp[bi] * torch.sin(2 * torch.pi * phase_k) + offset[bi]
                sine_blocks.append(sine_k.numpy())          # (K, latent_dim)

    sine_arr = np.concatenate(sine_blocks, axis=0)          # (n_blocks*K, latent_dim)
    n_frames = sine_arr.shape[0]
    t        = np.arange(n_frames) / sample_rate
    cmap     = plt.cm.tab10

    ax_list = np.atleast_1d(axes)
    for ch, ax in enumerate(ax_list):
        ax.cla()
        ax.plot(t, sine_arr[:, ch], color=cmap(ch % 10), lw=0.6, alpha=0.85)
        for blk in range(1, n_blocks):
            ax.axvline(blk * K / sample_rate, color="grey", lw=0.4, ls=":", alpha=0.35)
        ax.set_ylabel(f"ch {ch}", fontsize=9, rotation=0, labelpad=32, va="center")
        ax.grid(alpha=0.15)
        ax.tick_params(labelsize=7)

    ax_list[-1].set_xlabel("Time (s)")
    epoch_str = f"[Epoch {epoch+1}]  " if epoch is not None else ""
    fig.suptitle(
        f"{epoch_str}Latent Sine Curves — A·sin(2π·(φ + f·dt·k)) + b per channel\n"
        f"{latent_dim} channels  |  {n_blocks} blocks × {K} frames ({n_blocks*K/sample_rate:.1f}s)  |  "
        f"vertical lines = block boundaries",
        fontsize=9, fontweight="bold",
    )
    _show(fig)


# ---------------------------------------------------------------------------
# Long trajectory — GT vs prediction stitched over many windows
# ---------------------------------------------------------------------------

def _plot_long_trajectory(model, dataset, device,
                           history_horizon, fig, axes,
                           n_blocks=200, forecast_horizon=50,
                           sample_rate=150.0, epoch=None,
                           batch_size=64, start_frame=0,
                           feature_slice=None):
    """
    Step through the data in non-overlapping forecast_horizon-sized blocks,
    beginning at start_frame (pass val_start to plot validation data only).
    feature_slice: slice object selecting which output features to plot,
                   e.g. slice(0,10) for angles, slice(10,20) for moments.
    Blue = ground truth.  Red dashed = model prediction.
    """
    data_in  = dataset.data_in           # (N, in_dim) normalised — fed to encoder
    data_out = dataset.data_out          # (N, out_dim) normalised — GT for decoder
    std_np   = dataset.out_std_tensor.numpy()
    mean_np  = dataset.out_mean_tensor.numpy()
    N        = data_in.shape[0]
    D        = data_out.shape[1]
    H        = history_horizon
    K        = forecast_horizon

    # Maximum non-overlapping blocks available from start_frame onward
    max_blocks = (N - start_frame - H) // K
    n_blocks   = min(n_blocks, max_blocks)

    pred_traj, gt_traj = [], []   # each entry: (K, D) for one block

    model.eval()
    with torch.no_grad():
        for b_start in range(0, n_blocks, batch_size):
            b_end   = min(b_start + batch_size, n_blocks)
            ctx_starts = [start_frame + i * K for i in range(b_start, b_end)]
            wins = torch.stack(
                [data_in[s : s + H].T for s in ctx_starts]
            ).to(device)                                      # (B, in_dim, H)

            pred_dyn, _, _, _ = model(wins, k=K)
            # pred_dyn: (K, B, D, H) — take first frame of each predicted window
            # pred_dyn[k][b, :, 0] = predicted frame k steps ahead for sample b
            pred_block = torch.stack(
                [pred_dyn[k][:, :, 0] for k in range(K)], dim=1
            ).cpu().numpy()                                   # (B, K, D)

            for bi, s in enumerate(ctx_starts):
                pred_traj.append(pred_block[bi])              # (K, D)
                # GT must match what pred_dyn[k] is predicting:
                # pred_dyn[k][:,  :, 0] = first frame of window starting at s+k
                # so the matching GT frame is data[s+k]
                gt_traj.append(data_out[s : s + K].numpy())  # (K, out_dim)

    pred_arr = np.concatenate(pred_traj, axis=0)   # (n_blocks*K, out_dim)
    gt_arr   = np.concatenate(gt_traj,   axis=0)

    pred_orig = pred_arr * std_np + mean_np
    gt_orig   = gt_arr   * std_np + mean_np

    # Apply feature slice for this subplot group
    fs = feature_slice if feature_slice is not None else slice(None)
    pred_plot = pred_orig[:, fs]
    gt_plot   = gt_orig[:,   fs]
    col_names = OUTPUT_COLUMNS[fs] if feature_slice is not None else OUTPUT_COLUMNS
    all_units = ["°"] * 10 + ["N·m"] * 10
    units_plot = (all_units[fs] if isinstance(fs, slice)
                  else [all_units[i] for i in fs])
    group_label = ("Joint Angles (°)" if feature_slice == slice(0, 10)
                   else "Joint Moments (N·m)" if feature_slice == slice(10, 20)
                   else "Output Features")

    n_frames = pred_plot.shape[0]
    t        = np.arange(n_frames) / sample_rate

    for fi, ax in enumerate(axes):
        ax.cla()
        ax.plot(t, gt_plot[:, fi],   color="steelblue", lw=0.6, alpha=0.85, label="GT")
        ax.plot(t, pred_plot[:, fi], color="tomato",    lw=0.6, alpha=0.85, ls="--", label="Pred")
        for blk in range(1, n_blocks):
            ax.axvline(blk * K / sample_rate, color="grey", lw=0.4, ls=":", alpha=0.35)
        ax.set_ylabel(f"{col_names[fi][:20]}\n({units_plot[fi]})",
                      fontsize=7, rotation=0, labelpad=95, va="center")
        ax.grid(alpha=0.15)
        ax.tick_params(labelsize=6)
        if fi == 0:
            ax.legend(fontsize=7, loc="upper right", ncol=2)

    axes[-1].set_xlabel("Time (s)")
    epoch_str = f"[Epoch {epoch+1}]  " if epoch is not None else ""
    fig.suptitle(
        f"{epoch_str}Long Trajectory — {group_label} — GT (blue) vs Prediction (red dashed)\n"
        f"{n_blocks} blocks × {K} frames = {n_frames:,} frames ({n_frames/sample_rate:.1f}s)  |  "
        f"each block = one fresh {K/sample_rate*1000:.0f}ms prediction from a new context  |  "
        f"vertical lines = block boundaries  |  physical units",
        fontsize=9, fontweight="bold",
    )
    _show(fig)


# ---------------------------------------------------------------------------
# Future prediction vs ground truth
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(model, optimizer, epoch, config, dataset, log_dir):
    path = os.path.join(log_dir, f"fld_model_epoch_{epoch}.pt")
    torch.save(
        {
            "fld_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "in_mean": dataset.mean_tensor,
            "in_std":  dataset.std_tensor,
            "out_mean": dataset.out_mean_tensor,
            "out_std":  dataset.out_std_tensor,
        },
        path,
    )
    print(f"[FLD] Checkpoint saved → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # NOTE: history_horizon must be ODD for FLD's Conv1d same-size padding.
    # 151 is the nearest odd integer to the requested 150.
    config = {
        # Data is Trial 2 at ~150 Hz — all walking, no trimming needed
        "history_horizon": 151,     # ~1.0 s of context at 150 Hz (must be odd for FLD Conv1d)
        "forecast_horizon": 50,     # ~0.33 s ahead at 150 Hz
        "latent_dim": 8,             # Fourier latent channels — increased for cross-modal task
        "encoder_shape": [64, 64],   # larger encoder to handle 48→8 cross-modal compression
        "decoder_shape": [64, 64],   # larger decoder to expand 8→20 joint outputs
        "feature_set": "cross",      # input: 48 IMU/insole cols → output: 20 joint angle+moment cols
        # Loss settings
        "gamma":        0.95,        # horizon discount
        "weight_angle":  1.0,        # relative weight for joint angles (°)
        "weight_moment": 1.0,        # relative weight for joint moments (N·m)
        "batch_size": 32,
        "num_workers": 4,
        "lr": 1e-4,
        "weight_decay": 5e-4,
        "noise_level": 0.05,
        "epochs": 10,
        "save_every": 10,
    }

    train(config)


if __name__ == "__main__":
    main()
