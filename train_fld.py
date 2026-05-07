"""
train_fld.py

Training script for the FLD (Fourier Latent Dynamics) model applied to
biomechanical sensor data.

Phase 1 – latent space learning:
  Input  = EMG (8) + joint angles/IK (10) + joint moments/ID (10)  →  28 features
  Output = same 28 features, supervised against 50 future shifted windows

The FLD learns a compact Fourier latent representation (phase, frequency,
amplitude, offset per channel) from sliding windows of biomechanical signals.
It encodes history_horizon=151 past timesteps and is trained to predict
forecast_horizon=50 future shifted windows of the same signals.

Run:
    python3 train_fld.py

Outputs are saved to runs/FLD_<timestamp>/:
  - loss_log.csv          training and validation loss per epoch
  - fld_model_epoch_N.pt  model checkpoints
  - plots/                PNG figures saved every plot_every epochs
      dist_epoch_N.png       frequency / amplitude / offset distributions
      reconstruction_N.png   input vs latent signal vs reconstruction
      channel_params_N.png   per-channel Fourier params (phase, freq, amp, offset)
      phase_manifold_N.png   PCA of the phase manifold
      loss_curve.png         train / val loss curve (updated each epoch)
"""

import os
import sys
import csv
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from DataLoader.data_loader import FLDDataset
from Models.FLD import FLD
from Library.fld_plotter import Plotter

# ---------------------------------------------------------------------------
# Paths & column definitions
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(os.path.dirname(__file__), "Data", "trial_1_all_data.csv")

COLUMNS = [
    # IMU (42 features – 7 locations × 6 axes)
    'pelvis_acc_x', 'pelvis_acc_y', 'pelvis_acc_z', 'pelvis_gyro_x', 'pelvis_gyro_y', 'pelvis_gyro_z',
    'thigh_r_acc_x', 'thigh_r_acc_y', 'thigh_r_acc_z', 'thigh_r_gyro_x', 'thigh_r_gyro_y', 'thigh_r_gyro_z',
    'shank_r_acc_x', 'shank_r_acc_y', 'shank_r_acc_z', 'shank_r_gyro_x', 'shank_r_gyro_y', 'shank_r_gyro_z',
    'R_insole_accel_x', 'R_insole_accel_y', 'R_insole_accel_z', 'R_insole_gyro_x', 'R_insole_gyro_y', 'R_insole_gyro_z',
    'thigh_l_acc_x', 'thigh_l_acc_y', 'thigh_l_acc_z', 'thigh_l_gyro_x', 'thigh_l_gyro_y', 'thigh_l_gyro_z',
    'shank_l_acc_x', 'shank_l_acc_y', 'shank_l_acc_z', 'shank_l_gyro_x', 'shank_l_gyro_y', 'shank_l_gyro_z',
    'L_insole_accel_x', 'L_insole_accel_y', 'L_insole_accel_z', 'L_insole_gyro_x', 'L_insole_gyro_y', 'L_insole_gyro_z',
    # Force insole (6)
    'R_insole_force', 'L_insole_force',
    'R_insole_COPx', 'R_insole_COPz', 'L_insole_COPx', 'L_insole_COPz',
    # Filtered EMG (8)
    'R_RF', 'R_BF', 'R_TA', 'R_GAST',
    'L_RF', 'L_BF', 'L_TA', 'L_GAST',
    # Joint kinematics – IK (10)
    'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r',
    'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 'ankle_angle_l',
    # Joint dynamics – ID (10)
    'hip_flexion_r_moment', 'hip_adduction_r_moment', 'hip_rotation_r_moment',
    'knee_angle_r_moment', 'ankle_angle_r_moment',
    'hip_flexion_l_moment', 'hip_adduction_l_moment', 'hip_rotation_l_moment',
    'knee_angle_l_moment', 'ankle_angle_l_moment',
]

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

def compute_reconstruction_loss(pred, target, mean, std):
    """MSE in the original (un-normalised) scale.

    pred / target: (B, obs_dim, H) normalised
    mean / std:    (obs_dim,) on the same device
    """
    m = mean.unsqueeze(0).unsqueeze(-1)   # (1, obs_dim, 1)
    s = std.unsqueeze(0).unsqueeze(-1)
    return torch.mean(torch.square((pred * s + m) - (target * s + m)))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FLD] Using device: {device}")

    log_dir = os.path.join("runs", f"FLD_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    plt.ion()  # interactive mode: figures update without blocking training

    # ---- Data ----------------------------------------------------------------
    print("[FLD] Loading dataset …")
    df = pd.read_csv(DATA_PATH)
    if config["trim_to_walking"]:
        df = df.iloc[:config["walking_rows"]].reset_index(drop=True)
        print(f"[FLD] Trimmed to first {config['walking_rows']} rows (walking only)")
    df_features = df[COLUMNS]

    dataset = FLDDataset(
        df_features,
        history_horizon=config["history_horizon"],
        forecast_horizon=config["forecast_horizon"],
        feature_set=config["feature_set"],
    )
    print(f"[FLD] Dataset: {len(dataset)} windows, obs_dim={dataset.obs_dim}")

    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(42))

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

    buf_freq = DistributionBuffer(latent_dim)
    buf_amp  = DistributionBuffer(latent_dim)
    buf_off  = DistributionBuffer(latent_dim)

    train_losses, val_losses = [], []

    # ---- CSV loss log --------------------------------------------------------
    csv_path = os.path.join(log_dir, "loss_log.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss"])

    # ---- Training loop -------------------------------------------------------
    print("[FLD] Training started.")
    for epoch in range(config["epochs"]):
        model.train()
        buf_freq.clear(); buf_amp.clear(); buf_off.clear()
        running_loss = 0.0

        for batch_x, batch_y in train_dl:
            batch_x = batch_x.to(device)   # (B, W, input_dim)
            batch_y = batch_y.to(device)   # (B, W, output_dim)

            # unfold time dim → (B, F+1, dim, H)
            win_x = batch_x.unfold(1, history_horizon, 1)  # (B, F+1, input_dim,  H)
            win_y = batch_y.unfold(1, history_horizon, 1)  # (B, F+1, output_dim, H)

            batch_input = win_x[:, 0, :, :]   # (B, input_dim, H)
            if noise_level > 0.0:
                batch_input = batch_input + torch.randn_like(batch_input) * noise_level

            pred_dynamics, latent, signal, params = model(batch_input, k=forecast_horizon)
            # pred_dynamics: (forecast_horizon, B, output_dim, H)

            loss = torch.tensor(0.0, device=device)
            for i in range(forecast_horizon):
                loss = loss + compute_reconstruction_loss(
                    pred_dynamics[i], win_y[:, i, :, :], out_mean, out_std
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            phase, frequency, amplitude, offset = params
            buf_freq.insert(frequency)
            buf_amp.insert(amplitude)
            buf_off.insert(offset)

        train_loss = running_loss / len(train_dl)

        # Validation
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_dl:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                win_x = batch_x.unfold(1, history_horizon, 1)
                win_y = batch_y.unfold(1, history_horizon, 1)
                pred_dynamics, _, _, _ = model(win_x[:, 0, :, :], k=forecast_horizon)
                loss = torch.tensor(0.0, device=device)
                for i in range(forecast_horizon):
                    loss = loss + compute_reconstruction_loss(
                        pred_dynamics[i], win_y[:, i, :, :], out_mean, out_std
                    )
                val_running += loss.item()
        val_loss = val_running / len(val_dl)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"[FLD] Epoch [{epoch + 1:>4}/{config['epochs']}]  "
              f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        # Append to CSV
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, train_loss, val_loss])

        # Update loss curve every epoch
        _plot_loss_curve(fig_loss, ax_loss, train_losses, val_losses)

        # Periodic diagnostic plots
        if epoch % config["plot_every"] == 0:
            with torch.no_grad():
                model.eval()
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

    # --- PCA phase manifold --------------------------------------------------
    manifold_windows = 500
    # Stack 500 sliding windows from the start of the dataset
    manifold_data = dataset.data_in[:history_horizon + manifold_windows].to(device)
    seqs = torch.stack([manifold_data[i:i + history_horizon].T
                        for i in range(manifold_windows)])   # (500, obs_dim, H)
    _, _, _, m_params = model(seqs)
    phase_m     = m_params[0]   # (500, latent_dim)
    amplitude_m = m_params[2]
    manifold = torch.hstack((
        amplitude_m * torch.sin(2.0 * torch.pi * phase_m),
        amplitude_m * torch.cos(2.0 * torch.pi * phase_m),
    ))
    plotter.plot_pca(ax_pca, [manifold.cpu()], title="Phase Manifold (PCA)")
    _show(fig_pca)


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
        "history_horizon": 151,     # ≈ 150 past samples (must be odd)
        "forecast_horizon": 50,     # predict 50 steps into the future
        "latent_dim": 4,            # number of Fourier latent channels
        "encoder_shape": [64, 64],
        "decoder_shape": [64, 64],
        "feature_set": "output",    # 'output' → EMG+angles+moments (28→28) self-supervised latent learning
        "trim_to_walking": True,    # use only the first walking_rows rows
        "walking_rows": 40000,      # ~40k rows covers steady-state walking
        "batch_size": 32,
        "num_workers": 4,
        "lr": 1e-4,
        "weight_decay": 5e-4,
        "noise_level": 0.05,
        "epochs": 25,
        "plot_every": 1,           # save diagnostic plots every N epochs
        "save_every": 1,           # save model checkpoint every N epochs
    }

    train(config)


if __name__ == "__main__":
    main()
