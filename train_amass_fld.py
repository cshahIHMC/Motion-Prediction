"""
train_amass_fld.py

Train the FLD model on CMU AMASS lower-body motion data (walk / run / jog).

Input:  7 lower-body joints × 3 Euler angles (ZYX, degrees) = 21 features/frame
        Pelvis and global translation excluded — local rotations only.
        Captured at 120 Hz.

FLD config:
    history_horizon = 121  (nearest odd to 120 — FLD Conv1d requires odd)
    forecast_horizon = 60  (0.5 s of future prediction)
    latent_dim = 4

Run:
    python3 train_amass_fld.py
"""

import os
import sys
import csv
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from DataLoader.amass_loader import (
    CMUAMASSDataset, LOWER_BODY_JOINT_NAMES,
    _CLIP_CATALOG, load_clean_clips,
)
from Models.FLD import FLD
from Library.fld_plotter import Plotter
from sklearn.decomposition import PCA as _PCA  # used in _plot_manifold_scatter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

AMASS_ROOT = os.path.join(os.path.dirname(__file__), "Data", "AMASS Dataset", "CMU", "CMU")
MOTION_TYPES = ["walk", "run", "jog", "skip", "varied_terrain"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AMASSFLDDataset(Dataset):
    """
    Sliding-window dataset built from CMU AMASS lower-body clips.

    Each item is a tuple (x, x) of shape (window_size, 21) — self-supervised.
    Windows are clipped to individual motion clips so no window crosses a
    clip boundary.

    history_horizon must be odd (FLD Conv1d constraint).
    """

    def __init__(
        self,
        root_path: str,
        history_horizon: int = 121,
        forecast_horizon: int = 60,
        motion_types=None,
        euler_sequence: str = "ZYX",
    ):
        assert history_horizon % 2 == 1, (
            f"history_horizon must be odd for FLD Conv1d (got {history_horizon}). "
            f"Use {history_horizon + 1} or {history_horizon - 1}."
        )
        self.history_horizon  = history_horizon
        self.forecast_horizon = forecast_horizon
        self.window_size      = history_horizon + forecast_horizon
        self.obs_dim          = 21
        self.input_dim        = 21
        self.output_dim       = 21

        # --- load all clips (already Euler-converted) ----------------------
        base = CMUAMASSDataset(
            root_path=root_path,
            motion_types=motion_types,
            lower_body_only=True,
            euler_sequence=euler_sequence,
        )
        print(f"[Dataset] {len(base)} clips loaded.")

        # --- global normalisation ------------------------------------------
        print("[Dataset] Computing global mean/std …")
        all_frames = np.concatenate(
            [base[i]["poses"].numpy() for i in range(len(base))], axis=0
        )  # (N_total, 21)
        mean = all_frames.mean(axis=0).astype(np.float32)
        std  = all_frames.std(axis=0).astype(np.float32)
        std[std == 0] = 1.0

        self.mean_tensor     = torch.from_numpy(mean)
        self.std_tensor      = torch.from_numpy(std)
        self.out_mean_tensor = self.mean_tensor.clone()
        self.out_std_tensor  = self.std_tensor.clone()

        # --- normalise + build per-clip window list ------------------------
        print("[Dataset] Building windows …")
        self.windows: list = []   # (clip_tensor, start_idx)
        norm_clips = []
        self.clip_window_counts: list = []   # windows contributed by each clip
        self.clip_labels: list = []          # (subject, clip_name) per clip

        for i in range(len(base)):
            raw  = base[i]["poses"].numpy()              # (T, 21)
            norm = torch.from_numpy((raw - mean) / std)  # (T, 21)
            norm_clips.append(norm)
            self.clip_labels.append(
                (base.clips[i]["subject"], base.clips[i]["clip_name"])
            )
            T = norm.shape[0]
            n_windows_this_clip = max(0, T - self.window_size + 1)
            self.clip_window_counts.append(n_windows_this_clip)
            for start in range(n_windows_this_clip):
                self.windows.append((norm, start))

        # Expose individual normalised clips for per-clip diagnostics
        self.norm_clips = norm_clips

        # data_in used by phase evolution — first N normalised frames
        self.data_in = torch.cat(norm_clips, dim=0)[:20000]  # (<=20000, 21)

        total_skipped = sum(
            1 for c in norm_clips if c.shape[0] < self.window_size
        )
        print(f"[Dataset] {len(self.windows):,} windows across {len(norm_clips)} clips  "
              f"({total_skipped} clips too short, skipped)")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx):
        clip_t, start = self.windows[idx]
        w = clip_t[start : start + self.window_size]   # (W, 21)
        return w, w   # self-supervised


# ---------------------------------------------------------------------------
# Distribution buffer (collects Fourier params across batches for plotting)
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
    Loss in NORMALISED space, averaged over K windows.
    pred_all, target_all : (K, B, features, H)
    gamma                : per-step discount (default 1.0 = off)
    feature_weights      : (features,) or None
    """
    K      = pred_all.shape[0]
    total  = torch.zeros(1, device=pred_all.device)
    weight = 1.0
    for i in range(K):
        sq_err = (pred_all[i] - target_all[i]).pow(2)
        if feature_weights is not None:
            sq_err = sq_err * feature_weights.view(1, -1, 1)
        total  = total + sq_err.mean() * (gamma ** i)
        weight = weight + (gamma ** i)
    return total / weight


def per_feature_rmse(pred_all, target_all, std):
    """RMSE in original units per feature. Returns (features,)."""
    mse_norm = (pred_all - target_all).pow(2).mean(dim=(0, 1, 3))
    return mse_norm.sqrt() * std.clamp(min=1e-6).to(pred_all.device)


def rmse_by_horizon(pred_all, target_all):
    """RMSE in normalised space per forecast step. Returns (K,)."""
    return (pred_all - target_all).pow(2).mean(dim=(1, 2, 3)).sqrt()


def phase_propagation_error(model, window_t, window_tk, dt, k):
    """Mean circular phase error in radians at forecast step k."""
    with torch.no_grad():
        _, _, _, params_t  = model(window_t)
        _, _, _, params_tk = model(window_tk)
    phi_pred   = params_t[0] + params_t[1] * dt * k
    phi_actual = params_tk[0]
    diff       = phi_pred - phi_actual
    circular   = torch.atan2(
        torch.sin(2.0 * torch.pi * diff),
        torch.cos(2.0 * torch.pi * diff),
    )
    return circular.abs().mean().item()


def compute_diversity_loss(frequency):
    """
    Penalise latent channels for having the same frequency.

    frequency : (B, latent_dim)  — one dominant frequency per channel per sample,
                                   extracted by FLD's FFT step.

    HOW IT WORKS
    ------------
    For each sample in the batch we compute ALL pairwise absolute differences
    between channel frequencies:

        |freq_i  -  freq_j|   for every pair  i != j

    Example with 4 channels at [1.3, 1.3, 1.3, 1.3]:
        all differences = 0.0  →  loss = 0  →  no gradient push

    Example with 4 channels at [0.5, 1.0, 1.5, 2.0]:
        differences range 0.5–1.5  →  loss is large and negative
        minimising this term pushes channels further apart

    We return the NEGATIVE mean pairwise distance so that minimising
    the total loss (which includes this term) maximises channel spread.
    """
    B, L = frequency.shape

    # Expand to compute all pairs: (B, L, 1) vs (B, 1, L) → (B, L, L)
    freq_i = frequency.unsqueeze(2)   # channel i repeated across columns
    freq_j = frequency.unsqueeze(1)   # channel j repeated across rows
    pairwise_dist = (freq_i - freq_j).abs()   # (B, L, L)

    # Mask out the diagonal (distance of channel with itself = 0, not useful)
    mask = (1 - torch.eye(L, device=frequency.device)).unsqueeze(0)  # (1, L, L)

    # Mean over all off-diagonal pairs and over the batch, then negate
    # → minimising this pushes pairwise distances UP
    mean_dist = (pairwise_dist * mask).sum() / (B * L * (L - 1))
    return -mean_dist


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FLD] Device: {device}")

    log_dir = os.path.join("runs", f"FLD_AMASS_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    # plt.ion()  # plotting disabled during training — use inspect_amass_fld.py

    # ---- Dataset -------------------------------------------------------------
    dataset = AMASSFLDDataset(
        root_path=AMASS_ROOT,
        history_horizon=config["history_horizon"],
        forecast_horizon=config["forecast_horizon"],
        motion_types=MOTION_TYPES,
        euler_sequence="ZYX",
    )

    # Clip-level split — first 85% of clips → train, last 15% → val.
    # Windows from different clips are different recordings and cannot overlap.
    n_clips       = len(dataset.clip_window_counts)
    n_train_clips = int(0.85 * n_clips)
    train_end     = sum(dataset.clip_window_counts[:n_train_clips])
    val_start     = train_end   # no gap needed — different clips
    train_ds = Subset(dataset, list(range(train_end)))
    val_ds   = Subset(dataset, list(range(val_start, len(dataset))))

    train_dl = DataLoader(train_ds, batch_size=config["batch_size"],
                          shuffle=True, num_workers=config["num_workers"],
                          pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=config["batch_size"],
                          shuffle=False, num_workers=config["num_workers"],
                          pin_memory=True, drop_last=False)

    in_mean  = dataset.mean_tensor.to(device)
    in_std   = dataset.std_tensor.to(device)
    out_mean = dataset.out_mean_tensor.to(device)
    out_std  = dataset.out_std_tensor.to(device)

    print(f"[FLD] Clip-level split — train: {n_train_clips}/{n_clips} clips "
          f"({len(train_ds):,} windows)  |  val: {n_clips - n_train_clips} clips "
          f"({len(val_ds):,} windows)")

    # ---- Model ---------------------------------------------------------------
    model = FLD(
        observation_dim=dataset.input_dim,
        output_dim=dataset.output_dim,
        history_horizon=config["history_horizon"],
        latent_channel=config["latent_dim"],
        device=device,
        dt=1.0 / 120.0,              # AMASS CMU is 120 Hz
        encoder_shape=config["encoder_shape"],
        decoder_shape=config["decoder_shape"],
    )
    model.to(device)

    optimizer = optim.Adam(model.parameters(),
                           lr=config["lr"],
                           weight_decay=config["weight_decay"])

    latent_dim       = config["latent_dim"]
    history_horizon  = config["history_horizon"]
    forecast_horizon = config["forecast_horizon"]
    noise_level      = config["noise_level"]

    plotter = Plotter()

    # Persistent figures — all shown at the last epoch only
    fig_dist,       ax_dist       = plt.subplots(1, 3, figsize=(15, 4))
    fig_recon,      ax_recon      = plt.subplots(6, 1, figsize=(10, 18))
    fig_params,     ax_params     = plt.subplots(latent_dim, 5, figsize=(20, latent_dim * 3))
    fig_pca,        ax_pca        = plt.subplots(1, _N_MANIFOLD_CLIPS, figsize=(5 * _N_MANIFOLD_CLIPS, 5))
    # Clean manifold — one subplot per activity type using exclusively-labeled clips
    fig_loss,       ax_loss       = plt.subplots(figsize=(8, 4))
    fig_phase_evo,  ax_phase_evo  = plt.subplots(latent_dim, 1, figsize=(14, latent_dim * 2), sharex=True)
    fig_hist,       ax_hist       = plt.subplots(latent_dim, 3, figsize=(15, latent_dim * 2))
    fig_fourier_ch, ax_fourier_ch = plt.subplots(latent_dim, 1, figsize=(14, latent_dim * 2), sharex=True)
    fig_pred,       ax_pred       = plt.subplots(6, 1, figsize=(14, 3.5 * 6), sharex=False)
    # Long trajectory and latent sine — same diagnostics as train_fld.py
    _n_feat = dataset.obs_dim   # 21
    fig_long,   ax_long   = plt.subplots(_n_feat, 1, figsize=(30, 2.0 * _n_feat), sharex=True)
    fig_latent, ax_latent = plt.subplots(latent_dim, 1, figsize=(30, 3.5 * latent_dim), sharex=True)

    buf_freq = DistributionBuffer(latent_dim)
    buf_amp  = DistributionBuffer(latent_dim)
    buf_off  = DistributionBuffer(latent_dim)

    train_losses, val_losses = [], []

    csv_path = os.path.join(log_dir, "loss_log.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss_clean", "train_loss_noisy", "val_loss",
                                 "k10_rmse", "k30_rmse", "k60_rmse",
                                 "phase_err_k10", "phase_err_k30", "phase_err_k60"])

    # ---- Training loop -------------------------------------------------------
    print("[FLD] Training started.")
    for epoch in range(config["epochs"]):
        model.train()
        buf_freq.clear(); buf_amp.clear(); buf_off.clear()
        running_loss  = 0.0
        running_recon = 0.0

        for batch_x, batch_y in train_dl:
            batch_x = batch_x.to(device)   # (B, W, 21)
            batch_y = batch_y.to(device)

            win_x = batch_x.unfold(1, history_horizon, 1)   # (B, F+1, 21, H)
            win_y = batch_y.unfold(1, history_horizon, 1)

            batch_input = win_x[:, 0, :, :]                 # (B, 21, H)
            if noise_level > 0.0:
                batch_input = batch_input + torch.randn_like(batch_input) * noise_level

            pred_dynamics, latent, signal, params = model(batch_input, k=forecast_horizon)
            phase, frequency, amplitude, offset = params

            tgt_all    = torch.stack([win_y[:, i, :, :] for i in range(forecast_horizon)])
            recon_loss = compute_reconstruction_loss(
                pred_dynamics, tgt_all,
                gamma=config.get("gamma", 1.0),
            )
            loss       = recon_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss  += loss.item()
            running_recon += recon_loss.item()

            buf_freq.insert(frequency)
            buf_amp.insert(amplitude)
            buf_off.insert(offset)

        train_loss_noisy = running_loss  / len(train_dl)
        train_recon      = running_recon / len(train_dl)

        # ---- Clean train loss (eval mode, no noise, first 8 batches) ---------
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
                    pred_dyn, tgt_b, gamma=config.get("gamma", 1.0)
                ).item()
                n_clean += 1
        train_loss = clean_running / max(n_clean, 1)

        # ---- Validation + diagnostics ----------------------------------------
        model.eval()
        val_running = 0.0
        all_pred, all_tgt = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_dl:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                win_x   = batch_x.unfold(1, history_horizon, 1)
                win_y   = batch_y.unfold(1, history_horizon, 1)
                pred_dyn, _, _, _ = model(win_x[:, 0, :, :], k=forecast_horizon)
                tgt_b   = torch.stack([win_y[:, i, :, :] for i in range(forecast_horizon)])
                val_running += compute_reconstruction_loss(
                    pred_dyn, tgt_b, gamma=config.get("gamma", 1.0)
                ).item()
                if len(all_pred) < 8:
                    all_pred.append(pred_dyn.cpu())
                    all_tgt.append(tgt_b.cpu())
        val_loss = val_running / len(val_dl)

        pred_cat    = torch.cat(all_pred, dim=1)
        tgt_cat     = torch.cat(all_tgt,  dim=1)
        horiz_rmse  = rmse_by_horizon(pred_cat, tgt_cat)
        diag_k      = [9, 29, forecast_horizon - 1]
        diag_vals   = [horiz_rmse[k].item() for k in diag_k]
        phase_errs  = []
        for k_step in diag_k:
            w_t  = tgt_cat[0].to(device)
            w_tk = tgt_cat[min(k_step, forecast_horizon-1)].to(device)
            phase_errs.append(
                phase_propagation_error(model, w_t, w_tk, 1.0/120.0, k_step+1)
            )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        hz_str = "  ".join(f"k{k+1}={v:.4f}" for k, v in zip(diag_k, diag_vals))
        ph_str = "  ".join(f"k{k+1}={v:.4f}r" for k, v in zip(diag_k, phase_errs))
        print(f"[FLD] Epoch [{epoch + 1:>3}/{config['epochs']}]  "
              f"train(clean)={train_loss:.5f}  train(noisy)={train_loss_noisy:.5f}  val={val_loss:.5f}  "
              f"| recon={train_recon:.5f}")
        print(f"       horiz RMSE: {hz_str}   phase err: {ph_str}")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, train_loss, train_loss_noisy, val_loss]
                                   + diag_vals + phase_errs)

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
                    epoch,
                )
                _plot_phase_evolution(
                    model, dataset, device,
                    fig_phase_evo, ax_phase_evo,
                    latent_dim, history_horizon,
                )
                _plot_channel_histograms(
                    fig_hist, ax_hist,
                    buf_freq, buf_amp, buf_off,
                    latent_dim,
                )
                _plot_fourier_channels(
                    model, dataset, device,
                    fig_fourier_ch, ax_fourier_ch,
                    latent_dim, history_horizon,
                )
                _plot_future_predictions(
                    model, dataset, device,
                    history_horizon, forecast_horizon,
                    fig_pred, ax_pred,
                    n_samples=6, sample_rate=120.0, epoch=epoch,
                )
                _plot_long_trajectory(
                    model, dataset, device,
                    history_horizon, fig_long, ax_long,
                    n_blocks=200, forecast_horizon=forecast_horizon,
                    sample_rate=120.0, epoch=epoch,
                )
                _plot_latent_sine_trajectory(
                    model, dataset, device,
                    history_horizon, fig_latent, ax_latent,
                    n_blocks=200, forecast_horizon=forecast_horizon,
                    sample_rate=120.0, epoch=epoch,
                )
                model.train()

        if epoch % config["save_every"] == 0:
            _save(model, optimizer, epoch, config, dataset, log_dir)

    _save(model, optimizer, config["epochs"], config, dataset, log_dir)
    print(f"[FLD] Done. Checkpoints in: {log_dir}")
    plt.show()


# ---------------------------------------------------------------------------
# Annotation helper
# ---------------------------------------------------------------------------

def _note(ax, lines, loc="tl"):
    pass


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_loss_curve(fig, ax, train_losses, val_losses):
    ax.cla()
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train", lw=1.8)
    ax.plot(epochs, val_losses,   label="Val",   lw=1.8, ls="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss  (un-normalised scale, summed over 60 future windows)")
    ax.set_title("FLD AMASS — Training Loss", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.4)
    _note(ax, [
        "WHAT: MSE between predicted and actual joint angles,",
        "      summed across all 60 future forecast windows.",
        "",
        "GOOD: both curves decrease steadily. Val slightly",
        "      above Train is normal (generalisation gap).",
        "",
        "BAD:  Val rising while Train falls  → overfitting.",
        "      Both flat after epoch 2-3     → lr too high or",
        "                                      model too small.",
        "      Very spiky                    → lr too high.",
    ], loc="tr")
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
                latent_dim, history_horizon, forecast_horizon, epoch):

    # ------------------------------------------------------------------
    # fig_dist — Fourier parameter distributions across the epoch
    # ------------------------------------------------------------------
    plotter.plot_distribution(ax_dist[0], buf_freq.get(), title="Frequency  (cycles / window)")
    plotter.plot_distribution(ax_dist[1], buf_amp.get(),  title="Amplitude  (oscillation strength)")
    plotter.plot_distribution(ax_dist[2], buf_off.get(),  title="Offset  (DC bias per channel)")

    _note(ax_dist[0], [
        "WHAT: mean ± std of the dominant frequency",
        "      extracted by FFT for each latent channel,",
        "      collected across every batch this epoch.",
        "",
        "GOOD: bars spread at different heights → each",
        "      channel captured a different oscillation rate.",
        "      Walk cadence at 120 Hz ≈ 1–2 Hz (stride),",
        "      step freq ≈ 2–4 Hz.",
        "",
        "BAD:  all bars equal height → collapsed (model",
        "      treats all channels the same).",
        "      All near-zero → no oscillation learned yet.",
    ])
    _note(ax_dist[1], [
        "WHAT: oscillation strength per latent channel.",
        "",
        "GOOD: most channels non-zero and varied.",
        "      Taller = that channel encodes a stronger",
        "      periodic component of the motion.",
        "",
        "BAD:  many near-zero bars → dead channels that",
        "      contribute nothing to reconstruction.",
        "      All equal → channel collapse.",
    ])
    _note(ax_dist[2], [
        "WHAT: DC offset (mean level) each channel sits at.",
        "      Data is globally normalised so true DC ≈ 0.",
        "",
        "GOOD: bars close to zero.",
        "",
        "BAD:  large offsets → channel is encoding a mean",
        "      shift rather than an oscillation. Can indicate",
        "      poor normalisation or a vanishing channel.",
    ])
    fig_dist.suptitle(
        f"[Epoch {epoch+1}]  Latent Fourier Parameter Distributions  "
        f"(collected across all {latent_dim} channels × all batches)",
        fontsize=10, fontweight="bold",
    )
    _show(fig_dist)

    # ------------------------------------------------------------------
    # fig_recon — end-to-end pipeline visualised on one window
    # ------------------------------------------------------------------
    eval_x, eval_y = dataset[0]
    eval_x = eval_x.unsqueeze(0).to(device)
    eval_y = eval_y.unsqueeze(0).to(device)
    win_x  = eval_x.unfold(1, history_horizon, 1)
    win_y  = eval_y.unfold(1, history_horizon, 1)
    eval_input  = win_x[:, 0, :, :]
    eval_target = win_y[:, 0, :, :]

    pred_dynamics, latent, signal, params = model(eval_input, k=forecast_horizon)

    plotter.plot_curves(ax_recon[0], eval_input[0], -1.0, 1.0, -5.0, 5.0,
                        title=f"[Row 0]  INPUT — 21 normalised joint-angle channels × {history_horizon} frames  "
                              f"(7 joints × 3 Euler axes, ZYX degrees)",
                        show_axes=False)
    _note(ax_recon[0], [
        "Raw normalised lower-body Euler angles fed into the encoder.",
        "21 overlapping lines = spine1, L/R hip, L/R knee, L/R ankle × Z/Y/X.",
        "This is the ground truth the model must learn to reconstruct.",
        "Nothing to diagnose here — it is the reference signal.",
    ])

    plotter.plot_curves(ax_recon[1], latent[0], -1.0, 1.0, -2.0, 2.0,
                        title=f"[Row 1]  LATENT EMBEDDING — encoder output before FFT  "
                              f"({latent_dim} channels × {history_horizon} steps)",
                        show_axes=False)
    _note(ax_recon[1], [
        "Conv1d encoder output — the raw latent time-series before",
        "any Fourier parametrisation. Each of the 8 lines is one",
        "latent channel over the 121-frame window.",
        "",
        "EARLY training: noisy, irregular lines — not yet structured.",
        "GOOD training:  smooth, approximately sinusoidal curves.",
        "                This is what the FFT layer needs to extract",
        "                clean frequency/amplitude/phase parameters.",
        "BAD:            flat lines → encoder collapsed / dead channels.",
    ])

    plotter.plot_circles(ax_recon[2], params[0][0], params[2][0],
                         title=f"[Row 2]  PHASE CIRCLES — current phase & amplitude per latent channel",
                         show_axes=False)
    _note(ax_recon[2], [
        "One circle per latent channel (8 total).",
        "  Radius of circle  = amplitude (oscillation strength).",
        "  Angle of pointer  = current phase (0→1 mapped to 0→2π).",
        "  → think of it as 8 clock hands, each ticking at its own rate.",
        "",
        "GOOD: circles of varied radii with clear pointer angles.",
        "      Diverse angles = channels are at different phases of",
        "      the gait cycle — not all synchronised.",
        "BAD:  all tiny circles → amplitudes collapsed to zero.",
        "      All same radius  → no channel diversity.",
        "      All pointers at same angle → phase collapse.",
    ])

    plotter.plot_curves(ax_recon[3], signal[0], -1.0, 1.0, -2.0, 2.0,
                        title=f"[Row 3]  FOURIER SIGNAL — parametrised sine wave per channel  "
                              f"z = amp·sin(2π(freq·t + phase)) + offset",
                        show_axes=False)
    _note(ax_recon[3], [
        "The sine wave reconstructed from the extracted Fourier parameters.",
        "This is what actually gets passed to the decoder.",
        "  Signal = amplitude × sin(2π × (frequency × t + phase)) + offset",
        "",
        "GOOD: smooth sinusoids matching Row 1 (latent embedding).",
        "      When Rows 1 and 3 look alike, the FFT layer is",
        "      successfully parametrising the encoder output.",
        "BAD:  flat lines   → zero amplitude, dead channel.",
        "      Very different from Row 1 → Fourier fit is poor.",
        "      Mismatched frequency → FFT picked wrong dominant freq.",
    ])

    plotter.plot_curves(ax_recon[4], pred_dynamics[0][0], -1.0, 1.0, -5.0, 5.0,
                        title=f"[Row 4]  RECONSTRUCTION — decoder output for the current window  "
                              f"(should match Row 0)",
                        show_axes=False)
    _note(ax_recon[4], [
        "The decoder's reconstruction of the input window.",
        "Compare visually against Row 0 (input).",
        "",
        "GOOD: same rough shape, amplitude and frequency as Row 0.",
        "      Does not need to be perfect early in training.",
        "BAD:  completely different structure from Row 0.",
        "      All lines flat → decoder collapsed.",
        "      Wrong frequency → model learned wrong oscillation rate.",
    ])

    plotter.plot_curves(
        ax_recon[5],
        torch.vstack((eval_target[0].flatten(0, 1).unsqueeze(0),
                      pred_dynamics[0][0].flatten(0, 1).unsqueeze(0))),
        -1.0, 1.0, -5.0, 5.0,
        title="[Row 5]  TARGET vs RECONSTRUCTION — flattened overlay  "
              "(2 curves: ground truth & model output)",
        show_axes=False,
    )
    _note(ax_recon[5], [
        "Overlay of the flattened ground truth (all 21 channels",
        "concatenated) and the model's reconstruction.",
        "Two curves — if they overlap the model is reconstructing well.",
        "",
        "GOOD: curves tightly overlapping or hard to distinguish.",
        "BAD:  constant offset → systematic bias (check normalisation).",
        "      Out of phase    → model learned wrong phase.",
        "      Different amplitude → amplitude not yet converged.",
        "      Random-looking gap → not enough training yet.",
    ])

    fig_recon.suptitle(
        f"[Epoch {epoch+1}]  End-to-End Pipeline  —  One window from clip 0\n"
        "Read top to bottom: Input → Encoder → FFT params → Fourier signal → Decoder → Reconstruction",
        fontsize=9, fontweight="bold",
    )
    _show(fig_recon)

    # ------------------------------------------------------------------
    # fig_params — per-channel Fourier parameter grid
    # ------------------------------------------------------------------
    phase_all     = params[0][0]
    frequency_all = params[1][0]
    amplitude_all = params[2][0]
    offset_all    = params[3][0]

    for j in range(latent_dim):
        plotter.plot_phase_1d(ax_params[j, 0], phase_all[j].unsqueeze(0),
                              amplitude_all[j].unsqueeze(0),
                              title=("1D Phase  [0,1] weighted by amplitude" if j == 0 else None),
                              show_axes=False)
        plotter.plot_phase_2d(ax_params[j, 1], phase_all[j].unsqueeze(0),
                              amplitude_all[j].unsqueeze(0),
                              title=("2D Phase  sin/cos components" if j == 0 else None),
                              show_axes=False)
        plotter.plot_curves(ax_params[j, 2], frequency_all[j].view(1, 1),
                            -1.0, 1.0, 0.0, 4.0,
                            title=("Frequency  (cycles/window)" if j == 0 else None),
                            show_axes=False)
        plotter.plot_curves(ax_params[j, 3], amplitude_all[j].view(1, 1),
                            -1.0, 1.0, 0.0, 1.0,
                            title=("Amplitude  (oscillation strength)" if j == 0 else None),
                            show_axes=False)
        plotter.plot_curves(ax_params[j, 4], offset_all[j].view(1, 1),
                            -1.0, 1.0, -1.0, 1.0,
                            title=("Offset  (DC bias)" if j == 0 else None),
                            show_axes=False)
        # Row label
        ax_params[j, 0].set_ylabel(f"ch {j}", fontsize=8, rotation=0, labelpad=28, va="center")

    # Annotation only on row 0 columns (column headers are already titles)
    _note(ax_params[0, 0], [
        "Phase scalar on [0,1].",
        "Line brightness weighted by amplitude.",
        "GOOD: diverse values across channels.",
        "BAD: all at same value → phase collapse.",
    ])
    _note(ax_params[0, 2], [
        "How fast this channel oscillates.",
        "Walk stride ≈ 1-2 Hz at 120 Hz.",
        "Step freq   ≈ 2-4 Hz.",
        "GOOD: channels use different frequencies.",
        "BAD: all same → not diverse.",
    ])
    _note(ax_params[0, 3], [
        "Oscillation strength.",
        "GOOD: clearly above zero.",
        "BAD: near zero → dead channel.",
    ])
    _note(ax_params[0, 4], [
        "DC bias (mean level).",
        "GOOD: near zero (data normalised).",
        "BAD: large → encoding mean, not oscillation.",
    ])

    fig_params.suptitle(
        f"[Epoch {epoch+1}]  Per-Channel Fourier Parameters  —  Single window  "
        f"({latent_dim} channels × 5 params)\n"
        "Each row = one latent channel.  "
        "Cols: 1D Phase | 2D Phase | Frequency | Amplitude | Offset",
        fontsize=9, fontweight="bold",
    )
    _show(fig_params)

    # ------------------------------------------------------------------
    # fig_pca — per-type phase manifold scatter
    # ------------------------------------------------------------------
    _plot_manifold_scatter(
        model, dataset, device, history_horizon,
        fig_pca, ax_pca, epoch=epoch,
    )


# ---------------------------------------------------------------------------
# Per-clip phase manifold scatter
# ---------------------------------------------------------------------------

_N_MANIFOLD_CLIPS = 7   # number of clips to show side by side


def _plot_manifold_scatter(model, dataset, device, history_horizon,
                            fig, axes, epoch=None, n_windows=500):
    """
    One scatter subplot per individual clip, evenly sampled across the
    full dataset.  Each subplot shows 500 consecutive sliding windows from
    one distinct recording, so there is zero data bleed between subplots.

    PCA is fitted on ALL clips combined so every subplot shares the same
    2-D coordinate system — shapes and positions are directly comparable.
    """
    clips      = dataset.norm_clips    # list of (T_i, 21) tensors
    labels     = dataset.clip_labels   # list of (subject, clip_name)
    n_clips    = len(clips)
    H          = history_horizon

    # Pick clips evenly spaced through the full list
    indices = np.linspace(0, n_clips - 1, _N_MANIFOLD_CLIPS, dtype=int)
    cmap    = plt.cm.tab10

    # Encode each selected clip into manifold vectors
    clip_manifolds = []
    clip_titles    = []

    model.eval()
    with torch.no_grad():
        for ci in indices:
            frames = clips[ci]                             # (T, 21)
            subj, cname = labels[ci]
            n = min(n_windows, frames.shape[0] - H)
            if n <= 0:
                clip_manifolds.append(np.zeros((1, model.latent_channel * 2)))
                clip_titles.append(f"{subj}\n{cname[:12]}")
                continue
            seqs = torch.stack(
                [frames[i : i + H].T for i in range(n)]
            ).to(device)                                   # (n, 21, H)
            _, _, _, params = model(seqs)
            phase = params[0].cpu()                        # (n, latent_dim)
            amp   = params[2].cpu()
            vecs  = torch.hstack((
                amp * torch.sin(2.0 * torch.pi * phase),
                amp * torch.cos(2.0 * torch.pi * phase),
            )).numpy()                                     # (n, latent_dim*2)
            clip_manifolds.append(vecs)
            clip_titles.append(f"{subj} / {cname.replace('_poses.npz','')}")

    # Fit ONE PCA on all clips combined → shared coordinate system
    all_vecs = np.vstack(clip_manifolds)
    pca = _PCA(n_components=2)
    pca.fit(all_vecs)

    title_prefix = f"[Epoch {epoch+1}]  " if epoch is not None else ""

    for i, (ax, vecs, title) in enumerate(zip(axes, clip_manifolds, clip_titles)):
        ax.cla()
        proj = pca.transform(vecs)
        ax.scatter(proj[:, 0], proj[:, 1],
                   s=6, alpha=0.45, color=cmap(i % 10), edgecolors="none")
        ax.set_title(title, fontsize=7, fontweight="bold")
        ax.set_xlabel("PC1", fontsize=7)
        ax.set_ylabel("PC2", fontsize=7)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=6)

    fig.suptitle(
        f"{title_prefix}Phase Manifold — {_N_MANIFOLD_CLIPS} individual clips  |  "
        f"Each subplot = one distinct recording ({n_windows} consecutive windows)  |  "
        "Shared PCA space — shapes and positions are directly comparable",
        fontsize=9, fontweight="bold",
    )
    _show(fig)


# ---------------------------------------------------------------------------
# Per-channel diagnostic plots
# ---------------------------------------------------------------------------

def _plot_phase_evolution(model, dataset, device,
                          fig, axes, latent_dim, history_horizon):
    """
    FLD-specific dynamics test.
    Encode ONE anchor window at t=0 → propagate phase with internal dynamics
    (no re-encode) → compare against re-encoded actual future phases.
    """
    dt      = 1.0 / 120.0
    n_steps = min(200, dataset.data_in.shape[0] - history_horizon)
    clip    = dataset.data_in[:history_horizon + n_steps].to(device)

    anchor = clip[:history_horizon].T.unsqueeze(0)
    _, _, _, params_0 = model(anchor)
    phase_0 = params_0[0][0].cpu()
    freq_0  = params_0[1][0].cpu()

    ks         = torch.arange(n_steps, dtype=torch.float32)
    phase_pred = phase_0.unsqueeze(0) + freq_0.unsqueeze(0) * dt * ks.unsqueeze(1)

    future_seqs  = torch.stack([clip[k:k + history_horizon].T for k in range(n_steps)])
    _, _, _, params_future = model(future_seqs)
    phase_actual = params_future[0].cpu().numpy()

    time = ks.numpy() * dt
    cmap = plt.cm.tab10

    for j, ax in enumerate(axes):
        ax.cla()
        ax.plot(time, phase_pred[:, j].numpy(), color=cmap(j), lw=1.8,
                label="FLD predicted  (phase_0 + freq·dt·k,  no re-encode)")
        ax.plot(time, phase_actual[:, j],       color=cmap(j), lw=1.1,
                ls="--", alpha=0.75, label="Re-encoded actual")
        ax.set_ylabel(f"ch {j}", fontsize=8, rotation=0, labelpad=28, va="center")
        ax.set_title(f"Channel {j}   anchor freq = {freq_0[j]:.3f} Hz", fontsize=8)
        ax.legend(fontsize=6.5, loc="upper left")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time (s)")

    _note(axes[0], [
        "WHAT THIS TESTS (unique to FLD — impossible with a plain PAE):",
        "  Encodes ONE window at t=0 → gets phase_0 and frequency.",
        "  Predicts all future phases via: phase_pred[k] = phase_0 + freq·(1/120)·k",
        "  'Re-encoded actual' = independently encoding the real window at each t=k.",
        "  The two curves overlapping means FLD's internal dynamics model is correct.",
        "",
        "EARLY training:  both lines erratic, no agreement → not converged yet.",
        "GOOD (mid):      lines loosely track for the first 20-40 steps (0.15-0.3 s).",
        "GOOD (converged):solid and dashed closely track for 60-120+ steps (0.5-1 s).",
        "                 → FLD has learned that frequency predicts future phase.",
        "BAD:             solid line is a straight ramp but dashed is flat/noisy",
        "                 → encoder assigns inconsistent phases between windows.",
        "DEAD channel:    both lines flat near zero → amplitude≈0, channel unused.",
    ], loc="tr")

    fig.suptitle(
        "FLD Phase Dynamics Validation  —  Predicted (solid) vs Re-encoded Actual (dashed)\n"
        "Overlap = model learned a correct periodic dynamics model  |  Divergence = linear phase propagation limit",
        fontsize=9, fontweight="bold",
    )
    _show(fig)


def _plot_channel_histograms(fig, axes, buf_freq, buf_amp, buf_off, latent_dim):
    """
    Histograms of frequency, amplitude, and offset per latent channel,
    collected across every batch in the epoch.
    """
    freq_data = buf_freq.get().numpy()
    amp_data  = buf_amp.get().numpy()
    off_data  = buf_off.get().numpy()

    cmap     = plt.cm.tab10
    col_info = [
        ("Frequency  (cycles/window)", freq_data,
         "Tight narrow peak at a specific value = channel has stable identity.\n"
         "Walk stride at 120 Hz ≈ 1-2 Hz.  Step freq ≈ 2-4 Hz.\n"
         "Wide/flat = channel hasn't settled on a frequency.\n"
         "Bimodal = channel used differently for walk vs run."),
        ("Amplitude  (oscillation strength)", amp_data,
         "Peak away from zero = channel is actively encoding oscillation.\n"
         "Peak near zero = dead channel — contributing nothing.\n"
         "As training improves, peaks should sharpen and move right."),
        ("Offset  (DC bias per channel)", off_data,
         "Should be narrow and centered near zero (data is normalised).\n"
         "Wide spread = channel encoding mean-level shifts, not oscillations.\n"
         "Large positive/negative mean = possible normalisation issue."),
    ]

    for j in range(latent_dim):
        for col, (title, data, interp) in enumerate(col_info):
            ax = axes[j, col]
            ax.cla()
            ax.hist(data[:, j], bins=40, color=cmap(j), alpha=0.75, edgecolor="none")
            mean_v = data[:, j].mean()
            std_v  = data[:, j].std()
            ax.axvline(mean_v, color="k", lw=1.2, ls="--",
                       label=f"μ={mean_v:.3f}  σ={std_v:.3f}")
            ax.legend(fontsize=6.5, loc="upper right")
            ax.set_ylabel(f"ch {j}", fontsize=7, rotation=0, labelpad=28, va="center")
            ax.grid(alpha=0.25)
            if j == 0:
                ax.set_title(title, fontsize=9, fontweight="bold")
                _note(ax, interp.split("\n"), loc="tl")

    fig.suptitle(
        "Per-Channel Parameter Histograms  —  Collected across ALL batches this epoch\n"
        "Each row = one latent channel.  Dashed line = mean.  "
        "Tight unimodal peaks = stable, well-identified channels.",
        fontsize=9, fontweight="bold",
    )
    _show(fig)


def _plot_fourier_channels(model, dataset, device,
                            fig, axes, latent_dim, history_horizon):
    """
    For a single anchor window, show each channel's parametrised Fourier
    signal individually — the exact sine wave the decoder receives.
    """
    clip = dataset.data_in[:history_horizon].to(device)
    x    = clip.T.unsqueeze(0)

    _, _, signal, params = model(x)
    sig   = signal[0].cpu().numpy()
    phase = params[0][0].cpu().numpy()
    freq  = params[1][0].cpu().numpy()
    amp   = params[2][0].cpu().numpy()
    off   = params[3][0].cpu().numpy()

    time = np.arange(history_horizon) / 120.0
    cmap = plt.cm.tab10

    for j, ax in enumerate(axes):
        ax.cla()
        ax.plot(time, sig[j], color=cmap(j), lw=1.5)
        ax.axhline(off[j], color="#888888", lw=0.8, ls="--",
                   label=f"offset = {off[j]:.4f}")
        ax.fill_between(time,
                        np.full_like(time, off[j] - abs(amp[j])),
                        np.full_like(time, off[j] + abs(amp[j])),
                        alpha=0.08, color=cmap(j), label=f"±amp = ±{abs(amp[j]):.4f}")
        ax.set_ylabel(f"ch {j}", fontsize=8, rotation=0, labelpad=28, va="center")
        ax.set_title(
            f"Ch {j}   phase={phase[j]:.3f}   freq={freq[j]:.3f} Hz   "
            f"amp={amp[j]:.4f}   offset={off[j]:.4f}",
            fontsize=8,
        )
        ax.legend(fontsize=6.5, loc="upper right")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time (s)")

    _note(axes[0], [
        "WHAT: the exact sine wave the decoder receives for each latent channel.",
        "  Signal = amplitude × sin(2π × (freq × t + phase)) + offset",
        "  Dashed line = offset level (DC bias).  Shaded band = ±amplitude.",
        "",
        "GOOD (converged):  clear smooth sinusoid with visible oscillation.",
        "                   Shaded band away from zero = active channel.",
        "                   Different frequencies across channels.",
        "",
        "BAD — flat line:   amplitude ≈ 0 → dead channel, not contributing.",
        "                   All flat lines = model fully collapsed.",
        "BAD — all same:    every channel at same freq → no diversity,",
        "                   model is using redundant latent dimensions.",
        "EARLY training:    low amplitudes and noisy signal is normal —",
        "                   channels haven't specialised yet.",
    ], loc="tr")

    fig.suptitle(
        "Per-Channel Fourier Signals  —  What the decoder actually receives\n"
        "Signal = amp · sin(2π · (freq · t + phase)) + offset  |  "
        "Dashed = offset  |  Shaded band = ±amplitude",
        fontsize=9, fontweight="bold",
    )
    _show(fig)


# ---------------------------------------------------------------------------
# Future prediction vs ground truth
# ---------------------------------------------------------------------------

def _plot_future_predictions(model, dataset, device,
                              history_horizon, forecast_horizon,
                              fig, axes, n_samples=6,
                              sample_rate=120.0, epoch=None):
    """
    Pick n_samples windows evenly spaced across dataset.data_in,
    encode each, predict forecast_horizon steps ahead, and overlay
    predicted future vs ground truth.

    Each subplot = one sample window.
    Gray region  = context given to the model.
    Solid lines  = ground truth future.
    Dashed lines = model predicted future.
    All features overlaid with transparency for a gestalt view.
    """
    data = dataset.data_in        # (N_frames, D) normalised
    N, D = data.shape
    K    = forecast_horizon
    H    = history_horizon

    max_start = N - H - K - 1
    if max_start <= 0:
        return

    starts = np.linspace(0, max_start, n_samples, dtype=int)
    t_ctx  = np.arange(H) / sample_rate
    t_fut  = np.arange(H, H + K) / sample_rate
    cmap   = plt.cm.tab20

    for ax, start in zip(axes, starts):
        ax.cla()

        ctx    = data[start : start + H]               # (H, D)
        gt_fut = data[start + H : start + H + K]       # (K, D)

        inp = ctx.T.unsqueeze(0).to(device)            # (1, D, H)
        with torch.no_grad():
            pred_dyn, _, _, _ = model(inp, k=K)

        # first frame of each predicted window = predicted frame at that step
        pred_fut = torch.stack([
            pred_dyn[k][0, :, 0] for k in range(K)
        ]).cpu().numpy()                               # (K, D)

        ctx_np    = ctx.numpy()
        gt_fut_np = gt_fut.numpy()

        ax.axvspan(0, H / sample_rate, alpha=0.07, color="grey")
        ax.axvline(H / sample_rate, color="grey", lw=0.8, ls="--")

        for d in range(D):
            c = cmap(d % 20)
            ax.plot(t_ctx, ctx_np[:, d],    color=c, lw=0.7, alpha=0.25)
            ax.plot(t_fut, gt_fut_np[:, d], color=c, lw=1.0, alpha=0.50)
            ax.plot(t_fut, pred_fut[:, d],  color=c, lw=1.2, alpha=0.85, ls="--")

        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=6)
        ax.set_ylabel(f"t={start/sample_rate:.1f}s", fontsize=7,
                      rotation=0, labelpad=38, va="center")

    axes[-1].set_xlabel("Time (s)")

    epoch_str = f"[Epoch {epoch+1}]  " if epoch is not None else ""
    fig.suptitle(
        f"{epoch_str}Predicted Future vs Ground Truth  —  {n_samples} windows\n"
        f"Gray = context ({H} frames, {H/sample_rate:.2f} s)  |  "
        f"Solid = GT  |  Dashed = predicted  |  "
        f"{D} features overlaid (normalised)  |  Forecast = {K/sample_rate:.2f} s",
        fontsize=9, fontweight="bold",
    )
    _show(fig)


# ---------------------------------------------------------------------------
# Long trajectory — GT vs prediction stitched over many blocks
# ---------------------------------------------------------------------------

def _plot_long_trajectory(model, dataset, device,
                           history_horizon, fig, axes,
                           n_blocks=200, forecast_horizon=60,
                           sample_rate=120.0, epoch=None,
                           batch_size=64):
    """
    Step through data in forecast_horizon-sized blocks, predict all K frames
    for each block, stitch into a continuous trajectory.  One row per feature.
    Blue = GT.  Red dashed = prediction.  Vertical lines = block boundaries.
    """
    data    = dataset.data_in
    std_np  = dataset.std_tensor.numpy()
    mean_np = dataset.mean_tensor.numpy()
    N, D    = data.shape
    H, K    = history_horizon, forecast_horizon

    n_blocks = min(n_blocks, (N - H) // K)

    pred_traj, gt_traj = [], []

    model.eval()
    with torch.no_grad():
        for b_start in range(0, n_blocks, batch_size):
            b_end      = min(b_start + batch_size, n_blocks)
            ctx_starts = [i * K for i in range(b_start, b_end)]
            wins = torch.stack(
                [data[s : s + H].T for s in ctx_starts]
            ).to(device)
            pred_dyn, _, _, _ = model(wins, k=K)
            pred_block = torch.stack(
                [pred_dyn[k][:, :, 0] for k in range(K)], dim=1
            ).cpu().numpy()                          # (B, K, D)
            for bi, s in enumerate(ctx_starts):
                pred_traj.append(pred_block[bi])     # (K, D)
                gt_traj.append(data[s : s + K].numpy())

    pred_arr  = np.concatenate(pred_traj, axis=0)
    gt_arr    = np.concatenate(gt_traj,   axis=0)
    pred_orig = pred_arr * std_np + mean_np
    gt_orig   = gt_arr   * std_np + mean_np

    n_frames = pred_arr.shape[0]
    t        = np.arange(n_frames) / sample_rate
    # Build feature labels: one name per Euler axis per joint
    euler_axes = ["Z", "Y", "X"]
    feat_labels = [f"{jnt}.{ax}"
                   for jnt in LOWER_BODY_JOINT_NAMES
                   for ax in euler_axes]

    ax_list = np.atleast_1d(axes)
    for fi, ax in enumerate(ax_list):
        ax.cla()
        ax.plot(t, gt_orig[:, fi],   color="steelblue", lw=0.5, alpha=0.85, label="GT")
        ax.plot(t, pred_orig[:, fi], color="tomato",    lw=0.5, alpha=0.85, ls="--", label="Pred")
        for blk in range(1, n_blocks):
            ax.axvline(blk * K / sample_rate, color="grey", lw=0.4, ls=":", alpha=0.35)
        label = feat_labels[fi] if fi < len(feat_labels) else f"feat {fi}"
        ax.set_ylabel(f"{label}\n(°)", fontsize=7, rotation=0, labelpad=85, va="center")
        ax.grid(alpha=0.15)
        ax.tick_params(labelsize=6)
        if fi == 0:
            ax.legend(fontsize=7, loc="upper right", ncol=2)

    ax_list[-1].set_xlabel("Time (s)")
    epoch_str = f"[Epoch {epoch+1}]  " if epoch is not None else ""
    fig.suptitle(
        f"{epoch_str}Long Trajectory — GT (blue) vs Prediction (red dashed)\n"
        f"{n_blocks} blocks × {K} frames = {n_frames:,} frames ({n_frames/sample_rate:.1f}s)  |  "
        f"each block = fresh {K/sample_rate*1000:.0f}ms prediction  |  "
        f"vertical lines = block boundaries  |  degrees",
        fontsize=9, fontweight="bold",
    )
    _show(fig)


# ---------------------------------------------------------------------------
# Latent sine curves over the long trial
# ---------------------------------------------------------------------------

def _plot_latent_sine_trajectory(model, dataset, device,
                                  history_horizon, fig, axes,
                                  n_blocks=200, forecast_horizon=60,
                                  sample_rate=120.0, epoch=None,
                                  batch_size=64):
    """
    For each block, encode the context and compute
    A·sin(2π·(φ + f·dt·k)) + b for k=0..K-1 per latent channel.
    Stitch into a continuous trajectory.  One row per channel.
    """
    data = dataset.data_in
    N, D = data.shape
    H, K = history_horizon, forecast_horizon
    dt   = 1.0 / sample_rate

    n_blocks   = min(n_blocks, (N - H) // K)
    latent_dim = model.latent_channel

    sine_blocks = []

    model.eval()
    with torch.no_grad():
        for b_start in range(0, n_blocks, batch_size):
            b_end      = min(b_start + batch_size, n_blocks)
            ctx_starts = [i * K for i in range(b_start, b_end)]
            wins = torch.stack(
                [data[s : s + H].T for s in ctx_starts]
            ).to(device)
            _, _, _, params = model(wins, k=1)
            phase  = params[0].cpu()
            freq   = params[1].cpu()
            amp    = params[2].cpu()
            offset = params[3].cpu()

            k_steps = torch.arange(K, dtype=torch.float32)
            for bi in range(b_end - b_start):
                phase_k = phase[bi].unsqueeze(0) + freq[bi].unsqueeze(0) * dt * k_steps.unsqueeze(1)
                sine_k  = amp[bi] * torch.sin(2 * torch.pi * phase_k) + offset[bi]
                sine_blocks.append(sine_k.numpy())   # (K, latent_dim)

    sine_arr = np.concatenate(sine_blocks, axis=0)   # (n_blocks*K, latent_dim)
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
# Save helper
# ---------------------------------------------------------------------------

def _save(model, optimizer, epoch, config, dataset, log_dir):
    path = os.path.join(log_dir, f"fld_amass_epoch_{epoch}.pt")
    torch.save(
        {
            "fld_state_dict":       model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch":                epoch,
            "config":               config,
            "in_mean":              dataset.mean_tensor,
            "in_std":               dataset.std_tensor,
            "out_mean":             dataset.out_mean_tensor,
            "out_std":              dataset.out_std_tensor,
        },
        path,
    )
    print(f"[FLD] Checkpoint → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    config = {
        # FLD architecture
        "history_horizon":  121,        # nearest odd to 120 (FLD Conv1d requires odd)
        "forecast_horizon":  60,        # predict 0.5 s ahead at 120 Hz
        "latent_dim":         4,        # Fourier latent channels
        "encoder_shape":    [32, 32],
        "decoder_shape":    [32, 32],

        # Training
        "epochs":            30,
        "batch_size":       128,
        "num_workers":        4,
        "lr":               1e-4,
        "weight_decay":     5e-4,
        "noise_level":      0.02,       # light input noise for regularisation

        # Logging
        "save_every":         5,        # checkpoint every N epochs
    }

    train(config)


if __name__ == "__main__":
    main()
