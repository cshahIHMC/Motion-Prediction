"""
inspect_amass_fld.py

Load a saved FLD-AMASS checkpoint and reproduce every plot that
train_amass_fld.py generates at the end of training.

Usage:
    # Auto-picks the latest checkpoint across all run directories
    python3 inspect_amass_fld.py

    # Explicit checkpoint
    python3 inspect_amass_fld.py runs/FLD_AMASS_20260611_172517/fld_amass_epoch_40.pt
"""

import os
import sys
import csv
import glob
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from train_amass_fld import (
    AMASS_ROOT,
    MOTION_TYPES,
    AMASSFLDDataset,
    DistributionBuffer,
    _N_MANIFOLD_CLIPS,
    _plot_loss_curve,
    _plot_epoch,
    _plot_phase_evolution,
    _plot_channel_histograms,
    _plot_fourier_channels,
    _plot_future_predictions,
    _plot_long_trajectory,
    _plot_latent_sine_trajectory,
    _plot_manifold_scatter,

    _show,
)
from DataLoader.amass_loader import _CLIP_CATALOG, load_clean_clips
from sklearn.decomposition import PCA as _PCA
from Models.FLD import FLD
from Library.fld_plotter import Plotter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latest_checkpoint():
    """Return the most recently modified .pt file across all run directories."""
    pts = glob.glob(os.path.join("runs", "FLD_AMASS_*", "*.pt"))
    if not pts:
        raise FileNotFoundError("No FLD_AMASS checkpoints found under runs/")
    return max(pts, key=os.path.getmtime)


def load_checkpoint(path, device):
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    cfg   = ckpt["config"]
    model = FLD(
        observation_dim=21,
        output_dim=21,
        history_horizon=cfg["history_horizon"],
        latent_channel=cfg["latent_dim"],
        device=device,
        dt=1.0 / 120.0,
        encoder_shape=cfg["encoder_shape"],
        decoder_shape=cfg["decoder_shape"],
    )
    model.load_state_dict(ckpt["fld_state_dict"])
    model.to(device).eval()
    print(f"[Inspect] Checkpoint : {path}")
    print(f"[Inspect] Epoch      : {ckpt['epoch']}  |  "
          f"latent_dim={cfg['latent_dim']}  "
          f"H={cfg['history_horizon']}  K={cfg['forecast_horizon']}")
    return model, cfg, ckpt


def _read_loss_csv(run_dir):
    """Read loss_log.csv from the run directory — returns (train_losses, val_losses)."""
    csv_path = os.path.join(run_dir, "loss_log.csv")
    if not os.path.exists(csv_path):
        return [], []
    train_losses, val_losses = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # support both old (train_loss) and new (train_loss_clean) headers
            tkey = "train_loss_clean" if "train_loss_clean" in row else "train_loss"
            train_losses.append(float(row[tkey]))
            val_losses.append(float(row["val_loss"]))
    return train_losses, val_losses


def _collect_buffers(model, dataset, history_horizon, device, n_batches=200):
    """Run n_batches through the model in eval mode to populate distribution buffers."""
    dl  = DataLoader(dataset, batch_size=128, shuffle=True,
                     num_workers=4, pin_memory=True, drop_last=True)
    buf_freq = DistributionBuffer(model.latent_channel)
    buf_amp  = DistributionBuffer(model.latent_channel)
    buf_off  = DistributionBuffer(model.latent_channel)
    model.eval()
    with torch.no_grad():
        for i, (bx, _) in enumerate(dl):
            if i >= n_batches:
                break
            win = bx.to(device).unfold(1, history_horizon, 1)
            _, _, _, params = model(win[:, 0, :, :], k=1)
            buf_freq.insert(params[1])
            buf_amp.insert(params[2])
            buf_off.insert(params[3])
    print(f"[Inspect] Buffers collected over {min(n_batches, len(dl))} batches")
    return buf_freq, buf_amp, buf_off


# ---------------------------------------------------------------------------
# Subject 55 full manifold — all 28 clips (walk + run + jog mixed)
# ---------------------------------------------------------------------------

def plot_subject55_manifold(model, norm_mean, norm_std, history_horizon,
                             device, batch_size=512):
    """
    Encode every window from all 28 subject-55 clips into the phase manifold,
    then show a single PCA scatter coloured by clip so you can see whether
    the different recordings cluster separately within the mixed-activity space.

    Each clip gets its own colour.  Points from the same clip are plotted
    together so you can trace how the latent space evolves within one recording.
    """
    from matplotlib.cm import get_cmap

    mean_np = norm_mean.numpy()
    std_np  = norm_std.numpy()
    H       = history_horizon
    clips = load_clean_clips(
        AMASS_ROOT, "walk_run_jog_s55",
        lower_body_only=True, euler_sequence="ZYX",
        n_windows=0, history_horizon=H,
    )

    if not clips:
        print("[Inspect] No subject-55 clips found.")
        return

    # Use the longest clip
    clip = max(clips, key=lambda c: c["poses"].shape[0])
    frames_norm = torch.from_numpy(
        (clip["poses"].numpy() - mean_np) / std_np
    ).float()
    T = frames_norm.shape[0]
    n = max(0, T - H)

    print(f"[Inspect] Encoding {clip['clip_name']}  ({T:,} frames, {n:,} windows) …")
    vecs_list = []
    model.eval()
    with torch.no_grad():
        for b_start in range(0, n, batch_size):
            b_end = min(b_start + batch_size, n)
            seqs  = torch.stack(
                [frames_norm[i : i + H].T for i in range(b_start, b_end)]
            ).to(device)
            _, _, _, params = model(seqs)
            phase = params[0].cpu()
            amp   = params[2].cpu()
            vecs_list.append(torch.hstack((
                amp * torch.sin(2.0 * torch.pi * phase),
                amp * torch.cos(2.0 * torch.pi * phase),
            )).numpy())

    vecs = np.concatenate(vecs_list, axis=0)
    proj = _PCA(n_components=2).fit_transform(vecs)

    # Colour points by time so you can see trajectory through manifold
    t = np.arange(len(proj))

    fig, ax = plt.subplots(figsize=(10, 9))
    sc = ax.scatter(proj[:, 0], proj[:, 1],
                    c=t, cmap="plasma", s=4, alpha=0.5, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Window index (time →)")
    ax.set_xlabel("PC1", fontsize=10)
    ax.set_ylabel("PC2", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    fig.suptitle(
        f"Subject 55 — Phase Manifold  |  {clip['clip_name'].replace('_poses.npz','')}  |  "
        f"{T:,} frames  {T/120:.1f}s  |  {n:,} windows\n"
        "walk + run + jog mixed  |  colour = time  |  ring = healthy periodic latent space",
        fontsize=9, fontweight="bold",
    )
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce all train_amass_fld plots from a saved checkpoint."
    )
    parser.add_argument(
        "checkpoint", nargs="?", default=None,
        help="Path to .pt checkpoint. Defaults to the most recently saved checkpoint.",
    )
    args = parser.parse_args()

    ckpt_path = args.checkpoint or _latest_checkpoint()
    run_dir   = os.path.dirname(ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inspect] Device: {device}")

    # ---- Load model -----------------------------------------------------------
    model, cfg, ckpt = load_checkpoint(ckpt_path, device)
    H          = cfg["history_horizon"]
    K          = cfg["forecast_horizon"]
    latent_dim = cfg["latent_dim"]
    epoch      = ckpt["epoch"] - 1   # 0-indexed for plot titles

    # ---- Build dataset --------------------------------------------------------
    print("[Inspect] Building dataset …")
    dataset = AMASSFLDDataset(
        root_path=AMASS_ROOT,
        history_horizon=H,
        forecast_horizon=K,
        motion_types=MOTION_TYPES,
        euler_sequence="ZYX",
    )

    # ---- Collect distribution buffers ----------------------------------------
    print("[Inspect] Collecting parameter distributions …")
    buf_freq, buf_amp, buf_off = _collect_buffers(model, dataset, H, device)

    # ---- Create all figures (same layout as train_amass_fld.py) --------------
    plotter         = Plotter()
    _n_feat         = dataset.obs_dim

    fig_dist,       ax_dist       = plt.subplots(1, 3, figsize=(15, 4))
    fig_recon,      ax_recon      = plt.subplots(6, 1, figsize=(10, 18))
    fig_params,     ax_params     = plt.subplots(latent_dim, 5, figsize=(20, latent_dim * 3))
    fig_pca,        ax_pca        = plt.subplots(1, _N_MANIFOLD_CLIPS, figsize=(5 * _N_MANIFOLD_CLIPS, 5))
    fig_loss,       ax_loss       = plt.subplots(figsize=(8, 4))
    fig_phase_evo,  ax_phase_evo  = plt.subplots(latent_dim, 1, figsize=(14, latent_dim * 2), sharex=True)
    fig_hist,       ax_hist       = plt.subplots(latent_dim, 3, figsize=(15, latent_dim * 2))
    fig_fourier_ch, ax_fourier_ch = plt.subplots(latent_dim, 1, figsize=(14, latent_dim * 2), sharex=True)
    fig_pred,       ax_pred       = plt.subplots(6, 1, figsize=(14, 3.5 * 6), sharex=False)
    fig_long,       ax_long       = plt.subplots(_n_feat, 1, figsize=(30, 2.0 * _n_feat), sharex=True)
    fig_latent,     ax_latent     = plt.subplots(latent_dim, 1, figsize=(30, 3.5 * latent_dim), sharex=True)

    # ---- Loss curve -----------------------------------------------------------
    print("[Inspect] Plotting loss curve …")
    train_losses, val_losses = _read_loss_csv(run_dir)
    if train_losses:
        _plot_loss_curve(fig_loss, ax_loss, train_losses, val_losses)
    else:
        print("[Inspect] loss_log.csv not found — skipping loss curve")

    # ---- End-to-end pipeline + per-channel params + manifold scatter ---------
    print("[Inspect] Plotting pipeline / params / manifold …")
    with torch.no_grad():
        _plot_epoch(
            plotter, model, dataset, device,
            fig_dist, ax_dist,
            fig_recon, ax_recon,
            fig_params, ax_params,
            fig_pca, ax_pca,
            buf_freq, buf_amp, buf_off,
            latent_dim, H, K,
            epoch,
        )

    # ---- Clean manifold scatter -----------------------------------------------
    print("[Inspect] Plotting clean manifold scatter (exclusively-labeled clips) …")
    with torch.no_grad():
    # ---- Phase evolution ------------------------------------------------------
    print("[Inspect] Plotting phase evolution …")
    with torch.no_grad():
        _plot_phase_evolution(
            model, dataset, device,
            fig_phase_evo, ax_phase_evo,
            latent_dim, H,
        )

    # ---- Channel histograms ---------------------------------------------------
    print("[Inspect] Plotting channel histograms …")
    _plot_channel_histograms(
        fig_hist, ax_hist,
        buf_freq, buf_amp, buf_off,
        latent_dim,
    )

    # ---- Per-channel Fourier signals ------------------------------------------
    print("[Inspect] Plotting Fourier channels …")
    with torch.no_grad():
        _plot_fourier_channels(
            model, dataset, device,
            fig_fourier_ch, ax_fourier_ch,
            latent_dim, H,
        )

    # ---- Future predictions (6 sample windows) --------------------------------
    print("[Inspect] Plotting future predictions …")
    with torch.no_grad():
        _plot_future_predictions(
            model, dataset, device,
            H, K, fig_pred, ax_pred,
            n_samples=6, sample_rate=120.0, epoch=epoch,
        )

    # ---- Long trajectory ------------------------------------------------------
    print("[Inspect] Plotting long trajectory …")
    with torch.no_grad():
        _plot_long_trajectory(
            model, dataset, device,
            H, fig_long, ax_long,
            n_blocks=200, forecast_horizon=K,
            sample_rate=120.0, epoch=epoch,
        )

    # ---- Latent sine trajectory -----------------------------------------------
    print("[Inspect] Plotting latent sine trajectory …")
    with torch.no_grad():
        _plot_latent_sine_trajectory(
            model, dataset, device,
            H, fig_latent, ax_latent,
            n_blocks=200, forecast_horizon=K,
            sample_rate=120.0, epoch=epoch,
        )

    # ---- Subject 55 full manifold (walk + run + jog mixed, all 28 clips) ------
    print("[Inspect] Plotting subject-55 full manifold (135,776 windows) …")
    plot_subject55_manifold(
        model, dataset.mean_tensor, dataset.std_tensor,
        H, device,
    )

    print("[Inspect] All plots ready.")
    plt.show()


if __name__ == "__main__":
    main()
