"""
plot_tcnn_predictions.py

Evaluation/viewer script for a trained TCNN checkpoint (train_tcnn.py /
Models/TCNN.py's TCNModel_Forecast): loads the latest (or a pinned) run,
reads ONE trial directly from Data/<SUBJECT>/<RATE>/trial_<TRIAL_NUMBER>_
all_data.csv (the whole contiguous recording -- works for any subject,
S01-S05 included, not just whichever subject the checkpoint trained/
validated on), and tiles the model's predictions over ground truth across
the trial: a fresh history_horizon-frame context is fed to the model every
`--interval` frames (default = the checkpoint's own forecast_horizon, i.e.
non-overlapping tiling -- pass a smaller interval to sample more densely,
which makes consecutive blocks' predictions overlap in real time), and each
block's forecast_horizon-frame prediction is drawn at its true time
position, layered on the one continuous ground-truth curve.

Does NOT retrain -- read-only. Two figures (one per modality group: joint
angle in degrees, joint moment in N.m/kg -- TCNN has no velocity output,
unlike FLD), each with one subplot per joint, GT (blue) vs prediction
(red dashed). Also prints per-group RMSE (physical units) over the plotted
span. Figures are shown interactively AND saved to
runs/TCNNEval_<timestamp>/.

Reuses, not reimplements:
  - _find_checkpoint / _load_state_dict_flexible (plot_error_across_time.py)
    -- same newest-run-first / highest-epoch-checkpoint lookup every other
    plotting script in this repo uses.
  - INPUT_COLUMNS / OUTPUT_COLUMNS / MOMENT_COLUMNS / MODALITY_GROUPS /
    _forecast (train_tcnn.py) -- TCNN's own 48-input/20-output column
    convention and single-shot forecast call, not redeclared here.

Run:
    python3 plot_tcnn_predictions.py --subject S05 --trial 4
    # every 10 frames instead of one fresh block per forecast_horizon (denser,
    # overlapping predictions):
    python3 plot_tcnn_predictions.py --subject S05 --trial 4 --interval 10
    # only the first 60s, moments only, a pinned run instead of the latest:
    python3 plot_tcnn_predictions.py --subject S02 --trial 3 --seconds 60 \\
        --feature-group moment --run TCNN_20260924_173737
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from plot_error_across_time import _find_checkpoint, _load_state_dict_flexible   # reused, not reimplemented
from Models.TCNN import TCNModel_Forecast
from train_tcnn import (   # reused, not reimplemented -- TCNN's own column/group/forecast convention
    INPUT_COLUMNS, OUTPUT_COLUMNS, MOMENT_COLUMNS, MODALITY_GROUPS, _forecast,
)

RUN_PREFIX = "TCNN"             # matches train_tcnn.py's log_dir tag prefix
CHECKPOINT_STEM = "tcnn_model"  # matches train_tcnn.py's _save() filename stem

DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")
RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")


# ---------------------------------------------------------------------------
# Config -- CLI-overridable defaults (see parse_args()).
# ---------------------------------------------------------------------------

SUBJECT = "S02"       # <-- trial_id prefix; matches Data/<SUBJECT>/<RATE>/...
TRIAL_NUMBER = "2"    # <-- matches Data/<SUBJECT>/<RATE>/trial_<TRIAL_NUMBER>_all_data.csv
RATE = "150"          # matches Data/<SUBJECT>/<RATE>/... (no filename suffix at this rate)
SAMPLE_RATE = 150.0   # Hz

TCNN_RUN = ""         # "" = auto-pick the latest runs/TCNN_* run; or pin e.g. "TCNN_20260924_173737"

INTERVAL = None        # frames between fresh prediction blocks; None = checkpoint's own forecast_horizon
                        # (non-overlapping tiling). Smaller = denser/overlapping predictions.
START_SECONDS = 0.0    # where in the trial to start plotting
MAX_SECONDS = None     # None = plot to the end of the trial (or as many blocks as fit)
FEATURE_GROUP = "both"  # "angle", "moment", or "both" (produces two figures)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Overlay a trained TCNN's predictions on ground truth for one subject/trial.")
    p.add_argument("--subject", type=str, default=SUBJECT)
    p.add_argument("--trial", dest="trial_number", type=str, default=TRIAL_NUMBER)
    p.add_argument("--rate", type=str, default=RATE)
    p.add_argument("--run", dest="tcnn_run", type=str, default=TCNN_RUN,
                    help="Pin a specific runs/TCNN_* dir ('' = latest).")
    p.add_argument("--interval", type=int, default=INTERVAL,
                    help="Frames between fresh prediction blocks (default: the checkpoint's own "
                         "forecast_horizon, i.e. non-overlapping tiling).")
    p.add_argument("--start-seconds", type=float, default=START_SECONDS)
    p.add_argument("--seconds", dest="max_seconds", type=float, default=MAX_SECONDS,
                    help="How many seconds to plot from --start-seconds (default: to the end of the trial).")
    p.add_argument("--feature-group", choices=["angle", "moment", "both"], default=FEATURE_GROUP)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_tcnn_model(device, pinned_run=""):
    run_dir, ckpt_path = _find_checkpoint(RUN_PREFIX, CHECKPOINT_STEM, pinned_run)
    print(f"[TCNNEval] Checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    output_dim = ckpt["out_mean"].shape[0]   # read from the checkpoint, not assumed -- see plot_error_across_time.py
    model = TCNModel_Forecast(
        input_size=len(INPUT_COLUMNS), output_size=output_dim,
        horizon=config["forecast_horizon"], num_channels=config["num_channels"],
        kernel_size=config["kernel_size"], dropout=config["dropout"],
    )
    _load_state_dict_flexible(model, ckpt["tcnn_state_dict"])
    model.to(device)
    model.eval()

    norm_stats = {
        "mean_in": ckpt["in_mean"].cpu().numpy(), "std_in": ckpt["in_std"].cpu().numpy(),
        "mean_out": ckpt["out_mean"].cpu().numpy(), "std_out": ckpt["out_std"].cpu().numpy(),
    }
    print(f"[TCNNEval]   epoch={ckpt['epoch']}  history_horizon={config['history_horizon']}  "
          f"forecast_horizon={config['forecast_horizon']}  output_dim={output_dim}")
    return model, config, norm_stats


def _trial_path(subject_name, trial_number, rate):
    path = os.path.join(DATA_DIR, subject_name, rate, f"trial_{trial_number}_all_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- check --subject/--trial/--rate match a file under "
            f"Data/{subject_name}/{rate}/ (e.g. trial_2_all_data.csv -> --trial 2)."
        )
    return path


# ---------------------------------------------------------------------------
# Block-tiled prediction over the whole (or windowed) trial
# ---------------------------------------------------------------------------

def _predict_blocks(model, norm_in, history_horizon, forecast_horizon, device,
                     start_frame, end_frame, interval, batch_size=64):
    """Feed a fresh history_horizon-frame context every `interval` frames,
    starting at start_frame, predicting forecast_horizon genuinely-future
    frames each time, up to end_frame. Returns list of (t_block_frame_idx,
    pred_block) -- pred_block (forecast_horizon, output_dim), NORMALISED
    (denormalised by the caller, same convention as train_tcnn.py's
    _plot_long_trajectory)."""
    H, K = history_horizon, forecast_horizon
    N = norm_in.shape[0]
    max_start = min(end_frame, N) - H - K
    starts = list(range(start_frame, max_start + 1, interval)) if max_start >= start_frame else []
    if not starts:
        raise ValueError(f"No full blocks fit in [{start_frame}, {end_frame}) with H={H}, K={K} -- "
                          f"trial/window too short or start too late.")

    blocks = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(starts), batch_size):
            batch_starts = starts[i:i + batch_size]
            wins = torch.stack([norm_in[s:s + H].T for s in batch_starts]).to(device)   # (B, in_dim, H)
            pred = _forecast(model, wins, K)   # (K, B, out_dim)
            pred = pred.permute(1, 0, 2).cpu().numpy()   # (B, K, out_dim)
            for bi, s in enumerate(batch_starts):
                blocks.append((s + H, pred[bi]))
    return blocks


def _rmse(pred_blocks, gt_norm_out, feature_indices, std):
    """Pooled RMSE (physical units) over every predicted frame in
    pred_blocks, restricted to `feature_indices` -- a quick, printed-only
    sanity number for the plotted span (not a replacement for
    plot_error_across_time.py's full pooled evaluation)."""
    sq_sum = np.zeros(len(feature_indices))
    n = 0
    for frame_start, pred_block in pred_blocks:
        gt_block = gt_norm_out[frame_start:frame_start + pred_block.shape[0]]
        if gt_block.shape[0] < pred_block.shape[0]:
            continue   # trailing partial block past the end of the trial
        sq_sum += ((pred_block[:, feature_indices] - gt_block[:, feature_indices]) ** 2).sum(axis=0)
        n += gt_block.shape[0]
    rmse_norm = np.sqrt(sq_sum / max(n, 1))
    return rmse_norm * std[feature_indices]


# ---------------------------------------------------------------------------
# Plotting -- one figure per modality group, one subplot per joint. Same
# GT-blue/prediction-red-dashed convention as train_tcnn.py's own
# _plot_long_trajectory, adapted here for a standalone (non-training-time)
# evaluation run with a user-chosen start/span/interval.
# ---------------------------------------------------------------------------

def _plot_group(t_gt, gt_plot, pred_blocks_t, pred_blocks_vals, col_names, unit, group_title,
                 trial_id, task_label, history_horizon, forecast_horizon, interval, rmse, save_path):
    n = len(col_names)
    fig, axes = plt.subplots(n, 1, figsize=(16, 2.0 * n), sharex=True)
    axes = np.atleast_1d(axes)
    overlapping = interval < forecast_horizon
    pred_alpha = 0.45 if overlapping else 0.85

    for fi, ax in enumerate(axes):
        ax.plot(t_gt, gt_plot[:, fi], color="steelblue", lw=1.1, alpha=0.95, label="GT", zorder=10)
        for bi, (t_block, pred_block) in enumerate(zip(pred_blocks_t, pred_blocks_vals)):
            ax.plot(t_block, pred_block[:, fi], color="tomato", lw=1.4, alpha=pred_alpha, ls="--",
                    label=("Pred" if bi == 0 else None))
        ax.set_ylabel(f"{col_names[fi][:20]}\n({unit})", fontsize=7, rotation=0, labelpad=95, va="center")
        ax.grid(alpha=0.15)
        ax.tick_params(labelsize=6)
        if fi == 0:
            ax.legend(fontsize=7, loc="upper right", ncol=2)

    axes[-1].set_xlabel("Time (s)")
    rmse_str = "  ".join(f"{c}={r:.3f}{unit}" for c, r in zip(col_names, rmse))
    overlap_str = (f"interval={interval} frames < {forecast_horizon}-frame horizon -> predictions overlap"
                   if overlapping else f"interval={interval} frames = horizon -> predictions tile with no overlap")
    fig.suptitle(
        f"TCNN — {group_title} — {trial_id} ({task_label}) — GT (blue) vs Prediction (red dashed)\n"
        f"{history_horizon} frames context -> {forecast_horizon} genuinely-future frames per block  |  "
        f"{overlap_str}\nRMSE over plotted span: {rmse_str}",
        fontsize=9, fontweight="bold", wrap=True,
    )
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.012 * n])   # leave headroom for the (possibly wrapped) suptitle
    fig.savefig(save_path, dpi=150)
    print(f"[TCNNEval] Saved figure -> {save_path}")
    return fig


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TCNNEval] Using device: {device}")

    model, config, norm_stats = _load_tcnn_model(device, args.tcnn_run)
    H = config["history_horizon"]
    K = config["forecast_horizon"]
    interval = args.interval if args.interval is not None else K

    path = _trial_path(args.subject, args.trial_number, args.rate)
    print(f"[TCNNEval] Trial file: {path}")
    usecols = list(dict.fromkeys(INPUT_COLUMNS + OUTPUT_COLUMNS + ["task"]))
    df = pd.read_csv(path, usecols=usecols)
    task_label = str(df["task"].iloc[0])
    trial_id = f"{args.subject}_trial_{args.trial_number}"
    print(f"[TCNNEval] Trial: {trial_id} (task={task_label}), {len(df):,} rows @{SAMPLE_RATE:.0f}Hz "
          f"({len(df)/SAMPLE_RATE:.1f}s)")

    # Moment columns weight-normalized (N.m -> N.m/kg), same convention as train_tcnn.py's _load_all_data.
    df[MOMENT_COLUMNS] = df[MOMENT_COLUMNS].div(df["subject_weight"], axis=0)

    raw_in = df[INPUT_COLUMNS].values.astype("float32")
    raw_out = df[OUTPUT_COLUMNS].values.astype("float32")
    norm_in = torch.tensor((raw_in - norm_stats["mean_in"]) / norm_stats["std_in"], dtype=torch.float32)
    norm_out = (raw_out - norm_stats["mean_out"]) / norm_stats["std_out"]   # kept as np, for RMSE/plot denorm

    N = len(df)
    start_frame = int(round(args.start_seconds * SAMPLE_RATE))
    end_frame = N if args.max_seconds is None else min(N, start_frame + int(round(args.max_seconds * SAMPLE_RATE)))
    if start_frame >= end_frame:
        raise ValueError(f"--start-seconds {args.start_seconds} leaves no trial left to plot "
                          f"(trial is {N/SAMPLE_RATE:.1f}s long).")

    print(f"[TCNNEval] Plotting frames [{start_frame}, {end_frame}) of {N:,} "
          f"({start_frame/SAMPLE_RATE:.1f}s - {end_frame/SAMPLE_RATE:.1f}s), interval={interval} frames...")
    pred_blocks = _predict_blocks(model, norm_in, H, K, device, start_frame, end_frame, interval)
    print(f"[TCNNEval] {len(pred_blocks)} prediction blocks.")

    std_out = norm_stats["std_out"]
    mean_out = norm_stats["mean_out"]
    t_gt = np.arange(start_frame + H, end_frame) / SAMPLE_RATE
    gt_plot_all = norm_out[start_frame + H:end_frame] * std_out + mean_out   # (T, out_dim) physical units

    pred_blocks_t = [np.arange(s, s + pb.shape[0]) / SAMPLE_RATE for s, pb in pred_blocks]
    pred_blocks_vals = [pb * std_out + mean_out for _s, pb in pred_blocks]

    out_dir = os.path.join(RUNS_DIR, f"TCNNEval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    groups = (["angle", "moment"] if args.feature_group == "both" else [args.feature_group])
    for group_key in groups:
        group = MODALITY_GROUPS[group_key]
        idx = group["indices"]
        col_names = [OUTPUT_COLUMNS[i] for i in idx]
        rmse = _rmse(pred_blocks, norm_out, idx, std_out)
        _plot_group(
            t_gt, gt_plot_all[:, idx],
            pred_blocks_t, [pb[:, idx] for pb in pred_blocks_vals],
            col_names, group["unit"], group["label"], trial_id, task_label, H, K, interval, rmse,
            save_path=os.path.join(out_dir, f"{trial_id}_{group_key}.png"),
        )

    print("[TCNNEval] Displaying figure(s) (close the windows, or Ctrl+C, to exit).")
    plt.show()


if __name__ == "__main__":
    main()
