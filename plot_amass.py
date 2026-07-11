"""
Temporary data inspection script for the CMU AMASS dataset.

Run from the project root:
    python plot_amass.py
    python plot_amass.py --clip_idx 5          # pick a different clip
    python plot_amass.py --lower_body_only     # 7 local lower-body joints
    python plot_amass.py --subject 06          # filter to a specific subject

Plots are saved to Data/amass_preview.png and also displayed if a display is available.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from DataLoader.amass_loader import (
    CMUAMASSDataset,
    LOWER_BODY_JOINT_NAMES,
    SMPL_H_JOINT_NAMES,
    VALID_MOTION_TYPES,
)

ROOT = os.path.join(os.path.dirname(__file__), "Data", "AMASS Dataset", "CMU", "CMU")

# ALL_TYPES = ["walk", "run", "jog", "skip", "crawl",
#              "varied_terrain", "obstacle_course", "sit_to_stand"]
ALL_TYPES = ["walk", "run", "jog"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clip_idx",       type=int,  default=0,
                   help="Index of the clip to plot (default 0)")
    p.add_argument("--lower_body_only", action="store_true", default=True,
                   help="Show only the 7 lower-body joints (default True)")
    p.add_argument("--subject",        type=str,  default=None,
                   help="Restrict to a single subject folder, e.g. '06'")
    p.add_argument("--euler",          type=str,  default="ZYX",
                   help="Euler sequence (default ZYX)")
    return p.parse_args()


def main():
    args = parse_args()

    ds = CMUAMASSDataset(
        root_path=ROOT,
        motion_types=ALL_TYPES,
        lower_body_only=args.lower_body_only,
        euler_sequence=args.euler,
    )
    print(ds.summary())

    # optionally filter to a single subject
    clips = ds.clips
    if args.subject:
        clips = [c for c in clips if c["subject"] == args.subject]
        if not clips:
            print(f"No clips found for subject '{args.subject}'. Available: "
                  f"{sorted(ds.subject_clip_counts())}")
            sys.exit(1)
        ds.clips = clips

    idx = args.clip_idx % len(ds)
    sample = ds[idx]

    poses = sample["poses"].numpy()          # (T, D) Euler degrees
    fps   = sample["mocap_framerate"]
    T     = poses.shape[0]
    time  = np.arange(T) / fps

    joint_names = LOWER_BODY_JOINT_NAMES if args.lower_body_only else SMPL_H_JOINT_NAMES[:22]
    n_joints    = len(joint_names)
    D           = poses.shape[1]
    n_plot      = min(n_joints, D // 3)      # actual joints available in poses

    title = (f"CMU AMASS — {sample['subject']}/{sample['clip_name']}  "
             f"({T} frames @ {fps:.0f} Hz, {T/fps:.1f} s)  "
             f"[Euler {args.euler}, degrees]")

    # -----------------------------------------------------------------------
    # Figure layout: one row per joint, 3 columns (X, Y, Z Euler)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(n_plot, 1, figsize=(14, 2.2 * n_plot), sharex=True)
    fig.suptitle(title, fontsize=11, fontweight="bold")

    if n_plot == 1:
        axes = [axes]

    axis_labels = ["Z", "Y", "X"] if args.euler == "ZYX" else list(args.euler)
    colors = ["#e74c3c", "#2ecc71", "#3498db"]

    for j, (ax, name) in enumerate(zip(axes, joint_names[:n_plot])):
        for a, (albl, col) in enumerate(zip(axis_labels, colors)):
            ax.plot(time, poses[:, j * 3 + a],
                    label=f"{albl}", color=col, lw=1.1)
        ax.set_ylabel(name, fontsize=8, rotation=0, labelpad=80, va="center")
        ax.legend(loc="upper right", fontsize=7, ncol=3, framealpha=0.5)
        ax.grid(alpha=0.25)
        ax.axhline(0, color="k", lw=0.4, ls="--")

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
