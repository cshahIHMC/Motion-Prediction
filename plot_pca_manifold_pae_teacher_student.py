"""
plot_pca_manifold_pae_teacher_student.py

Standalone PCA-manifold plotting script for a trained
Models/PAE_teacher_student.py (PAETeacherStudent) checkpoint -- does NOT
retrain, only loads a checkpoint written by train_pae_teacher_student.py and
plots. Two modes, both teacher vs. student overlaid on a SINGLE shared PCA
fit (never one independent PCA fit per trial):

  - TASK_NAMES = [] (default): ONE COMBINED PLOT pooling EVERY trial in the
    checkpoint's own data_path.
  - TASK_NAMES = ["squats", ...]: ONE PLOT PER LISTED TASK -- per explicit
    request ("plotting individual task"), each figure pools only the
    trial(s) whose `task` column matches that name (e.g. "squats" pools
    every subject's squats trial), so a single gait/motion regime can be
    inspected on its own instead of blurred into the all-trials cloud. See
    _trial_task_map() below for exactly which task names exist (they differ
    slightly for S01 vs. S02-S04 -- see that function's docstring) and
    _filter_trial_datasets_by_task() for the pooling-restriction itself.

Reuses, not reimplements:
  - _find_checkpoint / _load_state_dict_flexible (plot_error_across_time.py)
    -- same newest-run-first / highest-epoch-checkpoint lookup every other
    plotting script in this repo uses (see README's "Known issues" for its
    one documented alphabetical-sort caveat -- doesn't bite here: every
    PAETeacherStudent_100Hz_<timestamp> run this script has seen so far
    sorts correctly newest-first because same-day timestamps compare
    correctly as strings).
  - _plot_pca_manifold_all_trials -- imported directly from
    train_pae_teacher_student.py, the EXACT function its own end-of-training
    plot calls. Called once for the combined mode, or once PER task (each
    time on a task-filtered `trial_datasets` dict) for the per-task mode --
    the function itself is untouched either way, so a task's figure is
    guaranteed to look exactly like what pooling just that task's trials
    through the all-trials plot would produce.
  - build_datasets (DataLoader/pae_teacher_student_data_loader.py) to rebuild
    the per-trial datasets (needed for manifold encoding) from whichever
    data/index paths + subjects the checkpoint's own saved config recorded --
    NOT the module's current defaults, so this stays correct even if a
    future checkpoint is trained against a different data/index pair.

plt.show() -- interactive, nothing saved to disk (same convention as
plot_pca_manifold_train_100hz.py / plot_pca_manifolds_all_tasks_100hz.py).
One figure window per task requested (or the one combined figure).

Run:
    python3 plot_pca_manifold_pae_teacher_student.py
    # one figure per task instead of the combined plot -- edit TASK_NAMES
    # below, or pass --task on the command line (repeatable):
    python3 plot_pca_manifold_pae_teacher_student.py --task squats --task standing
    # pin a specific run instead of the newest PAETeacherStudent_* one:
    python3 plot_pca_manifold_pae_teacher_student.py --run PAETeacherStudent_100Hz_20260910_213050
"""

import argparse
import os
import sys
from typing import Dict, List

import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from plot_error_across_time import _find_checkpoint, _load_state_dict_flexible   # reused, not reimplemented
from Models.PAE_teacher_student import PAETeacherStudent
from DataLoader.pae_teacher_student_data_loader import build_datasets, SENSOR_DIM, BIOMECH_DIM
from train_pae_teacher_student import _plot_pca_manifold_all_trials   # reused, not reimplemented
from Library.fld_plotter import Plotter

RUN_PREFIX = "PAETeacherStudent"          # matches train_pae_teacher_student.py's log_dir tag prefix
CHECKPOINT_STEM = "pae_teacher_student"   # matches train_pae_teacher_student.py's _save() filename stem

PCA_MAX_WINDOWS_PER_TRIAL = 1500   # cap on manifold points per trial (before pooling into a plot)

# [] = the original combined-across-everything plot. Non-empty = one plot
# PER listed task name instead, each pooling only that task's own trial(s).
# Task names differ slightly between S01 and S02-S04 -- see
# _trial_task_map()'s docstring, or just run this script once with an
# unrecognised name to have it print every available (trial_id, task) pair.
TASK_NAMES: List[str] = []


def _load_model(device, pinned_run=""):
    run_dir, ckpt_path = _find_checkpoint(RUN_PREFIX, CHECKPOINT_STEM, pinned_run)
    print(f"[PCA-PAE-TS] Checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]   # dict -- asdict(PAETeacherStudentConfig) at save time
    model = PAETeacherStudent(
        sensor_dim=SENSOR_DIM, biomech_dim=BIOMECH_DIM,
        history_horizon=config["history_horizon"], latent_dim=config["latent_dim"],
        device=device, dt=1.0 / config["sample_rate"],
        teacher_intermediate_channels=config["teacher_intermediate_channels"],
        student_intermediate_channels=config["student_intermediate_channels"],
    )
    _load_state_dict_flexible(model, ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"[PCA-PAE-TS]   epoch={ckpt['epoch']}  history_horizon={config['history_horizon']}  "
          f"latent_dim={config['latent_dim']}  sample_rate={config['sample_rate']}  "
          f"teacher_intermediate_channels={config['teacher_intermediate_channels']}  "
          f"student_intermediate_channels={config['student_intermediate_channels']}")
    return model, config


def _trial_task_map(data_path: str) -> Dict[str, str]:
    """{trial_id: task}, read cheaply (just these two columns, not the full
    ~180-column row) from `data_path`. `task` is constant within a trial (one
    recording = one task), so the first row per trial_id is enough --
    combine_data.py/train_fld.py never assume otherwise either.

    Task names differ slightly for S01 vs. S02-S04 (see README.md's
    "Collected data, in detail" section, and combine_data.py's own per-
    subject CSVs) -- e.g. S01 has "backward_walking_0_6_m_s" where S02-S04
    have "backward_walking_mix_speed". A TASK_NAMES entry only matches
    trials whose task column is EXACTLY that string -- to pool "backward
    walking regardless of subject naming", list both variants.
    """
    df = pd.read_csv(data_path, usecols=["trial_id", "task"])
    return df.drop_duplicates("trial_id").set_index("trial_id")["task"].to_dict()


def _filter_trial_datasets_by_task(trial_datasets: Dict, task_map: Dict[str, str], task_name: str) -> Dict:
    """Subset of `trial_datasets` whose trial_id maps to exactly `task_name`
    in `task_map` -- the "individual task" restriction _plot_pca_manifold_
    all_trials is then called against instead of the full dict, so it pools
    (and fits ONE shared PCA over) only that task's own trial(s)."""
    return {tid: ds for tid, ds in trial_datasets.items() if task_map.get(tid) == task_name}


def main():
    p = argparse.ArgumentParser(
        description="Plot PCA phase manifolds (teacher vs student, one combined plot -- across "
                    "everything, or per requested task) for a trained PAETeacherStudent checkpoint.")
    p.add_argument("--run", type=str, default="",
                    help="Pin a specific runs/<...> dir (default: newest PAETeacherStudent_* run).")
    p.add_argument("--task", type=str, action="append", default=None,
                    help="Plot only this task (repeatable for several tasks, one figure each) "
                         "instead of TASK_NAMES/the combined plot. Run once with a bogus name to "
                         "have it print every available task.")
    args = p.parse_args()
    task_names = args.task if args.task is not None else TASK_NAMES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PCA-PAE-TS] Using device: {device}")

    model, config = _load_model(device, pinned_run=args.run)

    data_device = device if device.type == "cuda" else None
    _train_ds, _val_ds, _norm_stats, trial_datasets = build_datasets(
        config["history_horizon"], subjects=config["subjects"],
        data_path=config["data_path"], index_path=config["index_path"], device=data_device,
    )

    plotter = Plotter()

    if not task_names:
        print(f"[PCA-PAE-TS] Combined mode -- pooling all {len(trial_datasets)} trials into one plot.")
        _plot_pca_manifold_all_trials(
            model, trial_datasets, device, config["history_horizon"], plotter,
            max_windows_per_trial=PCA_MAX_WINDOWS_PER_TRIAL,
        )
    else:
        task_map = _trial_task_map(config["data_path"])
        available = sorted(set(task_map.values()))
        for task_name in task_names:
            subset = _filter_trial_datasets_by_task(trial_datasets, task_map, task_name)
            if not subset:
                print(f"[PCA-PAE-TS] WARNING -- no trial has task == '{task_name}', skipping. "
                      f"Available tasks: {available}")
                continue
            print(f"[PCA-PAE-TS] Task '{task_name}': pooling {len(subset)} trial(s) "
                  f"({sorted(subset.keys())}) into one plot.")
            fig = _plot_pca_manifold_all_trials(
                model, subset, device, config["history_horizon"], plotter,
                max_windows_per_trial=PCA_MAX_WINDOWS_PER_TRIAL,
            )
            fig.axes[0].set_title(
                f"PCA Phase Manifold — Teacher vs Student — task: {task_name} "
                f"({len(subset)} trial(s))",
                fontsize=12, fontweight="bold",
            )

    print("[PCA-PAE-TS] Displaying figure(s) (close each window, or Ctrl+C, to exit).")
    plt.show()


if __name__ == "__main__":
    main()
