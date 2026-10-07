"""
teacher_student_100hz_data_loader.py

Same-shape-family variant of DataLoader/teacher_student_data_loader.py, for
a 100 Hz-resampled dataset: STUDENT (sensor) side drops insole force/COP
entirely, keeping only body IMU. TEACHER side's input is BIOMECH_COLUMNS,
UNCHANGED from the original (angle+velocity+moment, 30ch) -- matching the
shared decoder's own 30ch output exactly.

  SENSOR_COLUMNS       (30): BODY_IMU_COLUMNS only, reused verbatim from
                        teacher_student_data_loader.py (pelvis/thigh_r/
                        shank_r/thigh_l/shank_l, accel+gyro). NO insole, NO
                        EMG. "hip" in the original request maps to the THIGH
                        sensor here -- this dataset has no separately-labeled
                        hip IMU, thigh is the closest physical sensor to the
                        hip and pelvis/shank are unambiguous, so this is
                        pelvis+thigh+shank, not a 4th sensor location.
  TEACHER_INPUT_COLUMNS (30): == BIOMECH_COLUMNS exactly -- what the TEACHER
                        ENCODER reads. Was briefly BIOMECH_COLUMNS + 8ch EMG
                        (38ch total) as extra privileged teacher-only
                        conditioning; REVERTED to biomech-only per explicit
                        request. EMG_COLUMNS is still defined below (a
                        checked, valid column list -- see its own comment)
                        in case a future revision wants it back, it's just
                        not part of TEACHER_INPUT_COLUMNS any more.
                        IMPORTANT: runs/FLDTeacherStudent_100Hz_IMUOnly_20260903_173813
                        was already trained under the OLD (38ch) default --
                        this reversion does not change that checkpoint, and
                        scripts pinned to it (plot_pca_manifold_train_100hz.py,
                        plot_pca_manifolds_all_tasks_100hz.py) hardcode their
                        own PINNED_TEACHER_INPUT_COLUMNS rather than importing
                        this constant, so they keep working correctly.
  BIOMECH_COLUMNS      (30): UNCHANGED -- imported directly from
                        teacher_student_data_loader.py. Both the shared
                        decoder's output (future_biomech target) AND, once
                        again, the teacher's own context input.

Does NOT modify teacher_student_data_loader.py or train_fld.py -- runs side
by side with them. Models/FLD_teacher_student.py still carries the optional
teacher_input_dim constructor arg (added for the since-reverted EMG
experiment) -- harmless, backward-compatible infrastructure left in place;
with TEACHER_INPUT_COLUMNS == BIOMECH_COLUMNS again, passing
teacher_input_dim=TEACHER_INPUT_DIM (30) is simply a no-op identical to
omitting it.

Reuses BIOMECH_COLUMNS / MOMENT_COLUMNS / BODY_IMU_COLUMNS / assert_causal_window /
_load_split_index verbatim (column-agnostic or biomech-only, unaffected by
this loader's changes). compute_norm_stats_from_arrays is NOT reused here --
it only handles a (sensor, biomech) pair, and this loader needs THREE
independently-normalised tensors (sensor/teacher_input/biomech) -- see
_mean_std() below, a small local per-array helper instead.

--------------------------------------------------------------------------
Data files
--------------------------------------------------------------------------
This loader defaults to Data/all_data_100.csv / Data/train_val_index_100.csv
-- the 100 Hz-resampled counterpart to the original 150 Hz Data/all_data.csv /
Data/train_val_index.csv, produced by combine_data.py --rate 100 ->
build_train_val_index.py. Both now exist and have been used for a real
training run (runs/FLDTeacherStudent_100Hz_IMUOnly_20260903_173813). If the
actual resampled files ever end up with different names, update
ALL_DATA_100HZ_PATH / TRAIN_VAL_INDEX_100HZ_PATH below (or pass
--data-path/--index-path to train_fld_teacher_student_100hz.py) rather than
editing this docstring's assumption in place.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from train_fld import DATA_DIR, MOMENT_COLUMNS, _norm_stats_mask
from DataLoader.teacher_student_data_loader import (
    BODY_IMU_COLUMNS, BIOMECH_COLUMNS, BIOMECH_DIM,             # unchanged, reused verbatim
    assert_causal_window, _load_split_index,                    # column-agnostic, reused verbatim
)

# ---------------------------------------------------------------------------
# Data paths -- 100 Hz-resampled counterparts to Data/all_data.csv /
# Data/train_val_index.csv. Do not exist yet -- see module docstring.
# ---------------------------------------------------------------------------
ALL_DATA_100HZ_PATH = os.path.join(DATA_DIR, "all_data_100.csv")             # matches combine_data.py --rate 100's output tag
TRAIN_VAL_INDEX_100HZ_PATH = os.path.join(DATA_DIR, "train_val_index_100.csv")

# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------

SENSOR_COLUMNS = list(BODY_IMU_COLUMNS)   # 30 -- pelvis/thigh("hip")/shank IMU only, NO insole, NO EMG
SENSOR_DIM = len(SENSOR_COLUMNS)

EMG_COLUMNS = [
    'R_RF', 'R_BF', 'R_TA', 'R_GAST',   # right: rectus femoris, biceps femoris, tibialis anterior, gastrocnemius
    'L_RF', 'L_BF', 'L_TA', 'L_GAST',   # left: same 4 muscles
]  # 8 -- already a linear-envelope signal in the raw data (see module docstring's history below) --
   # kept defined (still a valid, checked column list) even though EMG is no longer in
   # TEACHER_INPUT_COLUMNS, in case a future revision wants it back.

# REVERTED: EMG removed from the teacher's input, per explicit request --
# TEACHER_INPUT_COLUMNS is back to exactly BIOMECH_COLUMNS (30 == BIOMECH_DIM),
# i.e. teacher_input_dim == biomech_dim again, matching FLDTeacherStudent's
# own teacher_input_dim=None default (see Models/FLD_teacher_student.py).
# NOTE: this does NOT retroactively change any already-trained checkpoint --
# runs/FLDTeacherStudent_100Hz_IMUOnly_20260903_173813 was trained WITH the
# 38ch (biomech+EMG) teacher input under the old default; scripts pinned to
# that specific checkpoint (plot_pca_manifold_train_100hz.py,
# plot_pca_manifolds_all_tasks_100hz.py) hardcode their own
# PINNED_TEACHER_INPUT_COLUMNS rather than importing this constant, precisely
# so they keep working correctly after this change. Only a NEW training run
# (train_fld_teacher_student_100hz.py) will use this reverted, EMG-less value.
TEACHER_INPUT_COLUMNS = list(BIOMECH_COLUMNS)   # 30 -- teacher encoder's own input, NOT the decode target
TEACHER_INPUT_DIM = len(TEACHER_INPUT_COLUMNS)


def _mean_std(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std with the same zero-std->1.0 clamp
    compute_norm_stats_from_arrays uses, duplicated here in miniature since
    that shared function only handles a (sensor, biomech) pair and this
    loader needs three independently-normalised tensors."""
    mean = arr.mean(axis=0).astype('float32')
    std = arr.std(axis=0).astype('float32')
    std[std == 0] = 1.0
    return mean, std


# ---------------------------------------------------------------------------
# Dataset -- one instance per trial. THREE separate tensors now, not two:
# data_sensor (student input, 30ch IMU), data_teacher_input (teacher's own
# context input, 38ch biomech+EMG), data_biomech (decode target /
# future_biomech ground truth, 30ch, UNCHANGED width).
# ---------------------------------------------------------------------------

class TeacherStudent100HzFLDDataset(Dataset):
    """One trial's worth of time-aligned sensor/teacher-input/biomech windows.

    Each sample is a dict:
      'sensor_window'  : (SENSOR_DIM, H)         -- student's own input, IMU-only
      'biomech_window' : (TEACHER_INPUT_DIM, H)  -- TEACHER's own input (biomech+EMG,
                          38ch) -- key name kept as 'biomech_window' for drop-in
                          compatibility with train_fld_teacher_student_100hz.py's
                          existing train() loop, even though it's wider than
                          BIOMECH_DIM now; it is NOT the decode target.
      'future_biomech'  : (K, BIOMECH_DIM)       -- UNCHANGED, 30ch angle+velocity+
                          moment, the propagation-loss target for BOTH branches
      'subject_id'      : str
      'trial_id'        : str
    """

    def __init__(
        self, trial_df: pd.DataFrame, subject_id: str, trial_id: str,
        history_horizon: int, forecast_horizon: int, norm_stats: Dict[str, np.ndarray],
        device: Optional[torch.device] = None,
    ) -> None:
        assert history_horizon % 2 == 1, (
            f"history_horizon must be odd (got {history_horizon}) -- FLD's Conv1d layers "
            f"need symmetric same-size padding."
        )
        self.subject_id = subject_id
        self.trial_id = trial_id
        self.history_horizon = history_horizon
        self.forecast_horizon = forecast_horizon
        self.window_size = history_horizon + forecast_horizon
        self.norm_stats = norm_stats

        raw_sensor = trial_df[SENSOR_COLUMNS].values.astype('float32')
        raw_teacher_input = trial_df[TEACHER_INPUT_COLUMNS].values.astype('float32')
        raw_biomech = trial_df[BIOMECH_COLUMNS].values.astype('float32')

        norm_sensor = (raw_sensor - norm_stats['mean_sensor']) / norm_stats['std_sensor']
        norm_teacher_input = (raw_teacher_input - norm_stats['mean_teacher_input']) / norm_stats['std_teacher_input']
        norm_biomech = (raw_biomech - norm_stats['mean_biomech']) / norm_stats['std_biomech']

        self.data_sensor = torch.tensor(norm_sensor, dtype=torch.float32)
        self.data_teacher_input = torch.tensor(norm_teacher_input, dtype=torch.float32)
        self.data_biomech = torch.tensor(norm_biomech, dtype=torch.float32)
        if device is not None:
            self.data_sensor = self.data_sensor.to(device)
            self.data_teacher_input = self.data_teacher_input.to(device)
            self.data_biomech = self.data_biomech.to(device)

    def __len__(self) -> int:
        return max(0, len(self.data_sensor) - self.window_size + 1)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        H, K = self.history_horizon, self.forecast_horizon
        context_start, context_end = idx, idx + H
        future_start = idx + H
        assert_causal_window(context_start, context_end, future_start)

        sensor_window = self.data_sensor[context_start:context_end].T             # (SENSOR_DIM, H)
        biomech_window = self.data_teacher_input[context_start:context_end].T     # (TEACHER_INPUT_DIM, H) -- teacher's input
        future_biomech = self.data_biomech[future_start:future_start + K]         # (K, BIOMECH_DIM) -- decode target, unchanged

        return {
            'sensor_window': sensor_window,
            'biomech_window': biomech_window,
            'future_biomech': future_biomech,
            'subject_id': self.subject_id,
            'trial_id': self.trial_id,
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_all_data(data_path: str = ALL_DATA_100HZ_PATH) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{data_path} not found. This is the 100 Hz-resampled dataset, which as of writing "
            f"this loader has NOT been produced yet -- run your resampling pipeline (then "
            f"combine_data.py / build_train_val_index.py against its output) first, or pass "
            f"--data-path/--index-path pointing at wherever the resampled files actually land."
        )
    usecols = list(dict.fromkeys(
        ['trial_id', 'subject_name', 'subject_weight'] + SENSOR_COLUMNS + TEACHER_INPUT_COLUMNS
    ))   # TEACHER_INPUT_COLUMNS already contains every BIOMECH_COLUMNS entry, so this covers both
    df = pd.read_csv(data_path, usecols=usecols)
    df[MOMENT_COLUMNS] = df[MOMENT_COLUMNS].div(df['subject_weight'], axis=0)
    return df


def build_datasets(
    history_horizon: int,
    forecast_horizon: int,
    subjects: Optional[List[str]] = None,
    data_path: str = ALL_DATA_100HZ_PATH,
    index_path: str = TRAIN_VAL_INDEX_100HZ_PATH,
    device: Optional[torch.device] = None,
) -> Tuple[ConcatDataset, ConcatDataset, Dict[str, np.ndarray], Dict[str, TeacherStudent100HzFLDDataset]]:
    """Same contract as teacher_student_data_loader.py's build_datasets() --
    see its docstring for the full train/val-by-window-within-trial /
    norm-stats-fit-on-train-only explanation (identical here). norm_stats now
    carries THREE pairs (mean/std for sensor, teacher_input, biomech) instead
    of two -- see _mean_std()."""
    df = load_all_data(data_path)
    if subjects:
        available = set(df['subject_name'].unique())
        missing = [s for s in subjects if s not in available]
        if missing:
            raise ValueError(f"subjects {missing} not found in {data_path}. "
                              f"Available: {sorted(available)}")
        df = df.loc[df['subject_name'].isin(subjects)].reset_index(drop=True)

    window_size = history_horizon + forecast_horizon
    split_index = _load_split_index(index_path, window_size)

    trial_dfs = {str(tid): g.reset_index(drop=True) for tid, g in df.groupby('trial_id', sort=False)}

    train_raw_sensor, train_raw_teacher_input, train_raw_biomech = [], [], []
    for trial_id, trial_df in trial_dfs.items():
        if trial_id not in split_index:
            continue
        mask = _norm_stats_mask(len(trial_df), split_index[trial_id]['val'], window_size)
        train_raw_sensor.append(trial_df.loc[mask, SENSOR_COLUMNS].values.astype('float64'))
        train_raw_teacher_input.append(trial_df.loc[mask, TEACHER_INPUT_COLUMNS].values.astype('float64'))
        train_raw_biomech.append(trial_df.loc[mask, BIOMECH_COLUMNS].values.astype('float64'))

    if not train_raw_sensor:
        raise ValueError(f"No trial in {data_path} (subjects={subjects}) was found in {index_path} -- "
                          f"nothing to fit normalisation stats on.")

    mean_sensor, std_sensor = _mean_std(np.concatenate(train_raw_sensor, axis=0))
    mean_teacher_input, std_teacher_input = _mean_std(np.concatenate(train_raw_teacher_input, axis=0))
    mean_biomech, std_biomech = _mean_std(np.concatenate(train_raw_biomech, axis=0))
    norm_stats = {
        'mean_sensor': mean_sensor, 'std_sensor': std_sensor,
        'mean_teacher_input': mean_teacher_input, 'std_teacher_input': std_teacher_input,
        'mean_biomech': mean_biomech, 'std_biomech': std_biomech,
    }

    trial_datasets: Dict[str, TeacherStudent100HzFLDDataset] = {}
    train_subsets, val_subsets = [], []
    for trial_id, trial_df in trial_dfs.items():
        if trial_id not in split_index:
            print(f"[teacher_student_100hz_data_loader] {trial_id}: not in {index_path}, skipping.")
            continue
        train_idx = split_index[trial_id]['train']
        val_idx = split_index[trial_id]['val']
        if not train_idx or not val_idx:
            print(f"[teacher_student_100hz_data_loader] {trial_id}: missing train or val windows "
                  f"(train={len(train_idx)}, val={len(val_idx)}), skipping.")
            continue
        subject_id = str(trial_df['subject_name'].iloc[0])
        dataset = TeacherStudent100HzFLDDataset(
            trial_df, subject_id, trial_id, history_horizon, forecast_horizon, norm_stats, device=device,
        )
        trial_datasets[trial_id] = dataset
        train_subsets.append(Subset(dataset, train_idx))
        val_subsets.append(Subset(dataset, val_idx))

    if not train_subsets:
        raise ValueError("No usable train/val windows -- check subjects/index_path/history_horizon.")

    print(f"[teacher_student_100hz_data_loader] train: {sum(len(s) for s in train_subsets):,} windows "
          f"across {len(train_subsets)} trials")
    print(f"[teacher_student_100hz_data_loader] val:   {sum(len(s) for s in val_subsets):,} windows "
          f"across {len(val_subsets)} trials")

    return ConcatDataset(train_subsets), ConcatDataset(val_subsets), norm_stats, trial_datasets
