"""
pae_teacher_student_data_loader.py

Data loader for Models/PAE_teacher_student.py (PAETeacherStudent). Sibling of
DataLoader/teacher_student_data_loader.py (the FLD teacher-student loader),
NOT a modification of it -- runs side by side. Two differences from that
loader, both driven directly by the request this was built for:

  1. SENSOR_COLUMNS (48ch, student's own input) is train_fld.COLUMNS
     verbatim -- body IMU accel+gyro (pelvis/thigh_r/shank_r/thigh_l/shank_l,
     30ch) + insole accel+gyro ("feet" IMU, 12ch) + insole force/COP (6ch).
     teacher_student_data_loader.py's own SENSOR_COLUMNS (36ch) deliberately
     EXCLUDES insole accel/gyro -- this loader deliberately INCLUDES them,
     per explicit request ("imus acc, gyro ... from pelvis, thighs and shanks
     and feet and insole force and cop" -- there is no separately-labeled
     foot IMU in this dataset; the insole IS the foot-mounted sensor, and its
     own accel/gyro columns are exactly train_fld.py's 48-dim COLUMNS minus
     the two SUBJECT_COLUMNS, which this loader also excludes -- teacher-
     student setups in this repo don't condition on subject mass/height, same
     as teacher_student_data_loader.py's own SENSOR_COLUMNS).
  2. No forecast horizon / no `future_biomech`. PAE.py's own Model
     reconstructs the SAME window it encoded (ordinary autoencoder), it does
     not roll phase forward to predict genuinely-future frames the way FLD
     does -- see Models/PAE_teacher_student.py's module docstring. So each
     sample here is just a single time-aligned (sensor_window, biomech_window)
     pair, `window_size == history_horizon` (no `+ forecast_horizon`), and
     biomech_window doubles as both the teacher's own context input AND the
     reconstruction target for both branches.

Reuses BIOMECH_COLUMNS (as train_fld.OUTPUT_COLUMNS), MOMENT_COLUMNS,
_norm_stats_mask, DATA_DIR verbatim from train_fld.py, and
assert_causal_window / _load_split_index verbatim from
teacher_student_data_loader.py (both column-agnostic, unaffected by the
SENSOR_COLUMNS change above -- _load_split_index's window_size compatibility
check is satisfied since this loader passes the SMALLER window_size=H, not
H+K, and the S01 100Hz index was built for window_size=131 >= any reasonable
H here).

--------------------------------------------------------------------------
Data files
--------------------------------------------------------------------------
Two ready-made (data_path, index_path) pairs are defined below:
  ALL_DATA_S01_100HZ_PATH / TRAIN_VAL_INDEX_S01_100HZ_PATH -- S01-only, what
    this loader was originally built against (a smaller/faster dev pair).
  ALL_DATA_100HZ_PATH / TRAIN_VAL_INDEX_100HZ_PATH -- the full 4-subject
    (S01-S04) 100Hz-resampled dataset, same column layout (verified
    identical header to the S01-only file), 46 trials, index built for
    window_size=131 (>= any history_horizon used here). build_datasets()
    itself still takes data_path/index_path as plain args -- these are just
    the two named pairs train_pae_teacher_student.py's CONFIG picks from.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from train_fld import (
    DATA_DIR, COLUMNS as SENSOR_COLUMNS, OUTPUT_COLUMNS as BIOMECH_COLUMNS,
    MOMENT_COLUMNS, _norm_stats_mask,
)
from DataLoader.teacher_student_data_loader import assert_causal_window, _load_split_index

# ---------------------------------------------------------------------------
# Data paths -- 100Hz-resampled files (combine_data.py --rate 100 -> build_train_val_index.py).
# ---------------------------------------------------------------------------
ALL_DATA_S01_100HZ_PATH = os.path.join(DATA_DIR, "all_data_S01_100.csv")               # S01-only (dev/smoke-test pair)
TRAIN_VAL_INDEX_S01_100HZ_PATH = os.path.join(DATA_DIR, "train_val_index_S01_100.csv")

ALL_DATA_100HZ_PATH = os.path.join(DATA_DIR, "all_data_100.csv")                       # full 4-subject (S01-S04) pair
TRAIN_VAL_INDEX_100HZ_PATH = os.path.join(DATA_DIR, "train_val_index_100.csv")

SENSOR_DIM = len(SENSOR_COLUMNS)     # 48 -- body IMU (30) + insole accel/gyro (12) + insole force/COP (6)
BIOMECH_DIM = len(BIOMECH_COLUMNS)   # 30 -- 10 joint angles + 10 angular velocities + 10 moments


def _mean_std(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std, zero-std clamped to 1.0 -- same convention as
    teacher_student_data_loader.py's compute_norm_stats_from_arrays, just a
    single-array helper since this loader only ever normalises SENSOR_COLUMNS
    and BIOMECH_COLUMNS (no third teacher-input array, unlike the 100Hz FLD
    variant -- teacher input IS biomech here, nothing wider)."""
    mean = arr.mean(axis=0).astype('float32')
    std = arr.std(axis=0).astype('float32')
    std[std == 0] = 1.0
    return mean, std


# ---------------------------------------------------------------------------
# Dataset -- one instance per trial.
# ---------------------------------------------------------------------------

class PAETeacherStudentDataset(Dataset):
    """One trial's worth of time-aligned sensor/biomech windows.

    Each sample is a dict:
      'sensor_window'  : (SENSOR_DIM, H)   -- student's own input, channels-first
      'biomech_window' : (BIOMECH_DIM, H)  -- teacher's own input AND the
                          reconstruction target for BOTH branches (PAE
                          autoencodes the window it was given, it does not
                          forecast beyond it -- see module docstring)
      'subject_id'      : str
      'trial_id'        : str
    """

    def __init__(
        self, trial_df: pd.DataFrame, subject_id: str, trial_id: str,
        history_horizon: int, norm_stats: Dict[str, np.ndarray],
        device: Optional[torch.device] = None,
    ) -> None:
        assert history_horizon % 2 == 1, (
            f"history_horizon must be odd (got {history_horizon}) -- PAE's Conv1d layers "
            f"need symmetric same-size padding (kernel_size == time_range)."
        )
        self.subject_id = subject_id
        self.trial_id = trial_id
        self.history_horizon = history_horizon
        self.window_size = history_horizon   # no forecast horizon -- see module docstring
        self.norm_stats = norm_stats

        raw_sensor = trial_df[SENSOR_COLUMNS].values.astype('float32')
        raw_biomech = trial_df[BIOMECH_COLUMNS].values.astype('float32')

        norm_sensor = (raw_sensor - norm_stats['mean_sensor']) / norm_stats['std_sensor']
        norm_biomech = (raw_biomech - norm_stats['mean_biomech']) / norm_stats['std_biomech']

        self.data_sensor = torch.tensor(norm_sensor, dtype=torch.float32)
        self.data_biomech = torch.tensor(norm_biomech, dtype=torch.float32)
        if device is not None:
            self.data_sensor = self.data_sensor.to(device)
            self.data_biomech = self.data_biomech.to(device)

    def __len__(self) -> int:
        return max(0, len(self.data_sensor) - self.window_size + 1)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        H = self.history_horizon
        context_start, context_end = idx, idx + H
        # No genuinely-future rows are read (window_size == H), so causality
        # is trivially satisfied here -- assert_causal_window is still called
        # for consistency with the rest of this repo's loaders / in case a
        # future revision adds a forecast target.
        assert_causal_window(context_start, context_end, context_end)

        sensor_window = self.data_sensor[context_start:context_end].T    # (SENSOR_DIM, H)
        biomech_window = self.data_biomech[context_start:context_end].T  # (BIOMECH_DIM, H)

        return {
            'sensor_window': sensor_window,
            'biomech_window': biomech_window,
            'subject_id': self.subject_id,
            'trial_id': self.trial_id,
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_all_data(data_path: str = ALL_DATA_S01_100HZ_PATH) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found.")
    usecols = list(dict.fromkeys(
        ['trial_id', 'subject_name', 'subject_weight'] + SENSOR_COLUMNS + BIOMECH_COLUMNS
    ))
    df = pd.read_csv(data_path, usecols=usecols)
    df[MOMENT_COLUMNS] = df[MOMENT_COLUMNS].div(df['subject_weight'], axis=0)
    return df


def build_datasets(
    history_horizon: int,
    subjects: Optional[List[str]] = None,
    data_path: str = ALL_DATA_S01_100HZ_PATH,
    index_path: str = TRAIN_VAL_INDEX_S01_100HZ_PATH,
    device: Optional[torch.device] = None,
) -> Tuple[ConcatDataset, ConcatDataset, Dict[str, np.ndarray], Dict[str, PAETeacherStudentDataset]]:
    """Same contract as teacher_student_data_loader.py's build_datasets() --
    train/val split BY WINDOW WITHIN EACH TRIAL via index_path, norm stats fit
    on train-window rows only. window_size here is just history_horizon (no
    `+ forecast_horizon` -- see module docstring)."""
    df = load_all_data(data_path)
    if subjects:
        available = set(df['subject_name'].unique())
        missing = [s for s in subjects if s not in available]
        if missing:
            raise ValueError(f"subjects {missing} not found in {data_path}. "
                              f"Available: {sorted(available)}")
        df = df.loc[df['subject_name'].isin(subjects)].reset_index(drop=True)

    window_size = history_horizon
    split_index = _load_split_index(index_path, window_size)

    trial_dfs = {str(tid): g.reset_index(drop=True) for tid, g in df.groupby('trial_id', sort=False)}

    train_raw_sensor, train_raw_biomech = [], []
    for trial_id, trial_df in trial_dfs.items():
        if trial_id not in split_index:
            continue
        mask = _norm_stats_mask(len(trial_df), split_index[trial_id]['val'], window_size)
        train_raw_sensor.append(trial_df.loc[mask, SENSOR_COLUMNS].values.astype('float64'))
        train_raw_biomech.append(trial_df.loc[mask, BIOMECH_COLUMNS].values.astype('float64'))

    if not train_raw_sensor:
        raise ValueError(f"No trial in {data_path} (subjects={subjects}) was found in {index_path} -- "
                          f"nothing to fit normalisation stats on.")

    mean_sensor, std_sensor = _mean_std(np.concatenate(train_raw_sensor, axis=0))
    mean_biomech, std_biomech = _mean_std(np.concatenate(train_raw_biomech, axis=0))
    norm_stats = {
        'mean_sensor': mean_sensor, 'std_sensor': std_sensor,
        'mean_biomech': mean_biomech, 'std_biomech': std_biomech,
    }

    trial_datasets: Dict[str, PAETeacherStudentDataset] = {}
    train_subsets, val_subsets = [], []
    for trial_id, trial_df in trial_dfs.items():
        if trial_id not in split_index:
            print(f"[pae_teacher_student_data_loader] {trial_id}: not in {index_path}, skipping.")
            continue
        train_idx = split_index[trial_id]['train']
        val_idx = split_index[trial_id]['val']
        if not train_idx or not val_idx:
            print(f"[pae_teacher_student_data_loader] {trial_id}: missing train or val windows "
                  f"(train={len(train_idx)}, val={len(val_idx)}), skipping.")
            continue
        subject_id = str(trial_df['subject_name'].iloc[0])
        dataset = PAETeacherStudentDataset(
            trial_df, subject_id, trial_id, history_horizon, norm_stats, device=device,
        )
        trial_datasets[trial_id] = dataset
        train_subsets.append(Subset(dataset, train_idx))
        val_subsets.append(Subset(dataset, val_idx))

    if not train_subsets:
        raise ValueError("No usable train/val windows -- check subjects/index_path/history_horizon.")

    print(f"[pae_teacher_student_data_loader] train: {sum(len(s) for s in train_subsets):,} windows "
          f"across {len(train_subsets)} trials")
    print(f"[pae_teacher_student_data_loader] val:   {sum(len(s) for s in val_subsets):,} windows "
          f"across {len(val_subsets)} trials")

    return ConcatDataset(train_subsets), ConcatDataset(val_subsets), norm_stats, trial_datasets
