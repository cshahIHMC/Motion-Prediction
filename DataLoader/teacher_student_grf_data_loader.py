"""
teacher_student_grf_data_loader.py

Same-shape variant of DataLoader/teacher_student_data_loader.py: identical
STUDENT (sensor) side, but the TEACHER side's biomechanics representation
swaps the 10 joint MOMENT columns for 10 ground-reaction-force / center-of-
pressure columns (GRF Fx/Fy/Fz + COP Cx/Cy, both force plates) -- so the
teacher encodes, and the shared decoder reconstructs, force-plate signals
instead of inverse-dynamics joint moments. Angles and velocities unchanged.

  BIOMECH_COLUMNS (30): 10 joint angles + 10 joint angular velocities (same
                        columns as teacher_student_data_loader.py, reused
                        via train_fld.OUTPUT_COLUMNS[0:20]) + 10 GRF/COP
                        columns (GRF_COP_COLUMNS below) -- replacing the 10
                        *_moment columns.
  SENSOR_COLUMNS (36): UNCHANGED -- imported directly from
                        teacher_student_data_loader.py, not redefined.

GRF force components (Fx/Fy/Fz, 6 of the 10) are divided by subject_weight
in load_all_data() -- N -> N/kg, same weight-normalization convention this
repo already applies to joint moments (comparable across subjects). COP
components (Cx/Cy, 4 of the 10) are left raw -- a position, not a force-like
quantity; nothing else in this repo weight-normalizes a position. L/R
*_plate_Cz is excluded (checked on real data: always exactly 0.0 -- a flat
plate has no meaningful out-of-plane COP).

Does NOT modify teacher_student_data_loader.py or train_fld.py -- runs side
by side with them. Reuses SENSOR_COLUMNS / SENSOR_DIM / assert_causal_window /
compute_norm_stats_from_arrays / _load_split_index verbatim (none of those
reference the biomech column list internally, so they're fully reusable
as-is, imported not reimplemented); only the pieces that actually depend on
BIOMECH_COLUMNS (the Dataset class, load_all_data, build_datasets) are
redefined here, mirroring the originals line-for-line except for that swap.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from train_fld import (
    ALL_DATA_PATH, INDEX_PATH as TRAIN_VAL_INDEX_PATH, OUTPUT_COLUMNS, _norm_stats_mask,
)
from DataLoader.teacher_student_data_loader import (
    SENSOR_COLUMNS, SENSOR_DIM,                                  # unchanged student side, reused verbatim
    assert_causal_window, compute_norm_stats_from_arrays, _load_split_index,   # column-agnostic, reused verbatim
    ALL_DATA_1_SUBJECT_PATH, TRAIN_VAL_INDEX_1_SUBJECT_PATH,      # dev-file convenience, re-exported
    ALL_DATA_2_SUBJECTS_PATH, TRAIN_VAL_INDEX_2_SUBJECTS_PATH,
)

# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------

ANGLE_VELOCITY_COLUMNS = OUTPUT_COLUMNS[0:20]   # 10 angles + 10 velocities, reused verbatim from train_fld.py

GRF_FORCE_COLUMNS = [   # weight-normalized (N -> N/kg), same convention as MOMENT_COLUMNS
    'L_plate_Fx', 'L_plate_Fy', 'L_plate_Fz',
    'R_plate_Fx', 'R_plate_Fy', 'R_plate_Fz',
]  # 6
COP_COLUMNS = [   # raw, NOT weight-normalized (position, not force) -- Cz excluded, always 0 on a flat plate
    'L_plate_Cx', 'L_plate_Cy',
    'R_plate_Cx', 'R_plate_Cy',
]  # 4
GRF_COP_COLUMNS = GRF_FORCE_COLUMNS + COP_COLUMNS   # 10 -- replaces MOMENT_COLUMNS

BIOMECH_COLUMNS = ANGLE_VELOCITY_COLUMNS + GRF_COP_COLUMNS   # 30
BIOMECH_DIM = len(BIOMECH_COLUMNS)


# ---------------------------------------------------------------------------
# Dataset -- identical contract to teacher_student_data_loader.py's
# TeacherStudentFLDDataset, reading THIS module's BIOMECH_COLUMNS.
# ---------------------------------------------------------------------------

class TeacherStudentGRFFLDDataset(Dataset):
    """One trial's worth of time-aligned sensor/biomech windows. Same shapes/
    keys as teacher_student_data_loader.py's TeacherStudentFLDDataset --
    'biomech_window'/'future_biomech' now hold (angle+velocity+GRF/COP)
    instead of (angle+velocity+moment). See that class's own docstring for
    the full field-by-field contract (unchanged here)."""

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
        H, K = self.history_horizon, self.forecast_horizon
        context_start, context_end = idx, idx + H
        future_start = idx + H
        assert_causal_window(context_start, context_end, future_start)

        sensor_window = self.data_sensor[context_start:context_end].T
        biomech_window = self.data_biomech[context_start:context_end].T
        future_biomech = self.data_biomech[future_start:future_start + K]

        return {
            'sensor_window': sensor_window,
            'biomech_window': biomech_window,
            'future_biomech': future_biomech,
            'subject_id': self.subject_id,
            'trial_id': self.trial_id,
        }


# ---------------------------------------------------------------------------
# Loading -- mirrors teacher_student_data_loader.py's load_all_data()/
# build_datasets(), GRF_FORCE_COLUMNS weight-normalized instead of MOMENT_COLUMNS.
# ---------------------------------------------------------------------------

def load_all_data(data_path: str = ALL_DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found -- run combine_data.py first.")
    usecols = list(dict.fromkeys(
        ['trial_id', 'subject_name', 'subject_weight'] + SENSOR_COLUMNS + BIOMECH_COLUMNS
    ))
    df = pd.read_csv(data_path, usecols=usecols)
    df[GRF_FORCE_COLUMNS] = df[GRF_FORCE_COLUMNS].div(df['subject_weight'], axis=0)
    return df


def build_datasets(
    history_horizon: int,
    forecast_horizon: int,
    subjects: Optional[List[str]] = None,
    data_path: str = ALL_DATA_PATH,
    index_path: str = TRAIN_VAL_INDEX_PATH,
    device: Optional[torch.device] = None,
) -> Tuple[ConcatDataset, ConcatDataset, Dict[str, np.ndarray], Dict[str, TeacherStudentGRFFLDDataset]]:
    """Same contract as teacher_student_data_loader.py's build_datasets() --
    see its docstring for the full train/val-by-window-within-trial /
    norm-stats-fit-on-train-only explanation (identical here, only the
    biomech column list differs)."""
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
    norm_stats = compute_norm_stats_from_arrays(
        np.concatenate(train_raw_sensor, axis=0), np.concatenate(train_raw_biomech, axis=0),
    )

    trial_datasets: Dict[str, TeacherStudentGRFFLDDataset] = {}
    train_subsets, val_subsets = [], []
    for trial_id, trial_df in trial_dfs.items():
        if trial_id not in split_index:
            print(f"[teacher_student_grf_data_loader] {trial_id}: not in {index_path}, skipping.")
            continue
        train_idx = split_index[trial_id]['train']
        val_idx = split_index[trial_id]['val']
        if not train_idx or not val_idx:
            print(f"[teacher_student_grf_data_loader] {trial_id}: missing train or val windows "
                  f"(train={len(train_idx)}, val={len(val_idx)}), skipping.")
            continue
        subject_id = str(trial_df['subject_name'].iloc[0])
        dataset = TeacherStudentGRFFLDDataset(
            trial_df, subject_id, trial_id, history_horizon, forecast_horizon, norm_stats, device=device,
        )
        trial_datasets[trial_id] = dataset
        train_subsets.append(Subset(dataset, train_idx))
        val_subsets.append(Subset(dataset, val_idx))

    if not train_subsets:
        raise ValueError("No usable train/val windows -- check subjects/index_path/history_horizon.")

    print(f"[teacher_student_grf_data_loader] train: {sum(len(s) for s in train_subsets):,} windows "
          f"across {len(train_subsets)} trials")
    print(f"[teacher_student_grf_data_loader] val:   {sum(len(s) for s in val_subsets):,} windows "
          f"across {len(val_subsets)} trials")

    return ConcatDataset(train_subsets), ConcatDataset(val_subsets), norm_stats, trial_datasets
