"""
teacher_student_data_loader.py

Data loader for FLD_teacher_student.py. Does NOT modify DataLoader/data_loader.py
-- runs side by side with it. Reads Data/all_data*.csv (the same files
combine_data.py builds and every other training script in this repo reads --
see train_fld.py's ALL_DATA_PATH), NOT a new per-subject .npz format: this
repo's real data already lives there.

Channels (see module-level constants below):
  SENSOR_COLUMNS  (36): full body IMU (5 locations x accel+gyro = 30) + insole
                        force + COP only (6) -- NOT insole accel/gyro (those
                        exist in train_fld.py's 48-dim INPUT_COLUMNS but are
                        deliberately excluded here).
  BIOMECH_COLUMNS (30): imported from train_fld.OUTPUT_COLUMNS verbatim (10
                        joint angles + 10 angular velocities + 10 moments) --
                        no need to redeclare, it's already exactly this set.

Tensor shape convention -- matches DataLoader/data_loader.py's FLDDataset and
Models/FLD_teacher_student.py exactly:
  sensor_window   : (sensor_dim, H)   -- channels-first
  biomech_window  : (biomech_dim, H)  -- channels-first, TIME-ALIGNED with
                     sensor_window (same [idx, idx+H) rows)
  future_biomech  : (K, biomech_dim)  -- time-first

Train/val splitting is BY WINDOW WITHIN EACH TRIAL, not by subject -- see
build_datasets(), which reads a train_val_index CSV (built by
build_train_val_index.py, e.g. Data/train_val_index_2_Subjects.csv) exactly
the way train_fld.py/train_tcnn.py do: every trial contributes windows to
BOTH train and val (~85/15 split, val spread across a few segments rather
than one chronological tail), and normalisation stats are fit on train
window rows only (val rows excluded via the same masking train_fld.py uses
-- imported, not reimplemented). `subjects` in build_datasets() only
restricts which subjects' trials participate at all (None/[] = every
subject in the file) -- it is NOT a train/val/test assignment.

Every window-producing Dataset is still built ONE PER TRIAL (mirroring
train_fld.py's own per-trial FLDDataset + ConcatDataset(Subset(...)) pattern)
so no window ever crosses a trial boundary, exactly like the rest of this repo.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from train_fld import (
    ALL_DATA_PATH, OUTPUT_COLUMNS as BIOMECH_COLUMNS, MOMENT_COLUMNS, DATA_DIR,
    INDEX_PATH as TRAIN_VAL_INDEX_PATH, _norm_stats_mask,
)

# A smaller, faster dev/test data file -- same combine_data.py pipeline,
# SUBJECTS restricted to just S01/S02 (see combine_data.py) -- and its
# matching train_val_index CSV (built by build_train_val_index.py against
# that same file).
ALL_DATA_2_SUBJECTS_PATH = os.path.join(DATA_DIR, "all_data_2_subjects.csv")
TRAIN_VAL_INDEX_2_SUBJECTS_PATH = os.path.join(DATA_DIR, "train_val_index_2_Subjects.csv")

# Same idea, just S01 -- smallest/fastest dev file + its matching index.
# train_fld_teacher_student.py's TeacherStudentConfig defaults to this pair.
ALL_DATA_1_SUBJECT_PATH = os.path.join(DATA_DIR, "all_data_1_subject.csv")
TRAIN_VAL_INDEX_1_SUBJECT_PATH = os.path.join(DATA_DIR, "train_val_index_1_Subject.csv")

# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------

BODY_IMU_COLUMNS = [
    'pelvis_acc_x', 'pelvis_acc_y', 'pelvis_acc_z',
    'thigh_r_acc_x', 'thigh_r_acc_y', 'thigh_r_acc_z',
    'shank_r_acc_x', 'shank_r_acc_y', 'shank_r_acc_z',
    'thigh_l_acc_x', 'thigh_l_acc_y', 'thigh_l_acc_z',
    'shank_l_acc_x', 'shank_l_acc_y', 'shank_l_acc_z',
    'pelvis_gyro_x', 'pelvis_gyro_y', 'pelvis_gyro_z',
    'thigh_r_gyro_x', 'thigh_r_gyro_y', 'thigh_r_gyro_z',
    'shank_r_gyro_x', 'shank_r_gyro_y', 'shank_r_gyro_z',
    'thigh_l_gyro_x', 'thigh_l_gyro_y', 'thigh_l_gyro_z',
    'shank_l_gyro_x', 'shank_l_gyro_y', 'shank_l_gyro_z',
]  # 30

INSOLE_FORCE_COP_COLUMNS = [
    'L_insole_force', 'L_insole_COPx', 'L_insole_COPz',
    'R_insole_force', 'R_insole_COPx', 'R_insole_COPz',
]  # 6 -- deliberately excludes insole accel/gyro

SENSOR_COLUMNS = BODY_IMU_COLUMNS + INSOLE_FORCE_COP_COLUMNS  # 36

SENSOR_DIM = len(SENSOR_COLUMNS)
BIOMECH_DIM = len(BIOMECH_COLUMNS)


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------

def assert_causal_window(context_start: int, context_end: int, future_start: int) -> None:
    """A window is causal iff its context never reads a row at or after
    `future_start`: context rows are [context_start, context_end), future
    rows begin at future_start. Raises ValueError (not a bare `assert`, so
    it fires even when Python is run with `-O`) if context_end > future_start
    -- i.e. if the context would contain samples at/after the point the
    model is being asked to predict past.

    Args:
        context_start: first row index of the context window (inclusive).
        context_end: one past the last row index of the context window
            (exclusive) -- i.e. context reads rows [context_start, context_end).
        future_start: first row index the future/target window reads.

    Raises:
        ValueError: if context_end > future_start.
    """
    if context_end > future_start:
        raise ValueError(
            f"Non-causal window: context reads rows [{context_start}, {context_end}) but the "
            f"future/target window starts at row {future_start} (context_end must be "
            f"<= future_start) -- context would contain samples at/after the prediction "
            f"target, violating strict causality."
        )


# ---------------------------------------------------------------------------
# Normalisation stats
# ---------------------------------------------------------------------------

def compute_norm_stats_from_arrays(sensor: np.ndarray, biomech: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-channel mean/std, computed directly from raw (rows, SENSOR_DIM)/
    (rows, BIOMECH_DIM) arrays -- the array-level building block both
    compute_norm_stats() (whole-dataframe convenience, below) and
    build_datasets() (mask-aware, val-window rows excluded) call.

    Returns:
        dict with keys 'mean_sensor', 'std_sensor', 'mean_biomech',
        'std_biomech' (each a float32 np.ndarray). std is clamped to 1.0
        wherever it's exactly 0, same convention as train_fld.py, to avoid a
        division-by-zero on a constant channel.
    """
    mean_sensor, std_sensor = sensor.mean(axis=0), sensor.std(axis=0)
    mean_biomech, std_biomech = biomech.mean(axis=0), biomech.std(axis=0)
    std_sensor[std_sensor == 0] = 1.0
    std_biomech[std_biomech == 0] = 1.0
    return {
        'mean_sensor': mean_sensor.astype('float32'),
        'std_sensor': std_sensor.astype('float32'),
        'mean_biomech': mean_biomech.astype('float32'),
        'std_biomech': std_biomech.astype('float32'),
    }


def compute_norm_stats(train_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Per-channel mean/std for SENSOR_COLUMNS/BIOMECH_COLUMNS, pooling ALL of
    `train_df`'s rows (no val-row exclusion) -- a simple whole-dataframe
    convenience wrapper around compute_norm_stats_from_arrays(). build_datasets()
    below does NOT call this -- it fits its own masked variant (excluding
    rows any val window reads) directly."""
    sensor = train_df[SENSOR_COLUMNS].values.astype('float64')
    biomech = train_df[BIOMECH_COLUMNS].values.astype('float64')
    return compute_norm_stats_from_arrays(sensor, biomech)


# ---------------------------------------------------------------------------
# Dataset -- one instance per trial (mirrors DataLoader/data_loader.py's
# FLDDataset, which train_fld.py also instantiates one-per-trial and
# ConcatDataset's together)
# ---------------------------------------------------------------------------

class TeacherStudentFLDDataset(Dataset):
    """One trial's worth of time-aligned sensor/biomech windows.

    Each sample is a dict (not a tuple):
      'sensor_window'  : (SENSOR_DIM, H)   -- channels-first, rows [idx, idx+H)
      'biomech_window' : (BIOMECH_DIM, H)  -- channels-first, SAME rows [idx, idx+H)
                          (time-aligned with sensor_window -- this is the
                          teacher branch's own input)
      'future_biomech'  : (K, BIOMECH_DIM)  -- time-first, rows [idx+H, idx+H+K),
                          the genuinely-future ground truth for the
                          propagation loss
      'subject_id'      : str
      'trial_id'        : str

    Windows never cross a trial boundary because this Dataset is built from
    exactly one trial's own contiguous rows (see build_datasets() below) --
    same structural guarantee build_train_val_index.py/train_fld.py rely on.
    """

    def __init__(
        self,
        trial_df: pd.DataFrame,
        subject_id: str,
        trial_id: str,
        history_horizon: int,
        forecast_horizon: int,
        norm_stats: Dict[str, np.ndarray],
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            trial_df: one trial's rows, in time order, already moment-normalised
                (see load_all_data()). Must contain every column in
                SENSOR_COLUMNS and BIOMECH_COLUMNS.
            subject_id: this trial's subject_name value.
            trial_id: this trial's trial_id value.
            history_horizon: H, context window length. Must be odd (same
                requirement as the underlying FLD instances).
            forecast_horizon: K, number of genuinely-future rows per sample.
            norm_stats: dict from compute_norm_stats() (or loaded from a
                checkpoint) -- ALWAYS the TRAIN split's stats, even when this
                instance is built for the val/test split (never re-fit on
                val/test rows).
            device: optional torch device. When given, this trial's (small)
                normalised tensors are moved there at construction time, same
                convention as FLDDataset (avoids per-batch host->device
                copies; forces num_workers=0 in the caller's DataLoader).
        """
        assert history_horizon % 2 == 1, (
            f"history_horizon must be odd (got {history_horizon}) -- FLD's Conv1d layers "
            f"need symmetric same-size padding."
        )
        self.subject_id = subject_id
        self.trial_id = trial_id
        self.history_horizon = history_horizon
        self.forecast_horizon = forecast_horizon
        self.window_size = history_horizon + forecast_horizon
        self.norm_stats = norm_stats   # stored on the dataset object, per spec

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
        # Count of valid window starts: idx in [0, N - window_size] inclusive,
        # i.e. N - window_size + 1 values. (Deliberately +1 vs. the original
        # FLDDataset's `len(data_in) - window_size` -- this is a NEW dataset,
        # not a modification of the existing one, so it's free to not carry
        # forward that off-by-one, which drops the trial's very last valid
        # window.)
        return max(0, len(self.data_sensor) - self.window_size + 1)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        H, K = self.history_horizon, self.forecast_horizon
        context_start, context_end = idx, idx + H
        future_start = idx + H
        assert_causal_window(context_start, context_end, future_start)

        sensor_window = self.data_sensor[context_start:context_end].T     # (SENSOR_DIM, H)
        biomech_window = self.data_biomech[context_start:context_end].T   # (BIOMECH_DIM, H)
        future_biomech = self.data_biomech[future_start:future_start + K]  # (K, BIOMECH_DIM)

        return {
            'sensor_window': sensor_window,
            'biomech_window': biomech_window,
            'future_biomech': future_biomech,
            'subject_id': self.subject_id,
            'trial_id': self.trial_id,
        }


# ---------------------------------------------------------------------------
# Loading + window-level train/val splitting (NOT by subject -- see module
# docstring)
# ---------------------------------------------------------------------------

def load_all_data(data_path: str = ALL_DATA_2_SUBJECTS_PATH) -> pd.DataFrame:
    """Read an all_data*.csv file, keeping only the columns this loader
    needs, and normalise joint moments to N.m/kg by dividing by each row's
    own subject_weight -- same convention as train_fld.py's _load_all_data
    (subject_weight itself is read only for this division; it is NOT part of
    SENSOR_COLUMNS, matching the architecture spec's "no subject
    conditioning" input channel counts)."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found -- run combine_data.py first.")
    usecols = list(dict.fromkeys(
        ['trial_id', 'subject_name', 'subject_weight'] + SENSOR_COLUMNS + BIOMECH_COLUMNS
    ))
    df = pd.read_csv(data_path, usecols=usecols)
    df[MOMENT_COLUMNS] = df[MOMENT_COLUMNS].div(df['subject_weight'], axis=0)
    return df


def _load_split_index(index_path: str, window_size: int) -> Dict[str, Dict[str, List[int]]]:
    """trial_id -> {'train': [local_start_idx...], 'val': [local_start_idx...]},
    from a train_val_index CSV built by build_train_val_index.py -- mirrors
    train_fld.py's own _load_split_index exactly (same columns, same
    window_size compatibility check), just parameterized by index_path
    since this loader supports more than one data/index file pair
    (Data/all_data.csv <-> Data/train_val_index.csv, and
    Data/all_data_2_subjects.csv <-> Data/train_val_index_2_Subjects.csv)
    rather than train_fld.py's one hardcoded pair."""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"{index_path} not found -- run build_train_val_index.py first.")
    idx_df = pd.read_csv(index_path, usecols=['trial_id', 'split', 'local_start_idx', 'window_size'])
    index_window_size = int(idx_df['window_size'].min())
    if index_window_size < window_size:
        raise ValueError(
            f"{index_path} was built for window_size={index_window_size}, smaller than this "
            f"config's window_size={window_size} -- rerun build_train_val_index.py with "
            f"HISTORY_HORIZON/FORECAST_HORIZON matching (or exceeding) this config first."
        )
    splits: Dict[str, Dict[str, List[int]]] = {}
    for trial_id, g in idx_df.groupby('trial_id', sort=False):
        splits[str(trial_id)] = {
            'train': g.loc[g['split'] == 'train', 'local_start_idx'].tolist(),
            'val':   g.loc[g['split'] == 'val',   'local_start_idx'].tolist(),
        }
    return splits


def build_datasets(
    history_horizon: int,
    forecast_horizon: int,
    subjects: Optional[List[str]] = None,
    data_path: str = ALL_DATA_2_SUBJECTS_PATH,
    index_path: str = TRAIN_VAL_INDEX_2_SUBJECTS_PATH,
    device: Optional[torch.device] = None,
) -> Tuple[ConcatDataset, ConcatDataset, Dict[str, np.ndarray], Dict[str, TeacherStudentFLDDataset]]:
    """End-to-end: load `data_path`, optionally restrict to `subjects`
    (None/[] = every subject present), then split TRAIN/VAL BY WINDOW WITHIN
    EACH TRIAL using `index_path` -- exactly train_fld.py's/train_tcnn.py's
    own convention (_load_split_index -> Subset(dataset, train_idx) /
    Subset(dataset, val_idx)), NOT a held-out subject. Every trial
    contributes windows to both train and val.

    Normalisation stats are fit on TRAIN WINDOW rows only, using the same
    row-exclusion mask train_fld.py's own _norm_stats_mask computes (rows
    any val window reads -- context or future -- are excluded, so val never
    leaks into the fitted train stats). _norm_stats_mask/window_size
    compatibility check are imported from train_fld.py, not reimplemented.

    Args:
        subjects: restrict to these subject_name values, or None/[] for
            every subject present in `data_path`. This ONLY filters which
            trials participate at all -- it is not a train/val assignment.

    Returns:
        (train_dataset, val_dataset, norm_stats, trial_datasets) --
        trial_datasets is {trial_id: TeacherStudentFLDDataset}, the raw
        (un-Subset-wrapped) per-trial datasets. train_dataset/val_dataset are
        ConcatDataset(Subset(...)) and don't expose a single trial's own
        data_sensor/data_biomech/norm_stats directly -- diagnostics/plotting
        that need that should index into trial_datasets instead.
    """
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

    # Norm stats: pool every trial's TRAIN-usable rows (val rows excluded via
    # _norm_stats_mask), matching train_fld.py's own approach exactly.
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

    trial_datasets: Dict[str, TeacherStudentFLDDataset] = {}
    train_subsets, val_subsets = [], []
    for trial_id, trial_df in trial_dfs.items():
        if trial_id not in split_index:
            print(f"[teacher_student_data_loader] {trial_id}: not in {index_path}, skipping.")
            continue
        train_idx = split_index[trial_id]['train']
        val_idx = split_index[trial_id]['val']
        if not train_idx or not val_idx:
            print(f"[teacher_student_data_loader] {trial_id}: missing train or val windows "
                  f"(train={len(train_idx)}, val={len(val_idx)}), skipping.")
            continue
        subject_id = str(trial_df['subject_name'].iloc[0])
        dataset = TeacherStudentFLDDataset(
            trial_df, subject_id, trial_id, history_horizon, forecast_horizon, norm_stats, device=device,
        )
        trial_datasets[trial_id] = dataset
        train_subsets.append(Subset(dataset, train_idx))
        val_subsets.append(Subset(dataset, val_idx))

    if not train_subsets:
        raise ValueError("No usable train/val windows -- check subjects/index_path/history_horizon.")

    print(f"[teacher_student_data_loader] train: {sum(len(s) for s in train_subsets):,} windows "
          f"across {len(train_subsets)} trials")
    print(f"[teacher_student_data_loader] val:   {sum(len(s) for s in val_subsets):,} windows "
          f"across {len(val_subsets)} trials")

    return ConcatDataset(train_subsets), ConcatDataset(val_subsets), norm_stats, trial_datasets


# ---------------------------------------------------------------------------
# Leave-subject-out variant -- NO train_val_index CSV. Train on every window of
# one file, validate on every window of a DIFFERENT file (a held-out subject).
# Additive: build_datasets() above is untouched and still serves every caller
# that uses the index-based within-trial split.
# ---------------------------------------------------------------------------

def _build_trial_datasets(
    df: pd.DataFrame, history_horizon: int, forecast_horizon: int,
    norm_stats: Dict[str, np.ndarray], device: Optional[torch.device],
) -> Dict[str, TeacherStudentFLDDataset]:
    """{trial_id: TeacherStudentFLDDataset}, one per trial in `df` (file order),
    skipping trials too short for even one window."""
    datasets: Dict[str, TeacherStudentFLDDataset] = {}
    for trial_id, g in df.groupby('trial_id', sort=False):
        g = g.reset_index(drop=True)
        dataset = TeacherStudentFLDDataset(
            g, str(g['subject_name'].iloc[0]), str(trial_id),
            history_horizon, forecast_horizon, norm_stats, device=device,
        )
        if len(dataset) == 0:
            print(f"[teacher_student_data_loader] {trial_id}: {len(g)} rows < window_size="
                  f"{dataset.window_size}, skipping.")
            continue
        datasets[str(trial_id)] = dataset
    return datasets


def build_datasets_held_out(
    history_horizon: int,
    forecast_horizon: int,
    train_data_path: str,
    val_data_path: str,
    train_subjects: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[ConcatDataset, ConcatDataset, Dict[str, np.ndarray],
           Dict[str, TeacherStudentFLDDataset], Dict[str, TeacherStudentFLDDataset]]:
    """Leave-subject-out datasets: TRAIN = every window (stride 1) of every
    trial in `train_data_path`; VAL = every window of every trial in
    `val_data_path`, a file holding a subject the model never trains on. No
    train_val_index CSV is read.

    Normalisation stats are fit on ALL rows of the train file and reused
    unchanged for val -- val rows never touch them. The two files must hold
    disjoint subjects (ValueError otherwise), since a shared subject would
    silently defeat the point of a leave-subject-out check.

    Args:
        train_subjects: restrict the TRAIN file to these subject_name values
            (None/[] = every subject in it). Never applied to the val file.

    Returns:
        (train_dataset, val_dataset, norm_stats, train_trial_datasets,
        val_trial_datasets) -- the two trial dicts are the raw per-trial
        datasets ({trial_id: TeacherStudentFLDDataset}), for plotting that
        needs a single trial's data_sensor/data_biomech (see build_datasets()).
    """
    train_df = load_all_data(train_data_path)
    if train_subjects:
        available = set(train_df['subject_name'].unique())
        missing = [s for s in train_subjects if s not in available]
        if missing:
            raise ValueError(f"subjects {missing} not found in {train_data_path}. "
                              f"Available: {sorted(available)}")
        train_df = train_df.loc[train_df['subject_name'].isin(train_subjects)].reset_index(drop=True)
    val_df = load_all_data(val_data_path)

    overlap = set(train_df['subject_name'].unique()) & set(val_df['subject_name'].unique())
    if overlap:
        raise ValueError(f"Subjects {sorted(overlap)} appear in BOTH {train_data_path} and "
                          f"{val_data_path} -- not a leave-subject-out split.")

    norm_stats = compute_norm_stats(train_df)   # train rows only, all of them

    train_trials = _build_trial_datasets(train_df, history_horizon, forecast_horizon, norm_stats, device)
    val_trials = _build_trial_datasets(val_df, history_horizon, forecast_horizon, norm_stats, device)
    if not train_trials or not val_trials:
        raise ValueError(f"No usable windows (train trials={len(train_trials)}, val trials={len(val_trials)}) "
                          f"-- check the data paths/history_horizon/forecast_horizon.")

    train_ds = ConcatDataset(list(train_trials.values()))
    val_ds = ConcatDataset(list(val_trials.values()))
    print(f"[teacher_student_data_loader] train: {len(train_ds):,} windows across {len(train_trials)} trials "
          f"(subjects {sorted(train_df['subject_name'].unique())})")
    print(f"[teacher_student_data_loader] val:   {len(val_ds):,} windows across {len(val_trials)} trials "
          f"(held-out subjects {sorted(val_df['subject_name'].unique())})")
    return train_ds, val_ds, norm_stats, train_trials, val_trials
