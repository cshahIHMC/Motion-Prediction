import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import List, Optional, Tuple, Dict, Iterable
from collections import defaultdict
import random


def load_csv(filepath: str) -> pd.DataFrame:
    """Load the dataset CSV into a pandas DataFrame."""
    df = pd.read_csv(filepath)
    return df


class SequenceDataset(Dataset):
    """
    Sliding-window sequence dataset for the Motion-Prediction project.

    Produces windows of shape:
        X: (input_features, seq_len)   — model input
        Y: (output_features, pred_len) — prediction target

    Windows never cross trial boundaries (group_col rows are kept contiguous).

    Args:
        df:           Full DataFrame loaded from CSV.
        seq_len:      Length of the look-back window (input).
        pred_len:     Number of future steps to predict.
        input_cols:   List of column names used as model input.
        output_cols:  List of column names used as prediction targets.
        group_col:    Column that identifies independent trials/groups.
                      Set to None if the whole file is one contiguous trial.
        stride:       Step size between consecutive windows.
        dtype:        Torch dtype for output tensors.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int,
        pred_len: int,
        input_cols: List[str],
        output_cols: List[str],
        group_col: Optional[str] = None,
        stride: int = 1,
        dtype: torch.dtype = torch.float32,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = seq_len + pred_len
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.stride = stride
        self.dtype = dtype

        df = df.copy()

        # --- global normalisation ---
        X_all = df[input_cols].to_numpy(dtype=np.float64)
        Y_all = df[output_cols].to_numpy(dtype=np.float64)

        self.input_mean = X_all.mean(axis=0)
        self.input_std  = X_all.std(axis=0)
        self.input_std[self.input_std == 0] = 1.0

        self.output_mean = Y_all.mean(axis=0)
        self.output_std  = Y_all.std(axis=0)
        self.output_std[self.output_std == 0] = 1.0

        # --- build per-group arrays ---
        self.groups: Dict[object, dict] = {}

        if group_col is None:
            df["_group"] = 0
            group_col = "_group"

        for key, g in df.groupby(group_col, sort=False):
            g = g.sort_index().reset_index(drop=True)

            X = g[input_cols].to_numpy(dtype=np.float64)
            Y = g[output_cols].to_numpy(dtype=np.float64)

            X = ((X - self.input_mean) / self.input_std).astype(np.float32)
            Y = ((Y - self.output_mean) / self.output_std).astype(np.float32)

            self.groups[key] = {"X": X, "Y": Y, "length": X.shape[0]}

        # --- window index ---
        self.index: List[Tuple[object, int]] = []
        for key, grp in self.groups.items():
            n = grp["length"]
            if n >= self.total_len:
                for s in range(0, n - self.total_len + 1, self.stride):
                    self.index.append((key, s))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        key, s = self.index[i]
        grp = self.groups[key]

        X = torch.tensor(grp["X"][s : s + self.seq_len].T, dtype=self.dtype)       # (F_in, seq_len)
        Y = torch.tensor(grp["Y"][s + self.seq_len : s + self.total_len].T,         # (F_out, pred_len)
                         dtype=self.dtype)
        return X, Y


class GroupedBatchSampler(Sampler[List[int]]):
    """
    Keeps every batch within a single group so sequences stay contiguous.
    Useful when the model is sensitive to within-group temporal structure.
    """

    def __init__(
        self,
        dataset: SequenceDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        buckets: Dict[object, List[int]] = defaultdict(list)
        for i, (key, _) in enumerate(dataset.index):
            buckets[key].append(i)
        self.buckets = dict(buckets)
        self.group_keys = list(self.buckets.keys())

    def __iter__(self):
        rng = random.Random(self.seed)
        keys = self.group_keys[:]
        if self.shuffle:
            rng.shuffle(keys)

        for key in keys:
            idxs = self.buckets[key][:]
            if self.shuffle:
                rng.shuffle(idxs)
            for start in range(0, len(idxs), self.batch_size):
                batch = idxs[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        total = 0
        for idxs in self.buckets.values():
            n = len(idxs) // self.batch_size
            if not self.drop_last and (len(idxs) % self.batch_size):
                n += 1
            total += n
        return total


def build_dataloader(
    df: pd.DataFrame,
    seq_len: int,
    pred_len: int,
    input_cols: List[str],
    output_cols: List[str],
    batch_size: int = 32,
    group_col: Optional[str] = None,
    stride: int = 1,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """
    Convenience wrapper: create a SequenceDataset and wrap it in a DataLoader
    with the GroupedBatchSampler so batches stay within trial boundaries.
    """
    dataset = SequenceDataset(
        df=df,
        seq_len=seq_len,
        pred_len=pred_len,
        input_cols=input_cols,
        output_cols=output_cols,
        group_col=group_col,
        stride=stride,
    )
    sampler = GroupedBatchSampler(dataset, batch_size=batch_size,
                                  shuffle=shuffle, drop_last=drop_last)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers)
