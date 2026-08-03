import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import List, Optional, Tuple, Dict, Iterable
from collections import defaultdict
import random

class dataLoader_seq(Dataset):
    def __init__(self, df, seq_length, predictor_seq_length=None):
        
        
        # Save all the data
        self.original_df = df.copy()
        self.seq_length = seq_length
        self.predictor_seq_length = predictor_seq_length
        
        # Store the indices
        self.indices = df.index.tolist()
        
        # # Normalize the data (column-wise)
        self.input_mean = self.original_df.iloc[:, :48].mean()
        self.input_std = self.original_df.iloc[:, :48].std()
        self.input_std[self.input_std == 0] = 1
                
        self.Input_normalized_df = (self.original_df.iloc[:, :48] - self.input_mean) / self.input_std
        
        self.output_mean = self.original_df.iloc[:, 48:76].mean()
        self.output_std = self.original_df.iloc[:, 48:76].std()
        self.output_std[self.output_std == 0] = 1
                
        self.Output_normalized_df = (self.original_df.iloc[:, 48:76] - self.output_mean) / self.output_std
        
    def __len__(self):
        return len(self.indices) - self.seq_length - self.predictor_seq_length 

    
    def __getitem__(self,idx):
        
        ######### Extract Sequences
        input_row_start_idx = self.indices[idx]
        input_row_end_idx = self.indices[idx + self.seq_length]
        
        prediction_row_idx = self.indices[idx + self.seq_length + self.predictor_seq_length]
        
        # Extract all col with that sequence length of data
        input_rows = self.Input_normalized_df.iloc[input_row_start_idx:input_row_end_idx, :].values
        output_rows = self.Output_normalized_df.iloc[prediction_row_idx, :].values
        
        # Inputs ( Transpose it to give cols, sequence length data)
        inputs = torch.tensor(input_rows , dtype=torch.float32).T
        outputs = torch.tensor(output_rows , dtype=torch.float32)
        

        # return PAE_inputs_centered, Predictor_inputs, Predictor_outputs
        return inputs, outputs


class FLDDataset(Dataset):
    """Context/future-split dataset for FLD training.

    Each sample is (context, future):
      context: (obs_dim, history_horizon)   — rows [idx, idx+H), channels-first,
               ready to feed the model directly.
      future:  (forecast_horizon, out_dim)  — rows [idx+H, idx+H+K), the genuinely
               future ground truth immediately following the context window.
    The two never overlap: window_size = history_horizon + forecast_horizon rows
    are consumed per sample (e.g. 151 + 50 = 201), split cleanly at H.

    The FLD model expects history_horizon to be an odd number so that its
    Conv1d layers produce same-length outputs with symmetric padding.
    Use history_horizon=151 to approximate "150 past samples".

    Args:
        df: DataFrame whose columns are the features to model.
        history_horizon: Length of the input context window (must be odd).
        forecast_horizon: Number of future steps the FLD predicts forward.
        feature_set: Which columns to use.
            'input'   – first `input_dim` columns only (self-reconstruction)
            'output'  – columns after `input_dim` only (self-reconstruction)
            'all'     – every column in df (default, same-space reconstruction)
            'cross'   – encoder sees the first `input_dim` columns, decoder
                        reconstructs the remaining columns
    """

    def __init__(self, df, history_horizon: int = 151, forecast_horizon: int = 50,
                 feature_set: str = 'all', norm_stats: dict = None, device=None,
                 input_dim: int = 48):
        """
        norm_stats: optional dict with keys 'mean_in', 'std_in', 'mean_out', 'std_out'
                    (all np.ndarray float32).  When provided the dataset is normalised
                    with these pre-computed stats instead of computing them from df.
                    Use this when combining multiple files so all datasets share the
                    same normalisation.
        device: optional torch device. When given, the (small) normalised tensors are
                moved there at construction time so __getitem__ slices directly out of
                GPU memory — avoids per-batch host→device copies and lets the DataLoader
                run with num_workers=0 (CUDA tensors can't cross process boundaries).
        input_dim: number of leading columns in `df` that make up the encoder input
                   (everything after this is treated as output). Only used by the
                   'input'/'output'/'cross' feature_sets.
        """
        assert history_horizon % 2 == 1, (
            f"history_horizon must be odd for FLD Conv1d compatibility (got {history_horizon})."
        )
        self.history_horizon = history_horizon
        self.forecast_horizon = forecast_horizon
        self.window_size = history_horizon + forecast_horizon
        self.feature_set = feature_set

        if feature_set == 'input':
            raw_in  = df.iloc[:, :input_dim].values.astype('float32')
            raw_out = raw_in
        elif feature_set == 'output':
            raw_in  = df.iloc[:, input_dim:].values.astype('float32')
            raw_out = raw_in
        elif feature_set == 'cross':
            # Caller builds df as [input columns] followed by [output columns].
            raw_in  = df.iloc[:, :input_dim].values.astype('float32')
            raw_out = df.iloc[:, input_dim:].values.astype('float32')
        else:  # 'all'
            raw_in  = df.values.astype('float32')
            raw_out = raw_in

        def _normalise(raw):
            mean = raw.mean(axis=0)
            std  = raw.std(axis=0)
            std[std == 0] = 1.0
            return (raw - mean) / std, mean, std

        if norm_stats is not None:
            mean_in  = norm_stats['mean_in'];  std_in  = norm_stats['std_in']
            mean_out = norm_stats['mean_out']; std_out = norm_stats['std_out']
            norm_in  = (raw_in  - mean_in)  / std_in
            norm_out = (raw_out - mean_out) / std_out
        else:
            norm_in,  mean_in,  std_in  = _normalise(raw_in)
            norm_out, mean_out, std_out = _normalise(raw_out)

        self.input_dim  = raw_in.shape[1]
        self.output_dim = raw_out.shape[1]
        self.obs_dim    = self.input_dim  # kept for backward compat

        self.mean_tensor     = torch.tensor(mean_in,  dtype=torch.float32)
        self.std_tensor      = torch.tensor(std_in,   dtype=torch.float32)
        self.out_mean_tensor = torch.tensor(mean_out, dtype=torch.float32)
        self.out_std_tensor  = torch.tensor(std_out,  dtype=torch.float32)

        self.data_in  = torch.tensor(norm_in,  dtype=torch.float32)
        self.data_out = torch.tensor(norm_out, dtype=torch.float32)

        if device is not None:
            self.data_in  = self.data_in.to(device)
            self.data_out = self.data_out.to(device)
            self.mean_tensor     = self.mean_tensor.to(device)
            self.std_tensor      = self.std_tensor.to(device)
            self.out_mean_tensor = self.out_mean_tensor.to(device)
            self.out_std_tensor  = self.out_std_tensor.to(device)

    def __len__(self):
        return len(self.data_in) - self.window_size

    def __getitem__(self, idx):
        H, K = self.history_horizon, self.forecast_horizon
        context = self.data_in[idx: idx + H].T           # (input_dim, H)
        future  = self.data_out[idx + H: idx + H + K]     # (K, output_dim) — strictly future rows
        return context, future


