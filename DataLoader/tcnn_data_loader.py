import torch
from torch.utils.data import Dataset


class TCNNDataset(Dataset):
    """Context/future-split dataset for TCNN training.

    Kept independent from DataLoader/data_loader.py's FLDDataset (no shared
    code) so that changes to one training script's data pipeline can never
    break the other's.

    Each sample is (context, future):
      context: (input_dim, history_horizon)   — rows [idx, idx+H), channels-first,
               ready to feed a Conv1d-based model directly.
      future:  (forecast_horizon, output_dim) — rows [idx+H, idx+H+K), the
               genuinely future ground truth immediately following the context
               window.
    The two never overlap: window_size = history_horizon + forecast_horizon
    rows are consumed per sample. Unlike FLD, the TCN's causal dilated
    convolutions have no odd-length requirement, so history_horizon may be
    any positive integer.

    Args:
        df: DataFrame built as [input columns] followed by [output columns].
        history_horizon: Length of the input context window.
        forecast_horizon: Number of future steps to predict.
        norm_stats: optional dict with keys 'mean_in', 'std_in', 'mean_out',
                    'std_out' (all np.ndarray float32). When provided the
                    dataset is normalised with these pre-computed stats
                    instead of computing them from df — use this so every
                    task shares one global normalisation.
        device: optional torch device. When given, the (small) normalised
                tensors are moved there at construction time so
                __getitem__ slices directly out of device memory.
        input_dim: number of leading columns in `df` that make up the
                   encoder input (everything after this is treated as
                   output).
    """

    def __init__(self, df, history_horizon: int = 150, forecast_horizon: int = 50,
                 norm_stats: dict = None, device=None, input_dim: int = 50):
        self.history_horizon = history_horizon
        self.forecast_horizon = forecast_horizon
        self.window_size = history_horizon + forecast_horizon

        raw_in  = df.iloc[:, :input_dim].values.astype('float32')
        raw_out = df.iloc[:, input_dim:].values.astype('float32')

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
