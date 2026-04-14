import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def to_device(x: torch.Tensor) -> torch.Tensor:
    """Move a tensor to GPU if available, otherwise keep on CPU."""
    return x.cuda() if torch.cuda.is_available() else x


def item(value: torch.Tensor) -> torch.Tensor:
    """Detach a tensor from the computation graph and move to CPU."""
    return value.detach().cpu()


def get_device() -> torch.device:
    """Return the active torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# DataFrame / data inspection
# ---------------------------------------------------------------------------

def print_column_names(df: pd.DataFrame) -> None:
    """Print all column names with their index, dtype, and null count."""
    print(f"\nDataFrame shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
    print(f"{'Index':<6}  {'Column':<45}  {'Dtype':<12}  {'Nulls'}")
    print("-" * 80)
    for i, col in enumerate(df.columns):
        nulls = df[col].isna().sum()
        print(f"{i:<6}  {col:<45}  {str(df[col].dtype):<12}  {nulls}")


def summarise_df(df: pd.DataFrame) -> None:
    """Print a brief statistical summary of a DataFrame."""
    print(df.describe().T.to_string())


def filter_columns_by_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Return a DataFrame containing only columns that start with *prefix*."""
    cols = [c for c in df.columns if c.startswith(prefix)]
    return df[cols]


# ---------------------------------------------------------------------------
# Train / val splitting
# ---------------------------------------------------------------------------

def split_by_group(
    df: pd.DataFrame,
    group_col: str,
    train_frac: float = 0.8,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train/val by unique values of *group_col*,
    so entire groups stay together (no leakage across windows).

    Returns:
        train_df, val_df
    """
    rng = np.random.default_rng(seed)
    groups = sorted(df[group_col].unique())
    rng.shuffle(groups)
    n_train = max(1, int(train_frac * len(groups)))
    train_groups = set(groups[:n_train])

    train_df = df[df[group_col].isin(train_groups)].copy()
    val_df   = df[~df[group_col].isin(train_groups)].copy()
    return train_df, val_df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_all_columns(df: pd.DataFrame, max_cols: Optional[int] = None) -> None:
    """
    Plot each column of a DataFrame in its own subplot stacked vertically.

    Args:
        df:       DataFrame to plot.
        max_cols: Optional cap on the number of columns to plot.
    """
    cols = list(df.columns) if max_cols is None else list(df.columns[:max_cols])
    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        ax.plot(df.index, df[col])
        ax.set_ylabel(col, fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
    axes[-1].set_xlabel("Index")
    plt.tight_layout()
    plt.show()


def plot_train_val_loss(
    train_losses: List[float],
    val_losses: List[float],
    title: str = "Training vs Validation Loss",
) -> None:
    """Plot training and validation loss curves."""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_losses,   label="Val Loss",   linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_predictions(
    pred: np.ndarray,
    ground_truth: np.ndarray,
    col_names: List[str],
    features_per_fig: int = 10,
) -> None:
    """
    Plot predicted vs ground-truth time series for each output feature.

    Args:
        pred:           Array of shape (N, F).
        ground_truth:   Array of shape (N, F).
        col_names:      Feature names (length F).
        features_per_fig: How many subplots to pack per figure.
    """
    N, F = pred.shape
    for group_start in range(0, F, features_per_fig):
        group_end = min(group_start + features_per_fig, F)
        n_plots = group_end - group_start
        rows, cols = 2, (n_plots + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(15, 8), sharex=True)
        axes = axes.flatten()

        for i, feat_idx in enumerate(range(group_start, group_end)):
            axes[i].plot(pred[:, feat_idx],         label="Prediction",   color="red",   linewidth=1.2)
            axes[i].plot(ground_truth[:, feat_idx],  label="Ground Truth", color="black", linewidth=1.0, alpha=0.7)
            axes[i].set_title(col_names[feat_idx], fontsize=9)
            if i == 0:
                axes[i].legend(fontsize=8)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.xlabel("Time (samples)")
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    col_names: List[str],
) -> Dict[str, np.ndarray]:
    """
    Compute per-feature MAE, RMSE, and std of absolute error.

    Args:
        pred:       (N, F) predictions (un-normalised).
        target:     (N, F) ground truth (un-normalised).
        col_names:  Feature names for printing.

    Returns:
        dict with keys 'mae', 'rmse', 'std'.
    """
    err      = pred - target
    abs_err  = np.abs(err)
    mae      = abs_err.mean(axis=0)
    rmse     = np.sqrt((err ** 2).mean(axis=0))
    std      = abs_err.std(axis=0)

    for i, name in enumerate(col_names):
        print(f"{name:<40}  MAE={mae[i]:.4f}  RMSE={rmse[i]:.4f}  STD={std[i]:.4f}")

    return {"mae": mae, "rmse": rmse, "std": std}
