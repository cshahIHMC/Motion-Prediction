"""Build train/val window-start index lists over Data/all_data.csv.

Splits 85/15 *within* each trial rather than chronologically (old approach:
last 20% of each trial -> val, which biases val toward end-of-trial behaviour
only). A small number of contiguous val "segments" are placed evenly spread
across each trial (e.g. start-ish / mid / end-ish thirds) instead of one
block at the tail, so validation sees the full range of trial-time.

Two safety properties, both enforced structurally:
  1. No window crosses a trial boundary -- windows are only ever generated
     from a single trial's own [start, end) row range, which is computed
     directly from all_data.csv's trial_id column.
  2. No train/val leakage from near-duplicate overlapping windows -- each
     val segment is shrunk inward by window_size on each side before
     generating window starts, so the closest train and val windows are
     always at least window_size rows apart. Segments too short to survive
     this (mostly very short trials) are dropped adaptively -- see
     _val_segments_for_trial -- rather than silently producing a near-empty
     or leaky val split.

This only writes one row per *window start*, not per raw sample -- a few
hundred thousand rows of a few ints, unlike materializing windowed data.

Two ways to use this module:
  1. As a script (`python3 build_train_val_index.py`) -- builds the default
     dense-window index (HISTORY_HORIZON/FORECAST_HORIZON below, matching
     train_fld.py/train_tcnn.py's window_size) and writes it to
     Data/train_val_index.csv. Run after combine_data.py (or after adding
     new trials) to refresh it.
  2. As a library (`from build_train_val_index import build_index_df`) --
     lets a caller with a DIFFERENT window_size (e.g. train_sparse_fld.py,
     whose effective window is max_lookback + forecast_horizon, not
     history_horizon + forecast_horizon) build its own compatible index at
     runtime instead of reusing a CSV sized for a different window. A
     persisted CSV built for one window_size is only safe to reuse for an
     EQUAL OR SMALLER window_size (see the `window_size` column each row
     carries, and the compatibility check callers should run against it).
"""
import os
import numpy as np
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "all_data.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "train_val_index.csv")

# --- constants used by the CLI entry point (`python3 build_train_val_index.py`) ---
HISTORY_HORIZON = 151       # frames of past context per window (must be odd, matches FLDDataset)
FORECAST_HORIZON = 50       # frames of future prediction horizon per window
VAL_FRACTION = 0.15         # target fraction of windows assigned to validation
NUM_VAL_SEGMENTS = 3        # target number of separate val regions spread across each trial
# ----------------------------------------------------------------------------


def _trial_ranges(meta):
    """meta: DataFrame with columns trial_id, subject_name, task, in file order.

    Returns list of dicts: trial_id, subject_name, task, start (global row,
    inclusive), end (global row, exclusive). Asserts each trial_id occupies
    one contiguous block of rows (true as long as all_data.csv was built by
    combine_data.py, which never interleaves trials).
    """
    trials = []
    trial_id_col = meta["trial_id"].values
    change_points = np.flatnonzero(np.r_[True, trial_id_col[1:] != trial_id_col[:-1]])
    change_points = np.r_[change_points, len(trial_id_col)]
    for i in range(len(change_points) - 1):
        start, end = change_points[i], change_points[i + 1]
        tid = trial_id_col[start]
        if np.any(trial_id_col[end:] == tid):
            raise ValueError(f"trial_id {tid!r} is not contiguous in all_data.csv -- "
                              f"was the file re-sorted or hand-edited after combine_data.py?")
        trials.append({
            "trial_id": tid,
            "subject_name": meta["subject_name"].values[start],
            "task": meta["task"].values[start],
            "start": int(start),
            "end": int(end),
        })
    return trials


def _val_segments_for_trial(trial_length, window_size, val_fraction, num_val_segments):
    """Local (start, end) ranges within [0, trial_length) to use as val,
    BEFORE the window_size inward trim. Picks as many of num_val_segments
    evenly-spaced segments as the trial is long enough to support; drops to
    fewer (down to 0) for short trials so no segment is too small to survive
    trimming.
    """
    min_val_segment_length = 3 * window_size
    num_segments = num_val_segments
    while num_segments > 0 and (val_fraction * trial_length / num_segments) < min_val_segment_length:
        num_segments -= 1
    if num_segments == 0:
        return []

    seg_len = val_fraction * trial_length / num_segments
    segments = []
    for k in range(num_segments):
        center = trial_length * (k + 0.5) / num_segments
        start = int(round(center - seg_len / 2))
        end = int(round(center + seg_len / 2))
        segments.append((max(0, start), min(trial_length, end)))
    return segments


def _windows_for_trial(trial, window_size, val_fraction, num_val_segments):
    """Yield (local_start_idx, split) window starts for one trial.

    local_start_idx is relative to the trial's own first row (0 = trial start).
    """
    trial_length = trial["end"] - trial["start"]
    if trial_length < window_size:
        print(f"[build_index]   {trial['trial_id']}: {trial_length} rows < window_size "
              f"({window_size}) -- skipped entirely")
        return

    val_segments = _val_segments_for_trial(trial_length, window_size, val_fraction, num_val_segments)
    if not val_segments:
        print(f"[build_index]   {trial['trial_id']}: {trial_length} rows too short for a safe "
              f"val carve-out at window_size={window_size} -- all windows assigned to train")

    # trim each val segment inward by window_size so no val window sits within
    # window_size rows of the train windows on either side of it
    for vs, ve in val_segments:
        tvs, tve = vs + window_size, ve - window_size
        for local_start in range(tvs, tve - window_size + 1):
            yield local_start, "val"

    # train = every window-start not inside an (untrimmed) val segment, i.e.
    # train windows may butt right up against a val segment's original edge
    # but never past it -- the window_size gap already lives inside the
    # trimmed-away part of the val segment itself.
    for local_start in range(0, trial_length - window_size + 1):
        if any(vs <= local_start < ve for vs, ve in val_segments):
            continue
        yield local_start, "train"


def build_index_df(all_df, history_horizon, forecast_horizon,
                    val_fraction=VAL_FRACTION, num_val_segments=NUM_VAL_SEGMENTS,
                    verbose=True):
    """Build the train/val window-start index for an already-loaded
    all_data.csv DataFrame (must have trial_id/subject_name/task columns),
    sized for a caller-specified window_size = history_horizon + forecast_horizon.

    Returns a DataFrame with columns: trial_id, subject_name, task, split,
    local_start_idx, global_start_idx, window_size -- the same schema written
    to Data/train_val_index.csv by the CLI entry point below. The window_size
    column lets a consumer verify a persisted CSV was built for a window at
    least as large as the one it's about to use (see module docstring).
    """
    window_size = history_horizon + forecast_horizon
    trials = _trial_ranges(all_df)
    if verbose:
        print(f"[build_index] {len(trials)} trials, window_size={window_size} "
              f"(history_horizon={history_horizon} + forecast_horizon={forecast_horizon})")

    rows = []
    for trial in trials:
        n_train = n_val = 0
        for local_start, split in _windows_for_trial(trial, window_size, val_fraction, num_val_segments):
            rows.append({
                "trial_id": trial["trial_id"],
                "subject_name": trial["subject_name"],
                "task": trial["task"],
                "split": split,
                "local_start_idx": local_start,
                "global_start_idx": trial["start"] + local_start,
                "window_size": window_size,
            })
            if split == "train":
                n_train += 1
            else:
                n_val += 1
        if verbose and (n_train or n_val):
            frac = n_val / (n_train + n_val)
            print(f"[build_index]   {trial['trial_id']} ({trial['subject_name']}, {trial['task']}): "
                  f"train={n_train:,}  val={n_val:,}  val_frac={frac:.1%}")

    return pd.DataFrame(rows)


def _print_summary(index_df, val_fraction):
    n_train_total = (index_df["split"] == "train").sum()
    n_val_total = (index_df["split"] == "val").sum()
    total = n_train_total + n_val_total
    print(f"\n[build_index] train={n_train_total:,}  val={n_val_total:,}  "
          f"val_frac={n_val_total / total:.1%}  (target {val_fraction:.0%})")

    print("\n[build_index] val_frac by subject:")
    print((index_df.groupby("subject_name")["split"]
           .apply(lambda s: (s == "val").mean())
           .rename("val_frac").to_string()))

    print("\n[build_index] val_frac by task:")
    print((index_df.groupby("task")["split"]
           .apply(lambda s: (s == "val").mean())
           .rename("val_frac").to_string()))


def build():
    """CLI entry point: build the default dense-window index from
    Data/all_data.csv and write it to Data/train_val_index.csv."""
    meta = pd.read_csv(DATA_PATH, usecols=["trial_id", "subject_name", "task"])
    index_df = build_index_df(meta, HISTORY_HORIZON, FORECAST_HORIZON, VAL_FRACTION, NUM_VAL_SEGMENTS)
    index_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[build_index] wrote {len(index_df):,} window starts -> {OUTPUT_PATH}")
    _print_summary(index_df, VAL_FRACTION)


if __name__ == "__main__":
    build()
