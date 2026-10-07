"""Combine every subject's per-trial CSVs into one Data/all_data_<rate>.csv.

Add new subjects to SUBJECTS as they're collected. For each subject,
Data/<subject>/<rate>/trial_*_all_data.csv are stacked on top of each other
(they all share the same columns) and tagged with a trial_id column
(e.g. "S01_trial_1") so trial/subject boundaries survive the concatenation --
downstream windowing/splitting needs that to avoid building a window that
spans two different trials.

Two parallel sample rates now exist side by side, one subfolder each:
  Data/<subject>/150/trial_*_all_data.csv  -- original 150 Hz data
  Data/<subject>/100/trial_*_all_data.csv  -- 100 Hz-resampled data (still
                                              being produced -- these folders
                                              exist but may be empty)
RATE below selects which one combine() reads, and the output filename is
tagged to match (Data/all_data_150.csv or Data/all_data_100.csv) -- never a
plain Data/all_data.csv anymore, so the two rates can never be silently
mixed up or overwrite each other. Downstream scripts that still default to
reading a plain Data/all_data.csv (train_fld.py's ALL_DATA_PATH and every
script that imports it, build_train_val_index.py, etc.) have NOT been
updated to the tagged names as part of this change -- update those
separately, or pass --rate 150 here and point them at Data/all_data_150.csv
explicitly, once you're ready to switch them over.

The per-trial CSVs already include the 10 joint angular-velocity
(*_velocity) columns, precomputed upstream -- so combine() just stacks them
as-is; nothing else to do. (Older trial exports didn't have these, which is
why this script used to compute them itself via add_joint_velocities() +
smooth_velocities() into a separate Data/all_data_with_velocities.csv --
that step has been removed now that every trial file already carries
velocities. Downstream scripts should read Data/all_data_<rate>.csv directly.)

Run this once after adding a new subject folder or new trial files:
    python3 combine_data.py                # RATE below (default 150)
    python3 combine_data.py --rate 100
    python3 combine_data.py --rate 150
"""
import argparse
import os
import glob
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")

SUBJECTS = ["S05"]

# Which sample-rate subfolder to combine by default when run with no --rate
# flag (python3 combine_data.py with no args, or combine() called directly).
RATE = "150"   # "150" or "100"


def combine(rate: str = RATE) -> pd.DataFrame:
    if rate not in ("100", "150"):
        raise ValueError(f"rate must be '100' or '150', got {rate!r}")

    output_path = os.path.join(DATA_DIR, f"all_data_S05_{rate}_corrected.csv")

    frames = []
    for subject in SUBJECTS:
        subject_dir = os.path.join(DATA_DIR, subject, rate)
        paths = sorted(glob.glob(os.path.join(subject_dir, "trial_*_all_data.csv")))
        if not paths:
            raise FileNotFoundError(
                f"No trial_*_all_data.csv files found in {subject_dir} -- "
                f"if rate={rate!r} is still being produced, this folder is "
                f"expected to be empty until then; combine with --rate "
                f"{'150' if rate == '100' else '100'} in the meantime, or "
                f"wait until {subject_dir} is populated."
            )
        for path in paths:
            trial_name = os.path.basename(path)[: -len("_all_data.csv")]
            df = pd.read_csv(path)
            df.insert(0, "trial_id", f"{subject}_{trial_name}")
            frames.append(df)
            print(f"[combine_data] {subject}/{rate}/{trial_name}: {len(df):,} rows")

    combined = pd.concat(frames, ignore_index=True)

    # Some trials' sensor streams stop a handful of samples short of the
    # rest of that trial's own columns (e.g. S03_trial_8's force-plate
    # columns end 29 rows early) -- those trailing rows read back as NaN.
    # Left alone, even a few NaN cells are catastrophic downstream: any
    # global normalisation stat (mean/std) computed by concatenating a
    # NaN-containing column poisons that column's mean/std into NaN, which
    # then poisons EVERY trial's normalised data for that column, not just
    # the trial the NaN came from -- confirmed on this exact data (a single
    # trial's 29 NaN rows turned an entire multi-subject training run's
    # loss to NaN from epoch 1 onward). Zero-fill numeric columns only
    # (trial_id/subject_name/task are strings and shouldn't have NaNs to
    # begin with; restricting to numeric dtypes avoids accidentally
    # stamping a 0 into a string column if one ever did).
    numeric_cols = combined.select_dtypes(include="number").columns
    nan_counts = combined[numeric_cols].isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if len(nan_counts):
        print(f"[combine_data] WARNING -- NaN found in {len(nan_counts)} numeric column(s), zero-filling:")
        for col, n in nan_counts.items():
            print(f"[combine_data]   {col}: {n} NaN row(s)")
        combined[numeric_cols] = combined[numeric_cols].fillna(0)
    else:
        print("[combine_data] No NaN found in any numeric column.")

    combined.to_csv(output_path, index=False)
    print(f"[combine_data] wrote {len(combined):,} rows x {combined.shape[1]} cols -> {output_path}")
    return combined


def _parse_args():
    p = argparse.ArgumentParser(description="Combine per-trial CSVs into one Data/all_data_<rate>.csv.")
    p.add_argument("--rate", choices=["100", "150"], default=RATE,
                    help=f"Which sample-rate subfolder to combine (Data/<subject>/<rate>/...). Default: {RATE}")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    combine(rate=args.rate)
