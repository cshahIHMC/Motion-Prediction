"""Combine every subject's per-trial CSVs into one Data/all_data.csv.

Add new subjects to SUBJECTS as they're collected. For each subject,
Data/<subject>/trial_*_all_data.csv are stacked on top of each other (they
all share the same 171 columns) and tagged with a trial_id column
(e.g. "S01_trial_1") so trial/subject boundaries survive the concatenation --
downstream windowing/splitting needs that to avoid building a window that
spans two different trials.

Run this once after adding a new subject folder or new trial files.
"""
import os
import glob
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
OUTPUT_PATH = os.path.join(DATA_DIR, "all_data.csv")

SUBJECTS = ["S01", "S02"]


def combine():
    frames = []
    for subject in SUBJECTS:
        subject_dir = os.path.join(DATA_DIR, subject)
        paths = sorted(glob.glob(os.path.join(subject_dir, "trial_*_all_data.csv")))
        if not paths:
            raise FileNotFoundError(f"No trial_*_all_data.csv files found in {subject_dir}")
        for path in paths:
            trial_name = os.path.basename(path)[: -len("_all_data.csv")]
            df = pd.read_csv(path)
            df.insert(0, "trial_id", f"{subject}_{trial_name}")
            frames.append(df)
            print(f"[combine_data] {subject}/{trial_name}: {len(df):,} rows")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"[combine_data] wrote {len(combined):,} rows x {combined.shape[1]} cols -> {OUTPUT_PATH}")


if __name__ == "__main__":
    combine()
