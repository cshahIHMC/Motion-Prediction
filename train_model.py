"""
train_model.py

Entry point for the Motion-Prediction pipeline.
Currently: loads the dataset CSV and prints all column names.
"""

import sys
import os

# Make sibling packages importable when running from the repo root.
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from Library.utility import print_column_names

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(os.path.dirname(__file__), "Data", "trial_1_all_data.csv")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading data from:\n  {DATA_PATH}\n")
    df = pd.read_csv(DATA_PATH)
    print_column_names(df)


if __name__ == "__main__":
    main()
