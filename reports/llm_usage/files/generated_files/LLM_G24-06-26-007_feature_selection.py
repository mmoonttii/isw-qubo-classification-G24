"""
feature_selection.py — QUBO-based feature selection for binary classification.

Implements select_features() and an argparse CLI.
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────────
import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime


# ── OUTPUT PATH HELPER ────────────────────────────────────────────────────────

def _to_output_path(filename: str) -> str:
    """Route a bare filename (or relative path) into the outputs/ directory."""
    os.makedirs("outputs", exist_ok=True)
    return os.path.join("outputs", os.path.basename(filename))


# ── MAIN FUNCTION ─────────────────────────────────────────────────────────────

def select_features(
    normalized_csv: str,       # Input: normalized dataset (output of preprocessing.py)
    reducedTrain_csv: str,     # Output: training dataset with only selected features
    reducedTest_csv: str,      # Output: test dataset with only selected features
    output_ottim_csv: str,     # Output: CSV with one row per alpha tried
    output_json: str,          # Output: JSON with stats and selected feature info
    target_column: str,        # Name of the binary target column
    percTest: float = 0.30,    # Fraction of dataset to use as test set
    percSelected: float = 0.20,# Fraction of features to select
    allowance: int = 1,        # Tolerance: K ± allowance features is acceptable
    seed: int = 42,            # RNG seed for reproducibility
    alpha_computations: int = 100  # Max number of alpha values to try
) -> None:

    # ── LOAD NORMALIZED CSV (CHUNKED) ─────────────────────────────────────────
    print(f"[{datetime.now():%H:%M:%S}] Loading normalized dataset from '{normalized_csv}' ...")

    chunks = []
    for chunk in pd.read_csv(normalized_csv, chunksize=131072):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)

    # Validate that the target column is present
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in '{normalized_csv}'. "
            f"Available columns: {list(df.columns)}"
        )

    total_samples = len(df)
    n_cols = len(df.columns)
    print(
        f"[{datetime.now():%H:%M:%S}] Dataset loaded: "
        f"{total_samples} rows, {n_cols} columns (including target)."
    )

    # ── TRAINING / TEST HARD SPLIT ────────────────────────────────────────────
    # First M rows → training set; remaining rows → test set.
    # M = total_samples - round(percTest * total_samples)
    M = total_samples - round(percTest * total_samples)

    df_train = df.iloc[:M].reset_index(drop=True)
    df_test  = df.iloc[M:].reset_index(drop=True)

    print(
        f"[{datetime.now():%H:%M:%S}] Split complete: "
        f"{len(df_train)} training samples, {len(df_test)} test samples "
        f"(cut at row index {M})."
    )

    # ── TODO: SPEARMAN CORRELATIONS & Q MATRIX ───────────────────────────────
    # (to be implemented in the next checklist step)

    # ── TODO: ALPHA BISECTION SEARCH ─────────────────────────────────────────
    # (to be implemented in the next checklist step)

    # ── TODO: SAVE OUTPUT FILES ───────────────────────────────────────────────
    # (to be implemented in the next checklist step)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QUBO-based feature selection for binary classification."
    )
    parser.add_argument("--in-normalized",    required=True,  help="Normalized input CSV")
    parser.add_argument("--out-train",        required=True,  help="Output training CSV")
    parser.add_argument("--out-test",         required=True,  help="Output test CSV")
    parser.add_argument("--out-optimizations",required=True,  help="Output optimizations CSV")
    parser.add_argument("--out-json",         required=True,  help="Output JSON file")
    parser.add_argument("--target",           required=True,  help="Name of the target column")
    parser.add_argument("--perc-selected",    type=float, default=0.20,
                        help="Fraction of features to select (default: 0.20)")
    parser.add_argument("--allowance",        type=int,   default=1,
                        help="Tolerance on the number of selected features (default: 1)")
    parser.add_argument("--perc-test",        type=float, default=0.30,
                        help="Fraction of dataset to use as test set (default: 0.30)")
    parser.add_argument("--seed",             type=int,   default=42,
                        help="RNG seed for reproducibility (default: 42)")
    parser.add_argument("--alpha-computations", type=int, default=100,
                        help="Max number of alpha values to try (default: 100)")

    args = parser.parse_args()

    select_features(
        normalized_csv    = args.in_normalized,
        reducedTrain_csv  = _to_output_path(args.out_train),
        reducedTest_csv   = _to_output_path(args.out_test),
        output_ottim_csv  = _to_output_path(args.out_optimizations),
        output_json       = _to_output_path(args.out_json),
        target_column     = args.target,
        percTest          = args.perc_test,
        percSelected      = args.perc_selected,
        allowance         = args.allowance,
        seed              = args.seed,
        alpha_computations= args.alpha_computations,
    )

    # Print the four output paths in the required order
    print(_to_output_path(args.out_train))
    print(_to_output_path(args.out_test))
    print(_to_output_path(args.out_optimizations))
    print(_to_output_path(args.out_json))