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
from scipy.stats import spearmanr


# ── OUTPUT PATH HELPER ────────────────────────────────────────────────────────

def _to_output_path(filename: str) -> str:
    """Route a bare filename (or relative path) into the outputs/ directory."""
    os.makedirs("outputs", exist_ok=True)
    return os.path.join("outputs", os.path.basename(filename))


# ── QUBO MATRIX BUILDER ───────────────────────────────────────────────────────

def _build_Q(alpha: float, rho_Vj: np.ndarray, rho_jk: np.ndarray) -> np.ndarray:
    """Build the symmetric QUBO matrix Q for the given alpha.

    Diagonal:     Q[j, j] = -alpha * |rho_Vj|
    Off-diagonal: Q[j, k] = (1 - alpha) * |rho_jk|   (j ≠ k, matrix is symmetric)

    Parameters
    ----------
    alpha   : weighting parameter in [0, 1]
    rho_Vj  : 1-D array of shape (n,) — |Spearman(feature_j, target)|
    rho_jk  : 2-D array of shape (n, n) — |Spearman(feature_j, feature_k)|
    """
    # Fill the whole matrix with the off-diagonal formula first …
    Q = (1.0 - alpha) * rho_jk
    # … then overwrite the diagonal with its own formula.
    np.fill_diagonal(Q, -alpha * rho_Vj)
    return Q


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

    # ── SPEARMAN CORRELATIONS (TRAINING SET ONLY, TIMED) ─────────────────────
    # Correlations are computed once here; only Q matrix values change with alpha.
    feature_cols = [c for c in df_train.columns if c != target_column]
    n = len(feature_cols)
    K = round(percSelected * n)   # target number of features to select

    U_train = df_train[feature_cols].values   # shape (M, n)
    V_train = df_train[target_column].values  # shape (M,)

    print(
        f"[{datetime.now():%H:%M:%S}] Computing Spearman correlations "
        f"({n} features × {len(U_train)} training samples, target K={K}) ..."
    )

    t_corr_start = time.perf_counter()

    # Stack features and target into one matrix so spearmanr processes everything
    # in a single call — avoids n redundant rank computations.
    combined  = np.column_stack([U_train, V_train])   # shape (M, n+1)
    stat      = spearmanr(combined).statistic

    # scipy returns a scalar when the input has exactly 2 columns (n == 1);
    # normalise to a 2×2 matrix so the slicing below is always valid.
    if n == 1:
        corr_full = np.array([[1.0, float(stat)],
                              [float(stat), 1.0]])
    else:
        corr_full = stat                               # shape (n+1, n+1)

    rho_jk = np.abs(corr_full[:n, :n])   # feature-feature absolute correlations
    rho_Vj = np.abs(corr_full[:n,  n])   # feature-target  absolute correlations

    q_matrix_creation_time = time.perf_counter() - t_corr_start

    print(
        f"[{datetime.now():%H:%M:%S}] Spearman correlations computed in "
        f"{q_matrix_creation_time:.3f}s."
    )

    # ── Q MATRIX CONSTRUCTION ─────────────────────────────────────────────────
    # The matrix is rebuilt inside the alpha loop via _build_Q(alpha, rho_Vj, rho_jk).
    # Diagonal:     Q[j, j] = -alpha * |rho_Vj|
    # Off-diagonal: Q[j, k] = (1 - alpha) * |rho_jk|   (symmetric)

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