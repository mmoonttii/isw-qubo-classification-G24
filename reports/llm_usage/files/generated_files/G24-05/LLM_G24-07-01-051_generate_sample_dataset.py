"""
scripts/generate_sample_dataset.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
One-time generator for the static sample dataset required by spec §13:

    data/sample_test_dataset.csv

This script is run manually (or once, when the file needs to be
(re)created) and its output is committed to version control. It is
NOT invoked by the test suite: tests/test_pipeline.py only *reads*
data/sample_test_dataset.csv, it never generates it.

The generated dataset deliberately mirrors the structure described in
spec §6/§7/§13 on a small scale, so it exercises every code path the
automatic tests check for:

    • N_SAMPLES  (120) rows
    • N_FEATURES (20) numeric feature columns: feature_00 ... feature_19
    • N_SPARSE_COLS (3) columns with >= 98% zeros → must be dropped by
      fit_normalize() at the default/near-default threshold
    • ~NAN_FRACTION (5%) random missing values in the remaining columns
    • a binary "target" column with POS_FRACTION (20%) positive samples,
      comfortably above the >=10%-per-class floor required by §13

Usage
-----
    python scripts/generate_sample_dataset.py

Re-running this script regenerates the exact same file (fixed seed), so
it is safe to run again if the CSV is ever lost or needs to be rebuilt.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── dataset shape constants (must match tests/test_pipeline.py) ───────────
TARGET        = "target"
N_SAMPLES     = 120
N_FEATURES    = 20
N_SPARSE_COLS = 3
NAN_FRACTION  = 0.05
POS_FRACTION  = 0.20
SEED          = 42

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_test_dataset.csv"


def build_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    # ── feature matrix: standard-normal values ─────────────────────────
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))

    # ── inject NaN into the non-sparse block (columns N_SPARSE_COLS…end)
    nan_mask = rng.random((N_SAMPLES, N_FEATURES - N_SPARSE_COLS)) < NAN_FRACTION
    X[:, N_SPARSE_COLS:][nan_mask] = np.nan

    # ── make the first N_SPARSE_COLS columns near-all-zero (>=98% zeros)
    for col_idx in range(N_SPARSE_COLS):
        col_data = np.zeros(N_SAMPLES)
        nz_idx = rng.choice(N_SAMPLES, size=2, replace=False)
        col_data[nz_idx] = rng.standard_normal(2)
        X[:, col_idx] = col_data

    # ── binary target: guarantee >=10% representation for each class ───
    n_pos = max(int(N_SAMPLES * POS_FRACTION), int(N_SAMPLES * 0.10) + 1)
    y = np.zeros(N_SAMPLES, dtype=np.int8)
    pos_idx = rng.choice(N_SAMPLES, size=n_pos, replace=False)
    y[pos_idx] = 1

    assert y.sum() >= 0.10 * N_SAMPLES, "fewer than 10% positive samples"
    assert (1 - y).sum() >= 0.10 * N_SAMPLES, "fewer than 10% negative samples"

    col_names = [f"feature_{i:02d}" for i in range(N_FEATURES)]
    df = pd.DataFrame(X, columns=col_names)
    df[TARGET] = y.astype(int)
    return df


def main() -> None:
    df = build_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows x {len(df.columns)} cols -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()