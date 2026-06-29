"""
preprocessing.py
================
Module: src/qubo_project/preprocessing.py

Reads a numerical CSV dataset, drops sparse/zero-heavy columns, applies
Z-score normalisation and saves the cleaned dataset together with a
statistics JSON file.

Designed to handle datasets with 1.5 M+ records via chunked I/O (pandas
read_csv with chunksize) and incremental statistics (Welford online
algorithm for mean/variance) so the full dataset never has to reside in
memory all at once.

CLI usage
---------
python preprocessing.py \\
    --input  dati_credito.csv \\
    --target target \\
    --out-data  normalized.csv \\
    --out-json  preprocessing_result.json \\
    --min-perc-valid 0.06

Output files are always written inside the ``outputs/`` directory that
sits next to the repository root.  Paths are kept relative so the
project is fully portable.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Number of rows processed per I/O chunk.  Tune this to balance memory
# usage and throughput.  128 k rows ≈ 50–150 MB for typical tabular data.
_CHUNK_SIZE: int = 128_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_outputs_dir() -> Path:
    """Return the absolute path to the ``outputs/`` directory.

    The function walks upward from this file's location until it finds
    a directory that contains an ``outputs/`` sub-directory (repository
    root) or falls back to ``./outputs`` relative to the current working
    directory.  Either way, the directory is created if it does not yet
    exist.
    """
    here = Path(__file__).resolve().parent
    # Walk up at most 6 levels looking for the repo root
    candidate = here
    for _ in range(6):
        outputs = candidate / "outputs"
        if outputs.is_dir():
            return outputs
        candidate = candidate.parent

    # Fallback: outputs/ next to cwd
    fallback = Path.cwd() / "outputs"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _to_output_path(user_path: str) -> Path:
    """Resolve *user_path* to a path inside the ``outputs/`` directory.

    If *user_path* is already an absolute path or contains directory
    components, only the filename part is extracted and placed inside
    ``outputs/``.  This guarantees no absolute paths leak into the code
    while still honouring the caller's desired filename.
    """
    outputs_dir = _resolve_outputs_dir()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(user_path).name  # keep only the basename
    return outputs_dir / filename


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def fit_normalize(
    input_csv: str,
    target_column: str,
    normalized_csv: str,
    outInitalRes_json: str,
    minPercValid: float = 0.05,
) -> dict:
    """Read, clean and Z-score normalise a numerical CSV dataset.

    Parameters
    ----------
    input_csv:
        Path to the input CSV file (relative or absolute).  The first row
        must contain column headers.
    target_column:
        Name of the binary (0/1) target column.  This column is excluded
        from filtering and normalisation but is preserved in the output.
    normalized_csv:
        Filename for the output normalised CSV.  The file is written to
        the ``outputs/`` directory regardless of any directory component
        in the supplied string.
    outInitalRes_json:
        Filename for the output statistics JSON.  Same path rules as
        *normalized_csv*.
    minPercValid:
        Minimum fraction (0–1) of rows that must contain a **non-NaN,
        non-zero** value for a feature column to be retained.  Columns
        below this threshold are dropped before normalisation.
        Default: 0.05 (5 %).

    Returns
    -------
    dict
        The same dictionary that is serialised to *outInitalRes_json*,
        containing the keys:
        ``n_input_features``, ``n_kept_features``, ``dataset_size``,
        ``dataset_input_time``, ``dataset_processing_time``,
        ``dropped_feature_names``.

    Raises
    ------
    FileNotFoundError
        If *input_csv* does not exist.
    ValueError
        If *target_column* is not found in the CSV header, or if the
        resulting feature set is empty after filtering.

    Notes
    -----
    **Memory efficiency for large datasets**

    The function uses two passes over the file:

    1. *Statistics pass* – iterates the CSV in chunks to accumulate the
       total row count, per-column non-zero/non-NaN counts, and Welford
       online mean/variance (numerically stable, single-pass).

    2. *Write pass* – applies the computed mean/std to each chunk,
       writes normalised output in append mode.

    Peak memory is proportional to *_CHUNK_SIZE* rows, not to the full
    dataset size, making the function suitable for 1.5 M+ record files.

    **Z-score normalisation**

    ``x_norm = (x - mean) / std``

    Columns with zero standard deviation (all-constant after filtering)
    are set to 0.0 to avoid division-by-zero.

    NaN values are imputed with the column mean **before** normalisation,
    which results in 0.0 in the normalised space (a sensible default that
    avoids leaking test-set statistics when the scaler is later applied
    to new data via a saved model).
    """
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path.resolve()}")

    out_csv_path = _to_output_path(normalized_csv)
    out_json_path = _to_output_path(outInitalRes_json)

    logger.info("Input  : %s", input_path.resolve())
    logger.info("Output CSV  : %s", out_csv_path)
    logger.info("Output JSON : %s", out_json_path)
    logger.info("Target column   : '%s'", target_column)
    logger.info("Min %% valid     : %.4f (%.2f%%)", minPercValid, minPercValid * 100)

    # ------------------------------------------------------------------
    # 0. Peek at the header to discover columns
    # ------------------------------------------------------------------
    header_df = pd.read_csv(input_path, nrows=0)
    all_columns: list[str] = list(header_df.columns)

    if target_column not in all_columns:
        raise ValueError(
            f"Target column '{target_column}' not found in CSV. "
            f"Available columns: {all_columns}"
        )

    feature_columns: list[str] = [c for c in all_columns if c != target_column]
    n_input_features: int = len(feature_columns)
    logger.info("Input features  : %d", n_input_features)

    # ------------------------------------------------------------------
    # PASS 1 — gather statistics
    # Uses Welford's online algorithm for numerically stable mean/variance
    # without loading the entire dataset into memory.
    # ------------------------------------------------------------------
    t_input_start = time.perf_counter()

    # Welford accumulators indexed over feature columns
    welford_n: np.ndarray = np.zeros(n_input_features, dtype=np.float64)
    welford_mean: np.ndarray = np.zeros(n_input_features, dtype=np.float64)
    welford_M2: np.ndarray = np.zeros(n_input_features, dtype=np.float64)

    # Count of rows with a non-NaN AND non-zero value per feature column
    valid_count: np.ndarray = np.zeros(n_input_features, dtype=np.float64)

    total_rows: int = 0

    chunk_iter = pd.read_csv(
        input_path,
        chunksize=_CHUNK_SIZE,
        low_memory=False,
    )

    for chunk in chunk_iter:
        # Coerce feature columns to numeric; non-parseable values become NaN
        feat_chunk: pd.DataFrame = chunk[feature_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        arr: np.ndarray = feat_chunk.to_numpy(dtype=np.float64)
        n_rows_chunk: int = arr.shape[0]
        total_rows += n_rows_chunk

        # --- valid-value counts: non-NaN AND non-zero ---
        valid_mask = (~np.isnan(arr)) & (arr != 0.0)
        valid_count += valid_mask.sum(axis=0)

        # --- Welford online update for mean/variance (ignoring NaN) ---
        # We use a vectorised batch update of the Welford algorithm.
        # For each column, iterate only over its finite (non-NaN) values.
        for j in range(n_input_features):
            col_vals = arr[:, j]
            col_finite = col_vals[~np.isnan(col_vals)]
            if col_finite.size == 0:
                continue
            # Batch Welford update (Chan et al. parallel form)
            n_b = col_finite.size
            mean_b = col_finite.mean()
            M2_b = ((col_finite - mean_b) ** 2).sum()

            combined_n = welford_n[j] + n_b
            delta = mean_b - welford_mean[j]
            welford_mean[j] = (
                welford_n[j] * welford_mean[j] + n_b * mean_b
            ) / combined_n
            welford_M2[j] += M2_b + delta ** 2 * welford_n[j] * n_b / combined_n
            welford_n[j] = combined_n

    t_input_end = time.perf_counter()
    dataset_input_time: float = round(t_input_end - t_input_start, 4)
    logger.info(
        "Pass 1 complete: %d rows read in %.2f s", total_rows, dataset_input_time
    )

    if total_rows == 0:
        raise ValueError("The input CSV contains no data rows.")

    # Finalise variance → population std (ddof=1, consistent with sklearn)
    with np.errstate(invalid="ignore", divide="ignore"):
        variance = np.where(
            welford_n > 1,
            welford_M2 / (welford_n - 1),
            0.0,
        )
    col_std: np.ndarray = np.sqrt(np.maximum(variance, 0.0))
    col_mean: np.ndarray = welford_mean.copy()

    # ------------------------------------------------------------------
    # Column filtering
    # ------------------------------------------------------------------
    valid_fraction: np.ndarray = valid_count / total_rows
    keep_mask: np.ndarray = valid_fraction >= minPercValid
    kept_features: list[str] = [
        c for c, keep in zip(feature_columns, keep_mask) if keep
    ]
    dropped_features: list[str] = [
        c for c, keep in zip(feature_columns, keep_mask) if not keep
    ]

    n_kept_features: int = len(kept_features)
    logger.info(
        "Columns kept: %d / %d  (dropped %d with < %.2f%% valid values)",
        n_kept_features,
        n_input_features,
        len(dropped_features),
        minPercValid * 100,
    )
    if n_kept_features == 0:
        raise ValueError(
            "All feature columns were dropped by the minPercValid filter. "
            "Lower the --min-perc-valid threshold."
        )
    if dropped_features:
        logger.info("Dropped columns: %s", dropped_features)

    # Restrict statistics arrays to kept columns only
    keep_idx: np.ndarray = np.array(
        [i for i, keep in enumerate(keep_mask) if keep], dtype=int
    )
    kept_mean: np.ndarray = col_mean[keep_idx]
    kept_std: np.ndarray = col_std[keep_idx]

    # Columns with std == 0 (constant) → normalised value is always 0
    safe_std: np.ndarray = np.where(kept_std == 0.0, 1.0, kept_std)

    # ------------------------------------------------------------------
    # PASS 2 — normalise chunks and write output CSV
    # ------------------------------------------------------------------
    t_proc_start = time.perf_counter()

    # Remove stale output file so we can safely append
    if out_csv_path.exists():
        out_csv_path.unlink()

    chunk_iter2 = pd.read_csv(
        input_path,
        chunksize=_CHUNK_SIZE,
        low_memory=False,
    )

    first_chunk: bool = True
    for chunk in chunk_iter2:
        # Preserve the target column as integer (0/1)
        target_series: pd.Series = (
            pd.to_numeric(chunk[target_column], errors="coerce")
            .fillna(0)
            .astype(int)
        )

        feat_chunk = chunk[kept_features].apply(pd.to_numeric, errors="coerce")
        arr = feat_chunk.to_numpy(dtype=np.float64).copy()  # ensure writeable

        # Impute NaN with the column training mean (→ 0.0 after z-score)
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            col_indices = np.where(nan_mask)[1]
            arr[nan_mask] = kept_mean[col_indices]

        # Z-score normalisation
        arr_norm = (arr - kept_mean) / safe_std

        # Reconstruct DataFrame: kept features + target column
        out_df = pd.DataFrame(arr_norm, columns=kept_features)
        out_df[target_column] = target_series.values

        out_df.to_csv(
            out_csv_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )
        first_chunk = False

    t_proc_end = time.perf_counter()
    dataset_processing_time: float = round(t_proc_end - t_proc_start, 4)
    logger.info(
        "Pass 2 complete: normalised CSV written in %.2f s",
        dataset_processing_time,
    )
    logger.info("Normalised CSV  : %s", out_csv_path)

    # ------------------------------------------------------------------
    # Write JSON statistics
    # ------------------------------------------------------------------
    stats: dict = {
        "n_input_features": n_input_features,
        "n_kept_features": n_kept_features,
        "dataset_size": total_rows,
        "dataset_input_time": dataset_input_time,
        "dataset_processing_time": dataset_processing_time,
        "dropped_feature_names": dropped_features,
    }

    out_json_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Statistics JSON : %s", out_json_path)
    logger.info("Preprocessing complete.")

    return stats


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="preprocessing.py",
        description=(
            "Preprocess a numerical CSV dataset: drop sparse/zero-heavy "
            "columns, apply Z-score normalisation and save results."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="FILE",
        help="Path to the input CSV dataset.",
    )
    parser.add_argument(
        "--target",
        required=True,
        metavar="COLUMN",
        help="Name of the binary (0/1) target column.",
    )
    parser.add_argument(
        "--out-data",
        required=True,
        metavar="FILE",
        help="Output filename for the normalised CSV (written to outputs/).",
    )
    parser.add_argument(
        "--out-json",
        required=True,
        metavar="FILE",
        help="Output filename for the statistics JSON (written to outputs/).",
    )
    parser.add_argument(
        "--min-perc-valid",
        type=float,
        default=0.05,
        metavar="FLOAT",
        help=(
            "Minimum fraction of rows that must be non-NaN and non-zero "
            "for a feature column to be retained (0–1)."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """Parse CLI arguments and invoke :func:`fit_normalize`."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not (0.0 <= args.min_perc_valid <= 1.0):
        parser.error("--min-perc-valid must be between 0.0 and 1.0")

    try:
        stats = fit_normalize(
            input_csv=args.input,
            target_column=args.target,
            normalized_csv=args.out_data,
            outInitalRes_json=args.out_json,
            minPercValid=args.min_perc_valid,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)

    # Echo the output paths to stdout (mirrors the spec "Output atteso")
    out_csv_path = _to_output_path(args.out_data)
    out_json_path = _to_output_path(args.out_json)
    print(str(out_csv_path))
    print(str(out_json_path))

    # Human-readable summary
    print(
        f"\nSummary:\n"
        f"  Input features  : {stats['n_input_features']}\n"
        f"  Kept features   : {stats['n_kept_features']}\n"
        f"  Dropped features: {len(stats['dropped_feature_names'])}\n"
        f"  Dataset rows    : {stats['dataset_size']}\n"
        f"  Read time       : {stats['dataset_input_time']:.2f} s\n"
        f"  Process time    : {stats['dataset_processing_time']:.2f} s\n"
    )


if __name__ == "__main__":
    main()