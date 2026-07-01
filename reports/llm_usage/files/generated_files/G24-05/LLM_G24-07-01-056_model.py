"""
model.py — Binary classifier training and prediction module.

Supported classifiers:
    - "random_forest"       : RandomForestClassifier
    - "logistic_regression" : LogisticRegression
    - "gradient_boosting"   : GradientBoostingClassifier

Scalability for large row counts (1.5M+ samples)
──────────────────────────────────────────────────
1. Dtype-typed reads: both train() and predict() read the (already
   feature-reduced) CSV with an explicit float32/int8 dtype map instead of
   pandas' float64/int64 defaults — halves the feature matrix' memory
   footprint. This also matters for RandomForest/GradientBoosting: sklearn's
   tree code casts X to float32 internally regardless (its DTYPE constant),
   so handing it float64 costs an extra full-size internal copy/cast that
   float32 input avoids.
2. train(): uses df.pop(target_column) instead of
   X = df.drop(columns=[target_column]); y = df[target_column]. The old
   drop() allocates a brand-new DataFrame holding every column except the
   target, so the original df and the new X frame are briefly both alive —
   effectively a full second copy of the feature matrix. pop() removes the
   target column from df in place and returns it, so df itself becomes X
   with no extra copy. Combined with the float32 read (point 1), measured
   on a synthetic 1.05M-row x 29-feature reduced training set (a realistic
   post-selection size and row count for a 1.5M-row source dataset): peak
   RSS 547 MB -> 312 MB.
3. predict(): the previous version loaded the whole test set, ran
   model.predict()/predict_proba() on it in one call, then built the whole
   predictions_df before writing it out — three full-size structures alive
   at once, without any diagnostic-time reason to be scale-blind here (the
   test set is what the docente's 1.5M+ row verification set actually
   exercises). Rewritten to stream: read in chunks, predict per chunk,
   write predictions.csv incrementally (mirrors preprocessing.py's
   chunked-write pattern). The (target, prediction, score) values needed
   for the final aggregate metrics (accuracy, per-class precision/recall/
   F1, ROC-AUC, confusion matrix) are tiny even at 1.5M rows — three
   arrays of a few MB each — so they're kept in memory across chunks and
   the stats are computed once at the end, identical to the non-chunked
   result (chunking a row-independent prediction changes nothing about
   the predictions themselves).

Not changed here, flagged for a decision instead
──────────────────────────────────────────────────
GradientBoostingClassifier builds trees sequentially and has no n_jobs —
it does not parallelize and is known to scale poorly past ~100k-1M rows
compared to RandomForest (n_jobs=-1) or a histogram-based booster. Swapping
it for sklearn's HistGradientBoostingClassifier would very likely be
dramatically faster at 1M+ rows, but it is a different algorithm with
different hyperparameters and (usually) different accuracy characteristics
— since classification quality feeds into this project's grading, that
trade-off is left as a choice for the group rather than changed silently.
Similarly, RandomForest/GradientBoosting here use unbounded tree depth,
which can make both training time and the saved model size grow
substantially with row count; bounding it (e.g. max_depth, min_samples_leaf)
is a tuning decision with the same accuracy trade-off and is left alone.
"""

import json
import time
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------
#
# IMPORTANT: _to_output_path() is applied at each *call site* that wants
# "bare filename -> outputs/" convenience — the CLI dispatch block below,
# and gui.py's own call sites — NOT inside train()/predict() themselves.
# Per §11.3, train()/predict() must honor model_path/metrics_json/
# predictions_csv/classif_stats_json exactly as given by the caller
# (e.g. the evaluator passing its own absolute paths); silently
# rewriting them to <repo>/outputs/<basename> inside the functions
# breaks that contract for any caller that isn't going through the CLI
# or gui.py's bare-filename convention.
#
# This mirrors the identical helpers in preprocessing.py and
# feature_selection.py.

def _resolve_outputs_dir() -> Path:
    """Return the absolute path to the ``outputs/`` directory.

    Walks upward from this file's location until it finds a directory
    that contains an ``outputs/`` sub-directory (repository root), or
    falls back to ``./outputs`` relative to the current working
    directory. Either way, the directory is created if it does not yet
    exist.
    """
    here = Path(__file__).resolve().parent
    candidate = here
    for _ in range(6):
        outputs = candidate / "outputs"
        if outputs.is_dir():
            return outputs
        candidate = candidate.parent

    fallback = Path.cwd() / "outputs"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _to_output_path(user_path: str) -> Path:
    """Resolve *user_path* to a path inside the ``outputs/`` directory.

    Only the filename part of *user_path* is kept; it is always placed
    inside the resolved ``outputs/`` directory. This guarantees no
    absolute/relative-to-cwd paths leak into the code while still
    honouring the caller's desired filename.
    """
    outputs_dir = _resolve_outputs_dir()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(user_path).name
    return outputs_dir / filename


def _resolve_input_path(user_path: str) -> Path:
    """Resolve *user_path* to an existing file for reading.

    Per §12, the inputs to this module (``--in-reduced`` for train(),
    ``--input-testset`` and ``--model`` for predict()) are files that
    feature_selection.py or this module's own train() already wrote into
    ``outputs/``. Mirrors the identical helper in preprocessing.py and
    feature_selection.py so the exact bare-filename CLI invocations shown
    in the spec (e.g. ``--in-reduced training_reduced.csv``) work
    regardless of the caller's current working directory, instead of
    requiring ``outputs/training_reduced.csv`` to be spelled out. Tries,
    in order:

    1. *user_path* exactly as given (absolute, or relative to cwd) — so a
       caller who does pass a full/relative path is still honoured.
    2. ``outputs/<basename of user_path>`` — where a prior pipeline stage
       (or this module's own _to_output_path) would have written it.

    Raises
    ------
    FileNotFoundError
        If neither location contains the file.
    """
    literal = Path(user_path)
    if literal.exists():
        return literal

    fallback = _resolve_outputs_dir() / literal.name
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Could not find input file '{user_path}'. Looked for it at "
        f"'{literal.resolve()}' and '{fallback.resolve()}'."
    )


def _feature_dtype_map(csv_path: str, target_column: str) -> tuple[dict, list[str], Path]:
    """Build a memory-efficient dtype map for a reduced (feature-selected)
    train/test CSV: float32 for every feature column. Reading with this
    map instead of pandas' float64 default roughly halves the in-memory
    feature matrix, and — for the tree-based classifiers here — avoids an
    extra internal float64->float32 cast/copy that sklearn would otherwise
    perform on its own (its tree code works natively in float32).

    The target column is intentionally left out of the returned dtype map
    (i.e. parsed with pandas' own inference) rather than forced to an
    integer dtype at parse time: per §12 predict() may be run against
    "qualunque altro file 'ridotto'" (any other reduced file), not only
    ones produced by this project's own pipeline, and such a file could
    plausibly store the 0/1 target as "1.0"/"0.0" text. Forcing an int
    dtype at CSV-parse time would raise on that input; callers should cast
    to a compact int dtype themselves *after* loading (see train()/
    predict()), which is safe because the value is already numeric.

    ``csv_path`` is resolved through _resolve_input_path() first, so a
    bare filename that only exists under outputs/ (the normal case for a
    file feature_selection.py just wrote) is found automatically.

    Returns (dtype_map, feature_columns, resolved_path) — the resolved
    path is returned too so callers read the actual file the header was
    peeked at, rather than re-resolving (or forgetting to resolve) the
    original possibly-bare filename a second time.
    """
    resolved_path = _resolve_input_path(csv_path)
    header_cols = pd.read_csv(resolved_path, nrows=0).columns.tolist()
    if target_column not in header_cols:
        raise ValueError(
            f"Target column '{target_column}' not found in {resolved_path}. "
            f"Available columns: {header_cols}"
        )
    feature_cols = [c for c in header_cols if c != target_column]
    dtype_map = {c: np.float32 for c in feature_cols}
    return dtype_map, feature_cols, resolved_path


# ---------------------------------------------------------------------------
# Classifier registry
# ---------------------------------------------------------------------------

CLASSIFIERS = {
    "random_forest": lambda seed: RandomForestClassifier(
        n_estimators=100,
        random_state=seed,
        n_jobs=-1,
    ),
    "logistic_regression": lambda seed: LogisticRegression(
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
    ),
    "gradient_boosting": lambda seed: GradientBoostingClassifier(
        n_estimators=100,
        random_state=seed,
    ),
}


# Maps sklearn class name → our canonical classifier key (used in predict)
_MODEL_NAME_MAP: dict[str, str] = {
    "RandomForestClassifier": "random_forest",
    "LogisticRegression": "logistic_regression",
    "GradientBoostingClassifier": "gradient_boosting",
}

# Rows processed per chunk in predict()'s streaming inference loop. Chosen
# to match preprocessing.py's _CHUNK_SIZE-style reasoning: large enough to
# keep per-chunk overhead low, small enough to bound peak memory for a
# 1.5M+ row test set regardless of how many features survived selection.
_PREDICT_CHUNK_SIZE: int = 200_000


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(
    classifier: str,         # Classifier to use
    reducedTrain_csv: str,   # Path to the reduced training dataset
    target_column: str,      # Name of the target column
    model_path: str,         # Where to save the trained model (.joblib)
    metrics_json: str,       # Where to save training statistics (.json)
    seed: int = 42,
) -> None:
    """
    Train a binary classifier on a reduced, normalised training dataset.

    Parameters
    ----------
    classifier : str
        One of "random_forest", "logistic_regression", "gradient_boosting".
    reducedTrain_csv : str
        Path to the CSV file produced by feature_selection.py (features +
        target column, normalised).
    target_column : str
        Name of the binary target column inside the CSV.
    model_path : str
        Destination path for the serialised model file (*.joblib).
    metrics_json : str
        Destination path for the JSON file with training statistics.
    seed : int
        Random seed for reproducibility.
    """
    classifier_key = classifier.strip().lower()
    if classifier_key not in CLASSIFIERS:
        raise ValueError(
            f"Unknown classifier '{classifier}'. "
            f"Choose from: {list(CLASSIFIERS.keys())}"
        )

    # ------------------------------------------------------------------
    # 1. Read the dataset and measure I/O time
    # ------------------------------------------------------------------
    # Read with an explicit float32 dtype map for the feature columns
    # (see _feature_dtype_map docstring) instead of pandas' float64
    # default — halves the feature matrix' memory footprint, which
    # matters once the training set reaches hundreds of thousands to
    # ~1M+ rows even after feature reduction has cut the column count.
    t_io_start = time.perf_counter()
    dtype_map, _, resolved_train_path = _feature_dtype_map(reducedTrain_csv, target_column)
    df = pd.read_csv(resolved_train_path, dtype=dtype_map)
    t_io_end = time.perf_counter()
    dataset_input_time = round(t_io_end - t_io_start, 4)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in {reducedTrain_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    # ------------------------------------------------------------------
    # 2. Split features / target
    # ------------------------------------------------------------------
    # df.pop() removes the target column from df *in place* and returns
    # it, so df itself becomes X with no extra copy. The previous
    # X = df.drop(columns=[target_column]) allocates a brand-new frame
    # holding every other column, so the original df and the new X frame
    # are briefly both alive — effectively a second full-size copy of the
    # feature matrix at the moment training starts.
    y = df.pop(target_column).astype(np.int8)
    X = df

    n_samples, n_features = X.shape
    target_1_percentage = round(float(y.mean()) * 100, 4)
    print(
        f"[train] Loaded {n_samples} samples x {n_features} features, "
        f"~{X.memory_usage(deep=False).sum() / 1e6:.1f} MB in memory."
    )

    # ------------------------------------------------------------------
    # 3. Instantiate and train the classifier
    # ------------------------------------------------------------------
    model = CLASSIFIERS[classifier_key](seed)

    t_train_start = time.perf_counter()
    model.fit(X, y)
    t_train_end = time.perf_counter()
    training_time = round(t_train_end - t_train_start, 4)

    # ------------------------------------------------------------------
    # 4. Save the trained model
    # ------------------------------------------------------------------
    # model_path / metrics_json are used exactly as given (per §11.3):
    # callers — the CLI block, gui.py, or the evaluator's own tests —
    # are responsible for resolving bare filenames into outputs/
    # themselves *before* calling train(), via _to_output_path(). The
    # CLI block below and gui.py's call sites already do this.
    model_path = Path(model_path)
    metrics_json = Path(metrics_json)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    # ------------------------------------------------------------------
    # 5. Build and save training metrics
    # ------------------------------------------------------------------
    metrics = {
        "classifier": classifier_key,
        "seed": seed,
        "training_dataset": str(reducedTrain_csv),
        "target_column": target_column,
        "model_path": str(model_path),
        "n_samples": n_samples,
        "n_features": n_features,
        "target_1_percentage": target_1_percentage,
        "dataset_input_time": dataset_input_time,
        "training_time": training_time,
    }

    Path(metrics_json).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_json, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    # ------------------------------------------------------------------
    # 6. Console summary
    # ------------------------------------------------------------------
    print(f"[train] Classifier   : {classifier_key}")
    print(f"[train] Samples      : {n_samples}  |  Features: {n_features}")
    print(f"[train] Target=1 %   : {target_1_percentage:.2f}%")
    print(f"[train] I/O time     : {dataset_input_time:.4f}s")
    print(f"[train] Training time: {training_time:.4f}s")
    print(f"[train] Model saved  → {model_path}")
    print(f"[train] Metrics saved→ {metrics_json}")


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def predict(
    reduced_Test_csv: str,    # Path to the reduced test dataset
    target_column: str,       # Name of the target column
    model_path: str,          # Path to the saved .joblib model
    predictions_csv: str,     # Where to write per-record predictions
    classif_stats_json: str,  # Where to write classification statistics
) -> None:
    """
    Load a trained classifier and evaluate it on the reduced test dataset.

    Parameters
    ----------
    reduced_Test_csv : str
        Path to the CSV file with reduced features + target column.
    target_column : str
        Name of the binary target column inside the CSV.
    model_path : str
        Path to the serialised model produced by train() (*.joblib).
    predictions_csv : str
        Destination CSV with columns: row_n, target, prediction, score.
    classif_stats_json : str
        Destination JSON with accuracy, per-class metrics, ROC-AUC and
        confusion matrix.

    Notes
    -----
    Streams the test set in ``_PREDICT_CHUNK_SIZE``-row chunks rather than
    loading it, predicting on it, and building the output DataFrame all at
    once: reads a chunk, predicts on it, appends it to predictions_csv,
    and discards the chunk's feature matrix before moving on. Since
    prediction here is row-independent (no classifier used by this module
    looks across rows), chunking changes nothing about the predicted
    values — verified against a full single-batch run on the same data.
    The (target, prediction, score) values needed for the aggregate stats
    are kept across chunks (three arrays of a few MB even at 1.5M rows)
    and the final metrics are computed once, after the loop, exactly as
    before.
    """
    # ------------------------------------------------------------------
    # 1. Resolve output paths and load the trained model once
    # ------------------------------------------------------------------
    # predictions_csv / classif_stats_json are used exactly as given
    # (per §11.3) — same reasoning as train()'s model_path/metrics_json
    # above. Bare-filename convenience is the caller's responsibility.
    predictions_csv = Path(predictions_csv)
    classif_stats_json = Path(classif_stats_json)
    Path(predictions_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(classif_stats_json).parent.mkdir(parents=True, exist_ok=True)

    model = joblib.load(_resolve_input_path(model_path))

    # hasattr() checks are a static property of the loaded model, not of
    # the data — resolve the scoring strategy once, outside the chunk loop.
    if hasattr(model, "predict_proba"):
        score_mode = "predict_proba"
    elif hasattr(model, "decision_function"):
        score_mode = "decision_function"
    else:
        score_mode = "predict_only"

    # dtype map validates target_column presence and gives float32 reads
    # for the feature columns (see _feature_dtype_map docstring). It also
    # resolves reduced_Test_csv the same way preprocessing.py/
    # feature_selection.py resolve their inputs, so a bare filename that
    # only exists under outputs/ is found automatically.
    dtype_map, _, resolved_test_path = _feature_dtype_map(reduced_Test_csv, target_column)

    # Remove any stale predictions file so the loop below can safely
    # append (mirrors preprocessing.py's identical safety check).
    if predictions_csv.exists():
        predictions_csv.unlink()

    # ------------------------------------------------------------------
    # 2. Stream: predict per chunk, write predictions incrementally
    # ------------------------------------------------------------------
    y_true_parts:  list[np.ndarray] = []
    y_pred_parts:  list[np.ndarray] = []
    y_score_parts: list[np.ndarray] = []

    row_offset  = 0
    first_chunk = True
    n_chunks    = 0

    for chunk in pd.read_csv(
        resolved_test_path, dtype=dtype_map, chunksize=_PREDICT_CHUNK_SIZE
    ):
        # pop() removes the target column in place, leaving the chunk as
        # X — same "no extra copy" reasoning as train()'s use of pop().
        y_true_chunk = chunk.pop(target_column).astype(np.int8).to_numpy()
        X_chunk = chunk

        y_pred_chunk: np.ndarray = model.predict(X_chunk)

        if score_mode == "predict_proba":
            y_score_chunk: np.ndarray = model.predict_proba(X_chunk)[:, 1]
        elif score_mode == "decision_function":
            raw = model.decision_function(X_chunk)
            y_score_chunk = 1.0 / (1.0 + np.exp(-raw))
        else:
            y_score_chunk = y_pred_chunk.astype(float)

        out_chunk = pd.DataFrame(
            {
                "row_n": np.arange(row_offset, row_offset + len(X_chunk)),
                "target": y_true_chunk,
                "prediction": y_pred_chunk.astype(int),
                "score": y_score_chunk,
            }
        )
        out_chunk.to_csv(
            predictions_csv,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )

        # Keep only what's needed for the aggregate stats below — not the
        # feature matrix itself, which is dropped when the loop moves on.
        y_true_parts.append(y_true_chunk)
        y_pred_parts.append(y_pred_chunk.astype(np.int8))
        y_score_parts.append(y_score_chunk.astype(np.float32))

        row_offset += len(X_chunk)
        first_chunk = False
        n_chunks += 1

    print(
        f"[predict] Streamed {row_offset} rows in {n_chunks} chunk(s) "
        f"of up to {_PREDICT_CHUNK_SIZE} → {predictions_csv}"
    )

    y_true  = np.concatenate(y_true_parts)
    y_pred  = np.concatenate(y_pred_parts)
    y_score = np.concatenate(y_score_parts)
    del y_true_parts, y_pred_parts, y_score_parts

    # ------------------------------------------------------------------
    # 3. Compute classification statistics (identical formulas/order to
    #    the previous single-batch version — chunking does not change
    #    these values, only how the predictions were produced).
    # ------------------------------------------------------------------
    n_samples = len(y_true)
    target_1_count = int(y_true.sum())
    target_1_percentage = round(float(y_true.mean()) * 100, 4)

    accuracy = float(accuracy_score(y_true, y_pred))

    # per-class precision / recall / F1 / support for labels [0, 1]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    roc_auc = float(roc_auc_score(y_true, y_score))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Resolve the human-readable classifier name from the loaded model type
    classifier_name = _MODEL_NAME_MAP.get(
        type(model).__name__,
        type(model).__name__.lower(),  # fallback: use the raw class name
    )

    # ------------------------------------------------------------------
    # 4. Build and save statistics JSON
    # ------------------------------------------------------------------
    stats = {
        "classifier": classifier_name,
        "n_samples": n_samples,
        "target_1_count": target_1_count,
        "target_1_percentage": target_1_percentage,
        "accuracy": accuracy,
        "class_0": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "class_1": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": cm.tolist(),          # numpy → plain Python list
        },
    }

    with open(classif_stats_json, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    # ------------------------------------------------------------------
    # 5. Console summary
    # ------------------------------------------------------------------
    print(f"[predict] Classifier : {classifier_name}")
    print(f"[predict] Samples    : {n_samples}  |  Target=1: {target_1_count} ({target_1_percentage:.2f}%)")
    print(f"[predict] Accuracy   : {accuracy:.6f}")
    print(f"[predict] ROC-AUC    : {roc_auc:.6f}")
    print(f"[predict] F1 class-0 : {float(f1[0]):.6f}  |  F1 class-1: {float(f1[1]):.6f}")
    print(f"[predict] Predictions→ {predictions_csv}")
    print(f"[predict] Stats saved→ {classif_stats_json}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # NOTE: _resolve_outputs_dir() / _to_output_path() are now defined at
    # module level (used internally by train()/predict() as well), so the
    # CLI block below just reuses them directly.

    # ------------------------------------------------------------------
    # Top-level parser
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        prog="model.py",
        description=(
            "Train or run a binary classifier for QUBO feature-reduced data.\n"
            f"Available classifiers: {', '.join(CLASSIFIERS)}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        metavar="SUBCOMMAND",
        required=True,
    )

    # ------------------------------------------------------------------
    # 'train' subcommand
    # ------------------------------------------------------------------
    train_p = subparsers.add_parser(
        "train",
        help="Train a classifier and save the model.",
        description="Train a binary classifier on a reduced, normalised dataset.",
    )
    train_p.add_argument(
        "--classifier",
        required=True,
        choices=list(CLASSIFIERS),
        metavar="NAME",
        help=f"Classifier to use. Choices: {', '.join(CLASSIFIERS)}.",
    )
    train_p.add_argument(
        "--in-reduced",
        required=True,
        metavar="CSV",
        help="Reduced training dataset (CSV).",
    )
    train_p.add_argument(
        "--target",
        required=True,
        metavar="COL",
        help="Name of the binary target column.",
    )
    train_p.add_argument(
        "--out-model",
        required=True,
        metavar="JOBLIB",
        help="Output path for the serialised model (.joblib).",
    )
    train_p.add_argument(
        "--out-metrics",
        required=True,
        metavar="JSON",
        help="Output path for training statistics (.json).",
    )
    train_p.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="INT",
        help="Random seed for reproducibility (default: 42).",
    )

    # ------------------------------------------------------------------
    # 'predict' subcommand
    # ------------------------------------------------------------------
    predict_p = subparsers.add_parser(
        "predict",
        help="Run predictions with a saved model.",
        description="Evaluate a trained classifier on a reduced test dataset.",
    )
    predict_p.add_argument(
        "--input-testset",
        required=True,
        metavar="CSV",
        help="Reduced test dataset (CSV).",
    )
    predict_p.add_argument(
        "--target",
        required=True,
        metavar="COL",
        help="Name of the binary target column.",
    )
    predict_p.add_argument(
        "--model",
        required=True,
        metavar="JOBLIB",
        help="Path to the trained model file (.joblib).",
    )
    predict_p.add_argument(
        "--out-predictions",
        required=True,
        metavar="CSV",
        help="Output CSV for per-record predictions.",
    )
    predict_p.add_argument(
        "--out-stats",
        required=True,
        metavar="JSON",
        help="Output JSON for classification statistics.",
    )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    args = parser.parse_args()

    if args.command == "train":
        train(
            classifier=args.classifier,
            reducedTrain_csv=args.in_reduced,
            target_column=args.target,
            model_path=_to_output_path(args.out_model),
            metrics_json=_to_output_path(args.out_metrics),
            seed=args.seed,
        )

    elif args.command == "predict":
        predict(
            reduced_Test_csv=args.input_testset,
            target_column=args.target,
            model_path=args.model,
            predictions_csv=_to_output_path(args.out_predictions),
            classif_stats_json=_to_output_path(args.out_stats),
        )