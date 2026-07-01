"""
model.py — Binary classifier training and prediction module.

Supported classifiers:
    - "random_forest"       : RandomForestClassifier
    - "logistic_regression" : LogisticRegression
    - "gradient_boosting"   : GradientBoostingClassifier
"""

import json
import time
import argparse
from pathlib import Path

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
# IMPORTANT: these are used by train()/predict() themselves (not just the
# CLI), so that callers — including gui.py, which imports and calls these
# functions directly as a library rather than going through the CLI — always
# get their output files written into the project's outputs/ directory,
# regardless of the process's current working directory.
#
# This mirrors the identical helpers in preprocessing.py and
# feature_selection.py. Without this, a bare filename like "model.joblib"
# resolves relative to os.getcwd() (wherever the GUI happened to be
# launched from) instead of <repo_root>/outputs/, which can silently leave
# a stale outputs/model.joblib in place from an earlier run while
# session_state.model_path keeps pointing at it — leading to predictions
# being run against the wrong (stale) model.

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

    Per §12, the CSV inputs to this module (``--in-reduced`` for train,
    ``--input-testset`` for predict) are files that feature_selection.py
    just wrote into ``outputs/``. To make the exact CLI invocations in
    the spec work regardless of the caller's current working directory,
    this tries, in order:

    1. *user_path* exactly as given (absolute, or relative to cwd).
    2. ``outputs/<basename of user_path>`` — matching where the prior
       pipeline stage would have written it.

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
    reducedTrain_csv = _resolve_input_path(reducedTrain_csv)
    t_io_start = time.perf_counter()
    df = pd.read_csv(reducedTrain_csv)
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
    X = df.drop(columns=[target_column])
    y = df[target_column]

    n_samples, n_features = X.shape
    target_1_percentage = round(float(y.mean()) * 100, 4)

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
    # Resolve into outputs/ here (not just in the CLI block) so that
    # library callers such as gui.py — which pass bare filenames like
    # "model.joblib" — don't silently write next to the process's cwd
    # while everything else (session_state, predict()) keeps assuming
    # the file lives under outputs/.
    model_path = _to_output_path(model_path)
    metrics_json = _to_output_path(metrics_json)

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
    """
    # ------------------------------------------------------------------
    # 1. Load test dataset
    # ------------------------------------------------------------------
    reduced_Test_csv = _resolve_input_path(reduced_Test_csv)
    df = pd.read_csv(reduced_Test_csv)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in {reduced_Test_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    X_test = df.drop(columns=[target_column])
    y_true = df[target_column].astype(int)

    # ------------------------------------------------------------------
    # 2. Load the trained model
    # ------------------------------------------------------------------
    model_path = _resolve_input_path(model_path)
    model = joblib.load(model_path)

    # ------------------------------------------------------------------
    # 3. Generate predictions and probability scores
    # ------------------------------------------------------------------
    y_pred: np.ndarray = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        # probability of the positive class (index 1)
        y_score: np.ndarray = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        # sigmoid-normalise raw decision scores to [0, 1]
        raw = model.decision_function(X_test)
        y_score = 1.0 / (1.0 + np.exp(-raw))
    else:
        y_score = y_pred.astype(float)

    # ------------------------------------------------------------------
    # 4. Save per-record predictions CSV
    # ------------------------------------------------------------------
    # Same outputs/ resolution as train() — see comment there. Without
    # this, a bare "predictions.csv" from gui.py lands next to the
    # process's cwd instead of outputs/, while the GUI reads back from
    # outputs/ and shows stale results from an earlier run.
    predictions_csv = _to_output_path(predictions_csv)
    classif_stats_json = _to_output_path(classif_stats_json)

    predictions_df = pd.DataFrame(
        {
            "row_n": range(len(y_true)),
            "target": y_true.values,
            "prediction": y_pred.astype(int),
            "score": y_score,
        }
    )
    Path(predictions_csv).parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(predictions_csv, index=False)

    # ------------------------------------------------------------------
    # 5. Compute classification statistics
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
    # 6. Build and save statistics JSON
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

    Path(classif_stats_json).parent.mkdir(parents=True, exist_ok=True)
    with open(classif_stats_json, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    # ------------------------------------------------------------------
    # 7. Console summary
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