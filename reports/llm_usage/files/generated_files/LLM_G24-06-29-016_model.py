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

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


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