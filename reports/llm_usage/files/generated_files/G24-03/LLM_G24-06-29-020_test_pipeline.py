"""
tests/test_pipeline.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 of 2 - Fixtures, Preprocessing, and Feature Selection.

Spec cross-reference
    §7    Preprocessing  (numeric output, NaN handling, sparse-column removal)
    §7.2  Normalisation  (z-score: mean ≈ 0, std ≈ 1)
    §7.3  Train/test split (clean cut, correct sizes)
    §8    QUBO feature selection (binary vector, ~20% selection rate)
    §11.1 Mandatory Python interface: fit_normalize()
    §11.2 Mandatory Python interface: select_features()
    §13   Automatic tests with sample_test_dataset.csv

All tests are self-contained: they create their own temporary CSV via
session-scoped pytest fixtures, so no external data file is required to
run the suite.  The produced sample_test_dataset.csv is also written to
data/ so that it is available for the evaluator (see §13).

Usage
-----
    # from the project root:
    pytest tests/test_pipeline.py -v

    # or run the full suite:
    pytest
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────────────
# Ensures `from qubo_project.X import Y` works regardless of the CWD from
# which pytest is invoked (project root, tests/, etc.).
# Tip: moving this block to a top-level conftest.py is even cleaner, because
# it applies to every test file automatically.  For a single-file suite this
# in-file approach is perfectly portable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from qubo_project.preprocessing import fit_normalize          # noqa: E402
from qubo_project.feature_selection import select_features    # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Dataset and pipeline constants shared by every fixture and test
# ─────────────────────────────────────────────────────────────────────────────
TARGET        = "target"     # name of the binary label column
N_SAMPLES     = 120          # rows  – small enough for fast QUBO, large enough for stats
N_FEATURES    = 20           # cols  – round(0.20 * 20) = K=4; with allowance=1 → [3, 5]
N_SPARSE_COLS = 3            # columns whose ≥ 98% values are zero → must be dropped
NAN_FRACTION  = 0.05         # random NaN rate in the remaining feature cells
POS_FRACTION  = 0.20         # fraction of target=1 (well above the 10% floor in §13)
SEED          = 42


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  F I X T U R E S
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture(scope="session")
def session_dir(tmp_path_factory):
    """
    Single temporary directory shared across the entire test session.
    Using session scope means fit_normalize() and select_features() each
    run exactly once no matter how many tests consume their outputs.
    """
    return tmp_path_factory.mktemp("qubo_pipeline")


@pytest.fixture(scope="session")
def raw_csv(session_dir) -> Path:
    """
    Builds a fully deterministic dummy CSV that mimics the real dataset (§6).

    ┌───────────────────────────────────────────────────────────────────────┐
    │  Property                          │  Value                           │
    ├───────────────────────────────────────────────────────────────────────┤
    │  Rows                              │  N_SAMPLES  (120)                │
    │  Feature columns                   │  N_FEATURES (20), all float64    │
    │  Sparse cols (≥ 98 % zeros)        │  N_SPARSE_COLS (3) – must drop   │
    │  Random NaN in non-sparse cols     │  NAN_FRACTION (~5 %) of cells    │
    │  Binary target, both classes       │  POS_FRACTION (20 %) positive    │
    └───────────────────────────────────────────────────────────────────────┘

    Returns
    -------
    pathlib.Path
        Absolute path to the written CSV file.
    """
    rng = np.random.default_rng(SEED)

    # ── feature matrix: standard-normal values ───────────────────────────
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))

    # ── inject NaN into the non-sparse block (columns N_SPARSE_COLS … end)
    nan_mask = rng.random((N_SAMPLES, N_FEATURES - N_SPARSE_COLS)) < NAN_FRACTION
    X[:, N_SPARSE_COLS:][nan_mask] = np.nan

    # ── make the first N_SPARSE_COLS columns near-all-zero (≥ 98 % zeros)
    #    → only 2 non-zero values per column → 98.3 % zeros → exceeds 95 %
    #    threshold used in fit_normalize(minPercValid=0.05)
    for col_idx in range(N_SPARSE_COLS):
        col_data = np.zeros(N_SAMPLES)
        nz_idx = rng.choice(N_SAMPLES, size=2, replace=False)
        col_data[nz_idx] = rng.standard_normal(2)
        X[:, col_idx] = col_data

    # ── binary target: guarantee ≥ 10 % representation for each class ────
    n_pos = max(int(N_SAMPLES * POS_FRACTION), int(N_SAMPLES * 0.10) + 1)
    y = np.zeros(N_SAMPLES, dtype=np.int8)
    pos_idx = rng.choice(N_SAMPLES, size=n_pos, replace=False)
    y[pos_idx] = 1

    # Fixture-level invariant assertions (not test assertions – these are
    # guarantees about the fixture itself, so they use plain assert).
    assert y.sum() >= 0.10 * N_SAMPLES, "Fixture: fewer than 10 % positive samples"
    assert (1 - y).sum() >= 0.10 * N_SAMPLES, "Fixture: fewer than 10 % negative samples"

    # ── assemble DataFrame and persist to the session temp directory ──────
    col_names = [f"feature_{i:02d}" for i in range(N_FEATURES)]
    df = pd.DataFrame(X, columns=col_names)
    df[TARGET] = y.astype(int)

    csv_path = session_dir / "sample_test_dataset.csv"
    df.to_csv(csv_path, index=False)

    # Also write a copy to data/ so the evaluator finds it (§13)
    data_dir = _PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    df.to_csv(data_dir / "sample_test_dataset.csv", index=False)

    return csv_path


@pytest.fixture(scope="session")
def preproc(raw_csv, session_dir) -> dict[str, str]:
    """
    Runs fit_normalize() exactly once per session and exposes its output
    paths so every test that needs the normalised data can reuse them
    without re-running the (potentially slow) preprocessing step.

    Returns
    -------
    dict with keys ``csv`` and ``json``.
    """
    norm_csv = str(session_dir / "normalized.csv")
    out_json = str(session_dir / "preprocessing_stats.json")

    fit_normalize(
        input_csv         = str(raw_csv),
        target_column     = TARGET,
        normalized_csv    = norm_csv,
        outInitalRes_json = out_json,  # name from spec §11.1 (intentional typo)
        minPercValid      = 0.05,      # drop columns with < 5 % valid non-zero data
    )
    return {"csv": norm_csv, "json": out_json}


@pytest.fixture(scope="session")
def feat_sel(preproc, session_dir) -> dict[str, str]:
    """
    Runs select_features() exactly once per session on the normalised CSV
    produced by ``preproc``.

    Returns
    -------
    dict with keys ``train_csv``, ``test_csv``, ``optim_csv``, ``json``.
    """
    out = {
        "train_csv": str(session_dir / "train_reduced.csv"),
        "test_csv":  str(session_dir / "test_reduced.csv"),
        "optim_csv": str(session_dir / "optimizations.csv"),
        "json":      str(session_dir / "feature_selection_stats.json"),
    }

    select_features(
        normalized_csv    = preproc["csv"],
        reducedTrain_csv  = out["train_csv"],
        reducedTest_csv   = out["test_csv"],
        output_ottim_csv  = out["optim_csv"],
        output_json       = out["json"],
        target_column     = TARGET,
        percTest          = 0.30,
        percSelected      = 0.20,
        allowance         = 1,
        seed              = SEED,
        alpha_computations = 20,   # limited for speed; still exercises the search loop
    )
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  P R E P R O C E S S I N G   T E S T S   (spec §7, §11.1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPreprocessing:
    """
    Validates fit_normalize() against spec §7 (preprocessing) and §11.1
    (mandatory Python interface and JSON output schema).
    """

    @pytest.fixture(autouse=True)
    def _setup(self, preproc):
        """Load the normalised CSV once per test method in this class."""
        self.norm_csv  = preproc["csv"]
        self.json_path = preproc["json"]
        self.df        = pd.read_csv(preproc["csv"])
        self.features  = self.df.drop(columns=[TARGET])

    # ── output-file existence ─────────────────────────────────────────────

    def test_output_csv_is_created(self):
        """fit_normalize() must write the normalised CSV to disk (§11.1)."""
        assert Path(self.norm_csv).exists(), (
            "fit_normalize() did not create the normalised CSV file."
        )

    def test_output_json_is_created(self):
        """fit_normalize() must write the JSON statistics file to disk (§11.1)."""
        assert Path(self.json_path).exists(), (
            "fit_normalize() did not create the JSON statistics file."
        )

    # ── numeric-only output (spec §7, requirement 4) ─────────────────────

    def test_all_feature_columns_are_numeric(self):
        """
        Every non-target column in the output must have a numeric dtype
        (int or float).  Non-numeric columns cannot be fed to the QUBO
        matrix builder or to any scikit-learn classifier.
        """
        bad_cols = [
            c for c in self.features.columns
            if not pd.api.types.is_numeric_dtype(self.features[c])
        ]
        assert not bad_cols, (
            f"Non-numeric feature column(s) remain after preprocessing: {bad_cols}"
        )

    # ── NaN handling (spec §7.1) ──────────────────────────────────────────

    def test_no_nan_values_in_output_features(self):
        """
        All NaN / missing values present in the raw input must be resolved
        (imputed or removed) before the CSV is written.  The downstream QUBO
        and classifier stages assume a clean, fully numeric matrix.
        """
        n_nan = int(self.features.isna().sum().sum())
        assert n_nan == 0, (
            f"Normalised CSV still contains {n_nan} NaN value(s) across "
            "feature columns."
        )

    # ── sparse-column elimination (spec §7.1) ────────────────────────────

    def test_near_zero_columns_are_removed(self, raw_csv):
        """
        Columns with ≥ 95 % zero/NaN values in the raw input must be absent
        from the normalised output.  The fixture deliberately places
        N_SPARSE_COLS (3) such columns at the start of the feature matrix.
        """
        raw_df = pd.read_csv(raw_csv)
        n_raw_features  = sum(1 for c in raw_df.columns  if c != TARGET)
        n_norm_features = len(self.features.columns)

        assert n_norm_features < n_raw_features, (
            "No columns were removed despite at least 3 having ≥ 98 % zeros."
        )

        sparse_names = [f"feature_{i:02d}" for i in range(N_SPARSE_COLS)]
        survivors = [c for c in sparse_names if c in self.features.columns]
        assert not survivors, (
            f"Sparse column(s) that should have been dropped are still present: "
            f"{survivors}"
        )

    # ── target-column integrity ───────────────────────────────────────────

    def test_target_column_survives_and_remains_binary(self):
        """
        The target column must be present in the output and retain only
        values in {0, 1}.  Any mutation would corrupt train/test labels.
        """
        assert TARGET in self.df.columns, (
            f"Target column '{TARGET}' is missing from the normalised CSV."
        )
        unique_vals = set(self.df[TARGET].unique())
        assert unique_vals.issubset({0, 1}), (
            f"Target column contains non-binary values: {unique_vals}"
        )

    # ── JSON schema (spec §11.1) ──────────────────────────────────────────

    def test_json_contains_all_required_keys(self):
        """
        The preprocessing JSON must contain every key specified in §11.1.
        The evaluator's parser accesses these keys by name; missing keys
        will raise KeyError at evaluation time.
        """
        required = {
            "n_input_features",
            "n_kept_features",
            "dataset_size",
            "dataset_input_time",
            "dataset_processing_time",
            "dropped_feature_names",
        }
        with open(self.json_path) as fh:
            data = json.load(fh)

        missing = required - data.keys()
        assert not missing, (
            f"Preprocessing JSON is missing {len(missing)} required key(s): "
            f"{missing}"
        )

    def test_json_n_kept_features_matches_csv_column_count(self):
        """
        n_kept_features in the JSON must exactly equal the number of feature
        columns written to the normalised CSV.  Discrepancy confuses the
        evaluator script that reads both artefacts.
        """
        with open(self.json_path) as fh:
            meta = json.load(fh)
        actual = len(self.features.columns)
        assert meta["n_kept_features"] == actual, (
            f"JSON n_kept_features={meta['n_kept_features']} but the normalised "
            f"CSV has {actual} feature column(s)."
        )

    def test_json_dataset_size_matches_row_count(self):
        """dataset_size in JSON must equal the number of rows in the CSV."""
        with open(self.json_path) as fh:
            meta = json.load(fh)
        assert meta["dataset_size"] == len(self.df), (
            f"JSON dataset_size={meta['dataset_size']} but CSV has "
            f"{len(self.df)} row(s)."
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  N O R M A L I S A T I O N   T E S T S   (spec §7.2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestNormalization:
    """
    Verifies z-score normalisation correctness (spec §7.2).

    After standardisation every feature column f must satisfy:

        |mean(f)|           < _MEAN_TOL   ← centering  was applied
        |std(f, ddof=0) − 1| < _STD_TOL   ← scaling    was applied

    Tolerances are deliberately asymmetric: the mean constraint is very
    tight (near machine-epsilon), while the std constraint is generous
    enough to accommodate both population std (ddof=0, used by sklearn)
    and sample std (ddof=1), as well as the slight variance shift caused
    by NaN imputation before scaling.
    """

    _MEAN_TOL = 1e-9   # any reasonable z-score implementation will hit this
    _STD_TOL  = 0.05   # accommodates ddof=0 vs ddof=1 and NaN-imputation effects

    @pytest.fixture(autouse=True)
    def _setup(self, preproc, raw_csv):
        df = pd.read_csv(preproc["csv"])
        self.features  = df.drop(columns=[TARGET])
        self.raw_rows  = pd.read_csv(raw_csv).shape[0]
        self.norm_rows = df.shape[0]

    # ── centering ─────────────────────────────────────────────────────────

    def test_feature_means_are_zero(self):
        """
        After z-score normalisation every column mean must be ≈ 0.
        A non-zero mean indicates the centering step was skipped or the
        wrong µ was used for subtraction.
        """
        violators = {
            col: float(self.features[col].mean())
            for col in self.features.columns
            if abs(self.features[col].mean()) > self._MEAN_TOL
        }
        assert not violators, (
            f"{len(violators)} column(s) have |mean| > {self._MEAN_TOL}:\n"
            + "\n".join(f"  {c}: mean = {v:.3e}" for c, v in violators.items())
        )

    # ── scaling ───────────────────────────────────────────────────────────

    def test_feature_stds_are_one(self):
        """
        After z-score normalisation every column std must be ≈ 1.
        Large deviation indicates that the scaling step was skipped, that
        min-max normalisation was used instead, or that the wrong σ was
        used for division.
        """
        violators = {
            col: float(self.features[col].std(ddof=0))
            for col in self.features.columns
            if abs(self.features[col].std(ddof=0) - 1.0) > self._STD_TOL
        }
        assert not violators, (
            f"{len(violators)} column(s) have |std(ddof=0) − 1| > {self._STD_TOL}:\n"
            + "\n".join(f"  {c}: std = {v:.4f}" for c, v in violators.items())
        )

    # ── row count ─────────────────────────────────────────────────────────

    def test_row_count_is_unchanged(self):
        """
        Normalisation is a column-wise transformation; it must NOT drop rows.
        Row loss would silently corrupt the train/test split that follows.
        """
        assert self.norm_rows == self.raw_rows, (
            f"Row count changed during preprocessing: "
            f"raw={self.raw_rows}, normalised={self.norm_rows}."
        )

    # ── finite-value guarantee ────────────────────────────────────────────

    def test_no_nan_or_inf_after_normalisation(self):
        """
        After z-score normalisation no feature cell may contain NaN, +Inf,
        or -Inf.  Such values propagate silently through the QUBO matrix
        and corrupt the Spearman correlations and the classifier fit.
        """
        n_nan = int(self.features.isna().sum().sum())

        # np.isinf raises on NaN, so replace NaN with a finite sentinel first
        arr = self.features.fillna(0).to_numpy(dtype=float)
        n_inf = int(np.isinf(arr).sum())

        errors: list[str] = []
        if n_nan:
            errors.append(f"{n_nan} NaN value(s)")
        if n_inf:
            errors.append(f"{n_inf} infinite value(s) (±Inf)")

        assert not errors, (
            "Normalised features contain non-finite values: "
            + ", ".join(errors) + "."
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  F E A T U R E   S E L E C T I O N   T E S T S   (spec §8, §11.2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestFeatureSelection:
    """
    Validates select_features() against spec §8 (QUBO formulation and
    binary vector) and §11.2 (mandatory Python interface and JSON schema).
    """

    @pytest.fixture(autouse=True)
    def _setup(self, feat_sel, preproc):
        self.paths       = feat_sel
        self.preproc_csv = preproc["csv"]

        with open(feat_sel["json"]) as fh:
            self.meta = json.load(fh)

    # ── output-file existence ─────────────────────────────────────────────

    def test_all_output_files_are_created(self):
        """
        All four output artefacts promised by spec §11.2 must exist on disk
        after select_features() returns.
        """
        for key, path in self.paths.items():
            assert Path(path).exists(), (
                f"Expected output file '{key}' was not created: {path}"
            )

    # ── binary vector correctness (spec §8.1) ────────────────────────────

    def test_selected_vector_contains_only_binary_values(self):
        """
        x* (selected_vector) must be a list of {0, 1} values (spec §8.1).
        Any other value signals an implementation error in the QUBO solver
        or in the post-processing of its output.
        """
        vec = self.meta.get("selected_vector")
        assert vec is not None, (
            "'selected_vector' key is missing from the feature-selection JSON."
        )
        assert isinstance(vec, list) and len(vec) > 0, (
            "'selected_vector' must be a non-empty list."
        )

        non_binary = [v for v in vec if v not in (0, 1)]
        assert not non_binary, (
            f"selected_vector contains {len(non_binary)} non-binary value(s): "
            + repr(non_binary[:6]) + ("…" if len(non_binary) > 6 else "")
        )

    def test_selected_vector_length_equals_features_after_preprocessing(self):
        """
        The length of x* must equal the number of features available *after*
        preprocessing, i.e. after sparse-column removal (spec §8.1:
        'La lunghezza del vettore deve essere uguale al numero di feature
        disponibili dopo il preprocessing').
        A length mismatch means QUBO consumed a different feature space than
        the one preprocessing produced.
        """
        norm_df = pd.read_csv(self.preproc_csv)
        n_after_preproc = sum(1 for c in norm_df.columns if c != TARGET)
        vec_len = len(self.meta["selected_vector"])

        assert vec_len == n_after_preproc, (
            f"selected_vector length ({vec_len}) ≠ number of features after "
            f"preprocessing ({n_after_preproc})."
        )

    def test_selected_vector_selects_at_least_one_feature(self):
        """
        An all-zero x* is degenerate: the downstream classifier would receive
        no input features and could not be trained.
        """
        vec = self.meta["selected_vector"]
        assert sum(vec) >= 1, (
            "selected_vector is all-zeros; QUBO selected no feature at all."
        )

    # ── ~20 % selection rate (spec §8) ───────────────────────────────────

    def test_k_selected_features_within_allowance_of_20_percent(self):
        """
        Core QUBO correctness test (spec §8):

            |K_actual − round(0.20 × n)| ≤ allowance

        Failure means the alpha-search loop did not converge to the requested
        selection ratio, or the QUBO objective is not correctly formulated.
        """
        vec       = self.meta["selected_vector"]
        allowance = self.meta.get("allowance", 1)
        n_total   = len(vec)
        k_target  = round(0.20 * n_total)
        k_actual  = sum(vec)

        assert abs(k_actual - k_target) <= allowance, (
            f"QUBO selected K={k_actual} feature(s); "
            f"expected K ∈ [{k_target - allowance}, {k_target + allowance}] "
            f"(n={n_total}, target ratio=20 %, allowance={allowance})."
        )

    # ── reduced-CSV structure (spec §8.4) ─────────────────────────────────

    def test_reduced_train_csv_has_correct_structure(self):
        """
        Reduced training CSV must contain exactly K feature columns plus the
        target column (spec §8.4: 'training set ridotto con K colonne,
        più la colonna target').
        """
        df       = pd.read_csv(self.paths["train_csv"])
        k_actual = sum(self.meta["selected_vector"])
        feat_cols = [c for c in df.columns if c != TARGET]

        assert TARGET in df.columns, (
            "Target column is missing from the reduced training CSV."
        )
        assert len(feat_cols) == k_actual, (
            f"Reduced training CSV has {len(feat_cols)} feature column(s); "
            f"expected {k_actual}."
        )

    def test_reduced_test_csv_has_correct_structure(self):
        """
        Reduced test CSV must mirror the training CSV structure (same K
        feature columns + target).  A column mismatch causes predict() to
        fail or silently produce wrong results.
        """
        df       = pd.read_csv(self.paths["test_csv"])
        k_actual = sum(self.meta["selected_vector"])
        feat_cols = [c for c in df.columns if c != TARGET]

        assert TARGET in df.columns, (
            "Target column is missing from the reduced test CSV."
        )
        assert len(feat_cols) == k_actual, (
            f"Reduced test CSV has {len(feat_cols)} feature column(s); "
            f"expected {k_actual}."
        )

    def test_train_and_test_csvs_have_identical_columns(self):
        """
        The exact same feature columns, in the same order, must appear in
        both the training and the test CSVs.  Any discrepancy means the
        model is evaluated on a different feature space than it was trained
        on – a silent data-leakage / alignment bug.
        """
        train_cols = list(pd.read_csv(self.paths["train_csv"]).columns)
        test_cols  = list(pd.read_csv(self.paths["test_csv"]).columns)

        assert train_cols == test_cols, (
            "Column mismatch between reduced train and test CSVs:\n"
            f"  train: {train_cols}\n"
            f"  test : {test_cols}"
        )

    # ── train/test split sizes (spec §7.3) ───────────────────────────────

    def test_train_test_split_is_exhaustive_and_sizes_are_correct(self):
        """
        spec §7.3 mandates a 'taglio netto' (clean cut): every normalised
        row lands in exactly one of the two sets, and the test fraction
        must match percTest=0.30 to within ±1 row of integer rounding.
        """
        n_norm  = pd.read_csv(self.preproc_csv).shape[0]
        n_train = pd.read_csv(self.paths["train_csv"]).shape[0]
        n_test  = pd.read_csv(self.paths["test_csv"]).shape[0]

        assert n_train + n_test == n_norm, (
            f"train ({n_train}) + test ({n_test}) ≠ total normalised rows "
            f"({n_norm}).  Rows were lost or duplicated during the split."
        )

        expected_test = round(n_norm * 0.30)
        assert abs(n_test - expected_test) <= 1, (
            f"Test set has {n_test} rows; expected ~{expected_test} "
            f"(percTest=0.30, n={n_norm})."
        )

    # ── JSON–CSV consistency ──────────────────────────────────────────────

    def test_selected_feature_names_in_json_match_csv_columns(self):
        """
        selected_feature_names in the JSON must exactly match (sorted) the
        feature columns of the reduced training CSV.  Inconsistency breaks
        any code that aligns features by name (e.g. inference on new data).
        """
        train_df = pd.read_csv(self.paths["train_csv"])
        csv_feat  = sorted(c for c in train_df.columns if c != TARGET)
        json_feat = sorted(self.meta.get("selected_feature_names", []))

        assert csv_feat == json_feat, (
            "Mismatch between selected_feature_names (JSON) and CSV columns:\n"
            f"  In JSON but not CSV : {sorted(set(json_feat) - set(csv_feat))}\n"
            f"  In CSV but not JSON : {sorted(set(csv_feat)  - set(json_feat))}"
        )

    # ── JSON schema (spec §11.2) ──────────────────────────────────────────

    def test_json_contains_all_required_keys(self):
        """
        The feature-selection JSON must contain every key listed in §11.2.
        The evaluator reads all of these by name.
        """
        required = {
            "n_features", "target_ratio", "target_k", "allowance",
            "n_selected", "alpha", "selected_vector", "selected_feature_names",
            "algorithm", "seed", "alpha_computations", "percTest",
            "training_dataset_size", "test_dataset_size",
            "q_matrix_creation_time", "mean_optimization_time",
            "std_dev_optimization_time",
        }
        missing = required - self.meta.keys()
        assert not missing, (
            f"Feature-selection JSON is missing {len(missing)} required key(s): "
            f"{missing}"
        )

    # ── optimisations CSV (spec §11.2) ────────────────────────────────────

    def test_optimizations_csv_is_non_empty_and_fully_numeric(self):
        """
        The per-alpha optimisation log (spec §11.2) must satisfy:
        • ≥ 1 data row  (at least one optimisation was performed)
        • ≥ 4 columns   (alpha, time, n_features, cost_value)
        • all numeric   (the evaluator plots these columns directly)
        """
        df = pd.read_csv(self.paths["optim_csv"])

        assert len(df) >= 1, "Optimisations CSV has no data rows."

        assert df.shape[1] >= 4, (
            f"Optimisations CSV has only {df.shape[1]} column(s); "
            "expected ≥ 4 (alpha, time, n_features, cost_value)."
        )

        bad_cols = [
            c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c])
        ]
        assert not bad_cols, (
            f"Non-numeric column(s) in optimisations CSV: {bad_cols}"
        )