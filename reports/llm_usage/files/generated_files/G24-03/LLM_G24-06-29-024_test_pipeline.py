"""
tests/test_pipeline.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Complete test suite – Preprocessing, Feature Selection, and ML Model.

Spec cross-reference
    §7    Preprocessing  (numeric output, NaN handling, sparse-column removal)
    §7.2  Normalisation  (z-score: mean ≈ 0, std ≈ 1)
    §7.3  Train/test split (clean cut, correct sizes)
    §8    QUBO feature selection (binary vector, ~20% selection rate)
    §9    Three classifiers including mandatory Random Forest
    §10   Prediction CSV format and classification statistics
    §11.1 Mandatory Python interface: fit_normalize()
    §11.2 Mandatory Python interface: select_features()
    §11.3 Mandatory Python interface: train() and predict()
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

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import is_classifier

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
from qubo_project.model import train, predict                 # noqa: E402

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

# Unique filename prefix used for every test-generated output file.
# This avoids silent collisions with production runs that write their
# own files (e.g. "normalized.csv") into the same outputs/ directory.
_TEST_PREFIX = "test_pipeline_"


# ─────────────────────────────────────────────────────────────────────────────
# Output-directory helper
# ─────────────────────────────────────────────────────────────────────────────

def _project_outputs_dir() -> Path:
    """Return the project's ``outputs/`` directory, creating it if absent.

    Both ``preprocessing.py`` and ``feature_selection.py`` contain a private
    helper ``_to_output_path()`` that **strips every directory component** from
    any path string passed to it and always writes inside the project's own
    ``outputs/`` directory:

        Path(user_path).name  →  bare filename only
        return outputs_dir / filename

    That means a full ``tmp_path`` absolute path like
    ``/tmp/pytest-xxx/qubo_pipeline0/normalized.csv``
    is silently reduced to ``normalized.csv`` and written to
    ``<project_root>/outputs/normalized.csv`` instead.

    The implementation resolves ``outputs/`` by walking upward from the
    source package (``src/qubo_project/``) until it finds a directory that
    already contains an ``outputs/`` subdirectory, then falls back to
    ``Path.cwd() / "outputs"``.

    This helper replicates that walk so fixtures can predict the exact
    destination without importing private implementation symbols.
    """
    # Walk upward from the source package — mirrors _resolve_outputs_dir()
    candidate = (_PROJECT_ROOT / "src" / "qubo_project").resolve()
    for _ in range(8):
        probe = candidate / "outputs"
        if probe.is_dir():
            return probe
        candidate = candidate.parent

    # Fallback: create outputs/ at the project root.
    # Equivalent to the implementation's `Path.cwd() / "outputs"` when pytest
    # is invoked from the project root (the standard way).
    fallback = _PROJECT_ROOT / "outputs"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


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
def preproc(raw_csv) -> dict[str, str]:
    """
    Runs fit_normalize() exactly once per session and exposes its output
    paths so every test that needs the normalised data can reuse them
    without re-running the (potentially slow) preprocessing step.

    Path contract
    -------------
    fit_normalize() routes every output parameter through its internal
    ``_to_output_path()`` helper, which calls ``Path(user_path).name`` and
    then prepends the project's ``outputs/`` directory.  Passing a full
    ``tmp_path`` absolute path would therefore be silently reduced to just
    the basename and written somewhere we cannot predict from the fixture.

    Fix: pass bare filenames prefixed with ``_TEST_PREFIX`` so the
    resulting path is deterministic and does not collide with production
    output files.  Compute the actual destination via ``_project_outputs_dir()``
    so the returned dict always points to the file that was really written.

    Returns
    -------
    dict with keys ``csv`` and ``json``.
    """
    out_dir       = _project_outputs_dir()
    norm_csv_name = _TEST_PREFIX + "normalized.csv"
    out_json_name = _TEST_PREFIX + "preprocessing_stats.json"

    fit_normalize(
        input_csv         = str(raw_csv),
        target_column     = TARGET,
        normalized_csv    = norm_csv_name,   # bare filename — _to_output_path keeps only .name
        outInitalRes_json = out_json_name,   # same rule; see §11.1 for the typo in param name
        minPercValid      = 0.05,
    )
    return {
        "csv":  str(out_dir / norm_csv_name),
        "json": str(out_dir / out_json_name),
    }


@pytest.fixture(scope="session")
def feat_sel(preproc) -> dict[str, str]:
    """
    Runs select_features() exactly once per session on the normalised CSV
    produced by ``preproc``.

    Path contract
    -------------
    select_features() applies the same ``_to_output_path()`` logic as
    fit_normalize() to every *output* parameter, stripping directory
    components and always writing into ``outputs/``.

    However, the *input* parameter ``normalized_csv`` is read directly with
    ``pd.read_csv()`` — no remapping occurs.  We therefore:

    • Pass the full resolved path from ``preproc["csv"]`` as the input so
      select_features() can find the file that fit_normalize() wrote.
    • Pass bare filenames (prefixed with ``_TEST_PREFIX``) for all four
      output parameters so the writes land at predictable locations.
    • Return paths built from ``_project_outputs_dir()`` so the dict always
      reflects where the files were actually written.

    Returns
    -------
    dict with keys ``train_csv``, ``test_csv``, ``optim_csv``, ``json``.
    """
    out_dir    = _project_outputs_dir()
    train_name = _TEST_PREFIX + "train_reduced.csv"
    test_name  = _TEST_PREFIX + "test_reduced.csv"
    optim_name = _TEST_PREFIX + "optimizations.csv"
    json_name  = _TEST_PREFIX + "feature_selection_stats.json"

    select_features(
        normalized_csv    = preproc["csv"],   # full path — inputs are NOT remapped
        reducedTrain_csv  = train_name,        # bare filename — output is remapped
        reducedTest_csv   = test_name,
        output_ottim_csv  = optim_name,
        output_json       = json_name,
        target_column     = TARGET,
        percTest          = 0.30,
        percSelected      = 0.20,
        allowance         = 1,
        seed              = SEED,
        alpha_computations = 20,   # limited for speed; still exercises the alpha search loop
    )
    return {
        "train_csv": str(out_dir / train_name),
        "test_csv":  str(out_dir / test_name),
        "optim_csv": str(out_dir / optim_name),
        "json":      str(out_dir / json_name),
    }


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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  A D D I T I O N A L   F I X T U R E S   (model training and prediction)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Classifier names that every group MUST support (§9: Random Forest is
# mandatory; the remaining two are implementation-defined).
_MANDATORY_CLF = "random_forest"

# Common alternatives tried by the multi-classifier parametrized test.
# Tests are automatically skipped for names not implemented by the group.
_CANDIDATE_CLFS = [
    "logistic_regression",
    "decision_tree",
    "svm",
    "gradient_boosting",
    "knn",
]


@pytest.fixture(scope="session")
def model_outputs(feat_sel, session_dir) -> dict[str, str]:
    """
    Trains a Random Forest classifier exactly once per session using the
    reduced training CSV produced by ``feat_sel``.

    Returns
    -------
    dict with keys ``model_path`` and ``metrics_json``.
    """
    out = {
        "model_path":    str(session_dir / "model.joblib"),
        "metrics_json":  str(session_dir / "training_metrics.json"),
    }
    train(
        classifier       = _MANDATORY_CLF,
        reducedTrain_csv = feat_sel["train_csv"],
        target_column    = TARGET,
        model_path       = out["model_path"],
        metrics_json     = out["metrics_json"],
        seed             = SEED,
    )
    return out


@pytest.fixture(scope="session")
def prediction_outputs(feat_sel, model_outputs, session_dir) -> dict[str, str]:
    """
    Runs predict() exactly once per session, feeding the saved Random Forest
    model against the reduced test CSV.

    Returns
    -------
    dict with keys ``predictions_csv`` and ``stats_json``.
    """
    out = {
        "predictions_csv": str(session_dir / "predictions.csv"),
        "stats_json":      str(session_dir / "classification_stats.json"),
    }
    predict(
        reduced_Test_csv  = feat_sel["test_csv"],
        target_column     = TARGET,
        model_path        = model_outputs["model_path"],
        predictions_csv   = out["predictions_csv"],
        classif_stats_json = out["stats_json"],
    )
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  M O D E L   T R A I N I N G   T E S T S   (spec §9, §11.3 – train)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestModelTraining:
    """
    Validates train() against spec §9 (three classifiers, Random Forest
    mandatory) and §11.3 (mandatory Python interface + training JSON schema).
    """

    @pytest.fixture(autouse=True)
    def _setup(self, model_outputs):
        """Cache output paths and load the training-metrics JSON once per test."""
        self.model_path   = model_outputs["model_path"]
        self.metrics_json = model_outputs["metrics_json"]

        with open(model_outputs["metrics_json"]) as fh:
            self.metrics = json.load(fh)

    # ── output-file existence ─────────────────────────────────────────────

    def test_model_joblib_file_is_created(self):
        """train() must write the serialised model to disk as a .joblib file (§9)."""
        assert Path(self.model_path).exists(), (
            f"Model file was not created at '{self.model_path}'."
        )
        assert self.model_path.endswith(".joblib"), (
            f"Model file extension is not '.joblib': '{self.model_path}'."
        )

    def test_training_metrics_json_is_created(self):
        """train() must write the training-metrics JSON to disk (§11.3)."""
        assert Path(self.metrics_json).exists(), (
            f"Training metrics JSON was not created at '{self.metrics_json}'."
        )

    # ── model integrity ───────────────────────────────────────────────────

    def test_model_file_is_loadable_with_joblib(self):
        """
        The .joblib file must be deserialised without errors.
        A corrupt or truncated file raises an exception here, which would also
        crash the evaluator's automated scoring script.
        """
        try:
            model = joblib.load(self.model_path)
        except Exception as exc:
            pytest.fail(
                f"joblib.load('{self.model_path}') raised {type(exc).__name__}: {exc}"
            )
        assert model is not None, "joblib.load() returned None."

    def test_loaded_model_is_a_sklearn_classifier(self):
        """
        The deserialised object must be a valid scikit-learn classifier –
        it must satisfy sklearn's is_classifier() predicate and expose both
        fit() and predict() methods, which predict() in model.py will call.
        """
        model = joblib.load(self.model_path)
        assert is_classifier(model), (
            f"Loaded object {type(model).__name__!r} is not recognised as a "
            "scikit-learn classifier by is_classifier()."
        )
        assert hasattr(model, "fit"),     f"{type(model).__name__} has no fit() method."
        assert hasattr(model, "predict"), f"{type(model).__name__} has no predict() method."

    def test_model_exposes_predict_proba(self):
        """
        The model must expose predict_proba() so that predict() in model.py
        can produce the probability-based score column required by spec §10.
        Classifiers without predict_proba() (e.g. bare SVM) cannot generate
        valid scores and must be wrapped with CalibratedClassifierCV.
        """
        model = joblib.load(self.model_path)
        assert hasattr(model, "predict_proba"), (
            f"{type(model).__name__} does not expose predict_proba().  "
            "Wrap it with CalibratedClassifierCV, or choose a probabilistic "
            "classifier that natively supports probability estimates."
        )

    # ── training JSON schema (spec §11.3) ────────────────────────────────

    def test_training_metrics_json_has_required_keys(self):
        """
        The training-metrics JSON must contain every key listed in §11.3.
        Missing keys will crash the evaluator's parser.
        """
        required = {
            "classifier",
            "seed",
            "training_dataset",
            "target_column",
            "model_path",
            "n_samples",
            "n_features",
            "target_1_percentage",
            "dataset_input_time",
            "training_time",
        }
        missing = required - self.metrics.keys()
        assert not missing, (
            f"Training metrics JSON is missing {len(missing)} key(s): {missing}"
        )

    def test_training_metrics_values_are_sensible(self):
        """
        Numeric fields in the training metrics must satisfy basic sanity
        constraints.  Negative times or zero feature counts indicate that
        the JSON was written before the training was complete.
        """
        violations: list[str] = []

        if self.metrics.get("training_time", -1) <= 0:
            violations.append(
                f"training_time = {self.metrics.get('training_time')} (must be > 0)"
            )
        if self.metrics.get("dataset_input_time", -1) < 0:
            violations.append(
                f"dataset_input_time = {self.metrics.get('dataset_input_time')} (must be ≥ 0)"
            )
        if self.metrics.get("n_samples", 0) <= 0:
            violations.append(
                f"n_samples = {self.metrics.get('n_samples')} (must be > 0)"
            )
        if self.metrics.get("n_features", 0) <= 0:
            violations.append(
                f"n_features = {self.metrics.get('n_features')} (must be > 0)"
            )
        pct = self.metrics.get("target_1_percentage", -1)
        if not (0.0 <= pct <= 100.0):
            violations.append(
                f"target_1_percentage = {pct} (must be in [0, 100])"
            )

        assert not violations, (
            "Training metrics contain invalid values:\n"
            + "\n".join(f"  • {v}" for v in violations)
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  M O D E L   P R E D I C T I O N   T E S T S   (spec §10, §11.3 – predict)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestModelPrediction:
    """
    Validates predict() against spec §10 (prediction CSV format and
    classification statistics) and §11.3 (mandatory Python interface
    and classification-stats JSON schema).
    """

    @pytest.fixture(autouse=True)
    def _setup(self, prediction_outputs, feat_sel):
        """Load all output artefacts once per test method in this class."""
        self.paths       = prediction_outputs
        self.test_csv    = feat_sel["test_csv"]
        self.pred_df     = pd.read_csv(prediction_outputs["predictions_csv"])
        self.test_df     = pd.read_csv(feat_sel["test_csv"])

        with open(prediction_outputs["stats_json"]) as fh:
            self.stats = json.load(fh)

    # ── output-file existence ─────────────────────────────────────────────

    def test_predictions_csv_is_created(self):
        """predict() must write the predictions CSV to disk (§10)."""
        assert Path(self.paths["predictions_csv"]).exists(), (
            "predictions CSV was not created."
        )

    def test_classification_stats_json_is_created(self):
        """predict() must write the classification-stats JSON to disk (§11.3)."""
        assert Path(self.paths["stats_json"]).exists(), (
            "Classification stats JSON was not created."
        )

    # ── predictions CSV columns (spec §10) ───────────────────────────────

    def test_predictions_csv_has_exactly_the_required_columns(self):
        """
        Spec §10 mandates exactly four columns: row_n, target, prediction,
        score.  Extra or missing columns break the evaluator's CSV parser.
        """
        expected = {"row_n", "target", "prediction", "score"}
        actual   = set(self.pred_df.columns)

        missing = expected - actual
        extra   = actual - expected

        assert not missing and not extra, (
            f"Predictions CSV columns are wrong.\n"
            f"  Missing : {missing or '—'}\n"
            f"  Extra   : {extra   or '—'}"
        )

    def test_row_count_matches_test_set(self):
        """
        The predictions CSV must have exactly one row per sample in the
        reduced test set.  A shorter file means some predictions are absent;
        a longer file means duplicate rows were written.
        """
        n_test = len(self.test_df)
        n_pred = len(self.pred_df)
        assert n_pred == n_test, (
            f"Predictions CSV has {n_pred} row(s); "
            f"test set has {n_test} sample(s)."
        )

    # ── row_n column ──────────────────────────────────────────────────────

    def test_row_n_is_zero_indexed_sequential_integers(self):
        """
        row_n must be the zero-based integer index 0, 1, …, n−1 (spec §10
        example: row_n=0, 1, 2, …).  Non-sequential or non-zero-based indices
        would misalign the evaluator's row-lookup logic.
        """
        n = len(self.pred_df)
        expected = list(range(n))
        actual   = self.pred_df["row_n"].tolist()
        assert actual == expected, (
            f"row_n is not a sequential zero-based index.\n"
            f"  First 10 values : {actual[:10]}\n"
            f"  Expected        : {expected[:10]}"
        )

    # ── target column consistency ─────────────────────────────────────────

    def test_target_in_predictions_matches_original_test_labels(self):
        """
        The target column in the predictions CSV must reproduce the ground-
        truth labels from the reduced test CSV in the same order.  Any
        mismatch means labels were shuffled or loaded from the wrong source,
        which would corrupt every reported metric.
        """
        expected_labels = self.test_df[TARGET].tolist()
        actual_labels   = self.pred_df["target"].tolist()
        assert actual_labels == expected_labels, (
            "target column in predictions CSV does not match the ground-truth "
            "labels in the reduced test CSV.\n"
            f"  First 10 expected : {expected_labels[:10]}\n"
            f"  First 10 actual   : {actual_labels[:10]}"
        )

    # ── prediction column edge cases ──────────────────────────────────────

    def test_prediction_column_contains_only_binary_values(self):
        """
        The prediction column must contain only {0, 1} (spec §10).  Any
        other value (e.g. continuous probabilities) indicates that the
        wrong output of the model was written to the file.
        """
        non_binary = self.pred_df.loc[
            ~self.pred_df["prediction"].isin([0, 1]), "prediction"
        ].tolist()
        assert not non_binary, (
            f"prediction column contains {len(non_binary)} non-binary value(s): "
            + repr(non_binary[:6])
            + ("…" if len(non_binary) > 6 else "")
        )

    # ── score column edge cases ───────────────────────────────────────────

    def test_score_column_values_are_valid_probabilities(self):
        """
        score must be the probability of the positive class (spec §10).
        Valid probabilities are strictly within [0.0, 1.0].  Values outside
        this range indicate that raw decision-function outputs (logits, SVM
        margins) were mistakenly written instead of calibrated probabilities.
        """
        scores = self.pred_df["score"]

        below_zero = scores[scores < 0.0].tolist()
        above_one  = scores[scores > 1.0].tolist()

        errors: list[str] = []
        if below_zero:
            errors.append(
                f"{len(below_zero)} score(s) < 0: {below_zero[:4]}"
                + ("…" if len(below_zero) > 4 else "")
            )
        if above_one:
            errors.append(
                f"{len(above_one)} score(s) > 1: {above_one[:4]}"
                + ("…" if len(above_one) > 4 else "")
            )

        assert not errors, (
            "score column contains out-of-range probability values:\n"
            + "\n".join(f"  • {e}" for e in errors)
        )

    def test_score_column_has_no_nan_or_inf(self):
        """
        score must not contain NaN or ±Inf.  These propagate silently through
        the ROC-AUC computation and produce NaN metrics.
        """
        scores = self.pred_df["score"].to_numpy(dtype=float)
        n_nan  = int(np.isnan(scores).sum())
        n_inf  = int(np.isinf(scores).sum())

        assert n_nan == 0, f"score column contains {n_nan} NaN value(s)."
        assert n_inf == 0, f"score column contains {n_inf} infinite value(s)."

    def test_score_is_not_constant_across_all_samples(self):
        """
        A constant score (every row has the same value) indicates a broken
        model that ignores the input entirely.  Even with a small test set
        a functioning probabilistic classifier must produce some variance.
        """
        score_std = self.pred_df["score"].std()
        assert score_std > 0.0, (
            f"score column has zero variance (all values = "
            f"{self.pred_df['score'].iloc[0]:.4f}).  The model appears to "
            "ignore its input."
        )

    # ── classification statistics JSON (spec §10 + §11.3) ─────────────────

    def test_classification_stats_json_has_required_top_level_keys(self):
        """
        The stats JSON must contain every top-level key specified in §11.3.
        The evaluator reads each of these by name.
        """
        required = {
            "classifier",
            "n_samples",
            "target_1_count",
            "target_1_percentage",
            "accuracy",
            "class_0",
            "class_1",
            "roc_auc",
            "confusion_matrix",
        }
        missing = required - self.stats.keys()
        assert not missing, (
            f"Classification stats JSON is missing {len(missing)} key(s): "
            f"{missing}"
        )

    def test_class_metric_dicts_have_required_sub_keys(self):
        """
        class_0 and class_1 sub-objects must each contain precision, recall,
        f1, and support (spec §10 output specification).
        """
        sub_keys = {"precision", "recall", "f1", "support"}
        for label in ("class_0", "class_1"):
            obj = self.stats.get(label, {})
            missing = sub_keys - obj.keys()
            assert not missing, (
                f"'{label}' dict in stats JSON is missing key(s): {missing}"
            )

    def test_classification_metric_values_are_in_valid_range(self):
        """
        precision, recall, f1, and roc_auc must all be in [0.0, 1.0].
        accuracy must also be in [0.0, 1.0].
        Values outside these bounds indicate a computation error or that the
        wrong metric was stored.
        """
        violations: list[str] = []

        for label in ("class_0", "class_1"):
            for metric in ("precision", "recall", "f1"):
                val = self.stats.get(label, {}).get(metric, -1)
                if not (0.0 <= val <= 1.0):
                    violations.append(f"{label}.{metric} = {val}")

        for top_metric in ("accuracy", "roc_auc"):
            val = self.stats.get(top_metric, -1)
            if not (0.0 <= val <= 1.0):
                violations.append(f"{top_metric} = {val}")

        assert not violations, (
            "Classification stats contain out-of-range metric value(s):\n"
            + "\n".join(f"  • {v}" for v in violations)
        )

    def test_class_support_sums_to_n_samples(self):
        """
        support for class_0 and class_1 must sum to the total number of test
        samples.  A mismatch means some rows were counted in both classes or
        neither, which silently corrupts all reported metrics.
        """
        s0      = self.stats.get("class_0", {}).get("support", 0)
        s1      = self.stats.get("class_1", {}).get("support", 0)
        n_total = self.stats.get("n_samples", -1)

        assert s0 + s1 == n_total, (
            f"class_0.support ({s0}) + class_1.support ({s1}) = {s0 + s1}, "
            f"but n_samples = {n_total}."
        )

    def test_confusion_matrix_structure_and_totals(self):
        """
        The confusion_matrix object must have:
        • a 'labels' field equal to [0, 1]
        • a 'matrix' field that is 2 × 2
        • the four cell values must sum to n_samples

        This verifies that the matrix was not transposed, truncated, or
        generated for the wrong label set.
        """
        cm = self.stats.get("confusion_matrix", {})

        assert "labels" in cm,  "confusion_matrix is missing the 'labels' key."
        assert "matrix" in cm,  "confusion_matrix is missing the 'matrix' key."
        assert cm["labels"] == [0, 1], (
            f"confusion_matrix.labels = {cm['labels']}; expected [0, 1]."
        )

        matrix = cm["matrix"]
        assert len(matrix) == 2 and all(len(row) == 2 for row in matrix), (
            f"confusion_matrix.matrix must be 2×2; got shape "
            f"{len(matrix)}×{len(matrix[0]) if matrix else '?'}."
        )

        cell_sum   = sum(matrix[r][c] for r in range(2) for c in range(2))
        n_samples  = self.stats.get("n_samples", -1)
        assert cell_sum == n_samples, (
            f"Sum of confusion matrix cells ({cell_sum}) ≠ n_samples ({n_samples})."
        )

    def test_target_1_count_is_consistent_with_csv(self):
        """
        target_1_count in the stats JSON must equal the actual number of
        positive samples in the predictions CSV target column.
        """
        actual_positives = int((self.pred_df["target"] == 1).sum())
        reported         = self.stats.get("target_1_count", -1)
        assert reported == actual_positives, (
            f"stats JSON reports target_1_count={reported} but the predictions "
            f"CSV has {actual_positives} positive sample(s)."
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  M U L T I - C L A S S I F I E R   T E S T S   (spec §9)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAllClassifiers:
    """
    Spec §9 mandates three classifiers, one of which must be Random Forest.

    Strategy
    --------
    •  Random Forest is tested unconditionally (failure → test fails).
    •  _CANDIDATE_CLFS is parametrized; if the group chose a different name,
       the test is skipped automatically – it does not fail.
    •  The combined_count test verifies that across all runs at least two
       non-RF classifiers succeeded (so three total are implemented).

    Note: The evaluator can grep the pytest output for PASSED / SKIPPED to
    confirm which names the group supports.
    """

    def test_random_forest_trains_and_saves_model(self, feat_sel, tmp_path):
        """
        Random Forest is the only classifier whose name is fixed by the spec
        (§9).  This test fails unconditionally if it is not implemented.
        """
        model_path   = str(tmp_path / "rf_model.joblib")
        metrics_json = str(tmp_path / "rf_metrics.json")

        train(
            classifier       = "random_forest",
            reducedTrain_csv = feat_sel["train_csv"],
            target_column    = TARGET,
            model_path       = model_path,
            metrics_json     = metrics_json,
            seed             = SEED,
        )

        assert Path(model_path).exists(), (
            "random_forest did not produce a .joblib model file."
        )
        model = joblib.load(model_path)
        assert is_classifier(model), (
            "Object loaded from random_forest .joblib is not a classifier."
        )

    @pytest.mark.parametrize("clf_name", _CANDIDATE_CLFS)
    def test_alternative_classifier_trains_and_saves_model(
        self, clf_name, feat_sel, tmp_path
    ):
        """
        Tries each name in _CANDIDATE_CLFS.  Tests are skipped for names the
        group did not implement; the suite does not penalise unknown names.
        The group must ensure that at least two of these pass (§9 requires
        three classifiers total).
        """
        model_path   = str(tmp_path / f"{clf_name}_model.joblib")
        metrics_json = str(tmp_path / f"{clf_name}_metrics.json")

        try:
            train(
                classifier       = clf_name,
                reducedTrain_csv = feat_sel["train_csv"],
                target_column    = TARGET,
                model_path       = model_path,
                metrics_json     = metrics_json,
                seed             = SEED,
            )
        except (ValueError, KeyError, NotImplementedError) as exc:
            pytest.skip(
                f"Classifier '{clf_name}' is not implemented by this group "
                f"(raised {type(exc).__name__}: {exc})."
            )

        # If train() returned without raising, the artefacts must exist and
        # the model must be a valid scikit-learn classifier.
        assert Path(model_path).exists(), (
            f"train('{clf_name}') did not raise an error but also did not "
            "create the model file."
        )
        model = joblib.load(model_path)
        assert is_classifier(model), (
            f"Object saved for '{clf_name}' is not a scikit-learn classifier."
        )
        assert hasattr(model, "predict_proba"), (
            f"'{clf_name}' model has no predict_proba(); wrap it with "
            "CalibratedClassifierCV to produce valid probability scores."
        )