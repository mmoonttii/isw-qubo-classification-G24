"""
feature_selection.py — QUBO-based feature selection for binary classification.

Implements select_features() and an argparse CLI.

Fixes applied vs. previous version
────────────────────────────────────
1. FACTOR-OF-2 BUG (critical correctness fix)
   The spec's double-sum  Σ_j Σ_{k≠j} x_j x_k |ρ_jk|  counts every
   (j,k) pair TWICE (once as (j,k), once as (k,j)).  When the full
   symmetric matrix Q is handed to D-Wave/neal via its upper-triangle
   dict API, each off-diagonal pair is counted only ONCE, halving the
   independence penalty relative to the spec.  The fix: multiply every
   off-diagonal entry in Q_dict by 2.0 so that D-Wave's energy
       E = Σ_{i≤j} Q_dict[i,j] · x_i x_j
   equals the spec's objective  f(x) = x^T Q_code x  (full matrix).
   The numpy fallback already used `x @ Q @ x` (full matrix), so it was
   already correct and is left unchanged.

2. NEAL NUM_READS (optimization quality)
   num_reads=1 gives a single SA trajectory — high variance and poor
   minima for dense QUBO matrices.  Raised to _NEAL_NUM_READS (200).
   D-Wave/neal automatically returns the sample with the lowest energy
   across all reads.

3. NUMPY SA QUALITY (fallback robustness)
   n_steps raised from max(10_000, 150·n) to max(50_000, 500·n) and
   the solver now runs _SA_RESTARTS (5) independent restarts, keeping
   the globally best solution.  This keeps the fallback competitive
   when dwave-neal is unavailable.
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────────
import os
import json
from pathlib import Path
import time
import warnings
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import spearmanr


# ── TUNABLE SOLVER CONSTANTS ──────────────────────────────────────────────────
_NEAL_NUM_READS: int = 200   # independent SA restarts inside dwave-neal
_SA_RESTARTS:    int = 5     # restarts for the pure-numpy fallback
_SA_STEPS_BASE:  int = 50_000
_SA_STEPS_PER_N: int = 500


# ── OUTPUT PATH HELPER ────────────────────────────────────────────────────────

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


# ── QUBO MATRIX BUILDER ───────────────────────────────────────────────────────

def _build_Q(alpha: float, rho_Vj: np.ndarray, rho_jk: np.ndarray) -> np.ndarray:
    """Build the symmetric QUBO matrix Q_code for the given alpha.

    The spec defines the objective to minimise as:

        f(x) = −x^T Q_spec x

    where  Q_spec[j,j] =  α |ρ_Vj|  and  Q_spec[j,k] = −(1−α)|ρ_jk|.

    We store Q_code = −Q_spec so that  f(x) = x^T Q_code x,
    i.e. we directly minimise  x^T Q_code x:

        Diagonal:     Q_code[j,j] = −α |ρ_Vj|
        Off-diagonal: Q_code[j,k] = (1−α)|ρ_jk|   (j ≠ k, symmetric)

    Parameters
    ----------
    alpha   : weighting parameter in [0, 1]
    rho_Vj  : 1-D array shape (n,)    — |Spearman(feature_j, target)|
    rho_jk  : 2-D array shape (n, n)  — |Spearman(feature_j, feature_k)|
    """
    Q = (1.0 - alpha) * rho_jk          # fills off-diagonal (and diagonal temporarily)
    np.fill_diagonal(Q, -alpha * rho_Vj) # overwrite diagonal with correct values
    return Q


# ── SA BACKENDS ───────────────────────────────────────────────────────────────

def _optimize_qubo_neal(Q: np.ndarray, seed: int) -> np.ndarray:
    """Solve QUBO via dwave-neal SimulatedAnnealingSampler.

    The spec's objective (full-matrix form) is:

        f(x) = x^T Q_code x
             = Σ_j Q[j,j] x_j
               + 2 · Σ_{j<k} Q[j,k] x_j x_k        ← factor of 2 matters

    D-Wave/neal's sample_qubo computes the upper-triangle energy:

        E_neal = Σ_{i≤j} Q_dict[i,j] · x_i x_j
               = Σ_j Q_dict[j,j] x_j
                 + 1 · Σ_{j<k} Q_dict[j,k] x_j x_k  ← only one copy per pair

    To make E_neal = f(x), every OFF-DIAGONAL entry in Q_dict must be
    multiplied by 2.  The diagonal is unchanged (x_j^2 = x_j for binary x).
    """
    import neal
    n = Q.shape[0]

    # Build upper-triangle QUBO dict with the factor-of-2 correction.
    Q_dict: dict = {}
    for i in range(n):
        for j in range(i, n):
            if i == j:
                # Diagonal: same in both conventions.
                val = float(Q[i, i])
            else:
                # Off-diagonal: multiply by 2 so D-Wave energy == f(x).
                val = 2.0 * float(Q[i, j])
            if val != 0.0:
                Q_dict[(i, j)] = val

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="dwave")
        response = neal.SimulatedAnnealingSampler().sample_qubo(
            Q_dict,
            num_reads=_NEAL_NUM_READS,  # was 1 — now runs many restarts
            seed=seed,
        )

    sample = response.first.sample   # lowest-energy solution across all reads
    return np.array([sample.get(i, 0) for i in range(n)], dtype=int)


def _optimize_qubo_numpy(Q: np.ndarray, seed: int) -> np.ndarray:
    """Fallback SA using only numpy (used when dwave-neal is unavailable).

    Correctly minimises  f(x) = x^T Q x  (full symmetric matrix).

    Runs _SA_RESTARTS independent trajectories from random starting points
    and returns the solution with the globally lowest cost.  Each trajectory
    uses a geometric cooling schedule from T_start → T_end.
    """
    rng    = np.random.default_rng(seed)
    n      = Q.shape[0]
    n_steps = max(_SA_STEPS_BASE, _SA_STEPS_PER_N * n)

    T_start = 2.0
    T_end   = 0.001

    best_x    = None
    best_cost = np.inf

    for restart in range(_SA_RESTARTS):
        x    = rng.integers(0, 2, size=n).astype(float)
        cost = float(x @ Q @ x)   # full-matrix: correct per spec

        for step in range(n_steps):
            T        = T_start * (T_end / T_start) ** (step / n_steps)
            j        = int(rng.integers(0, n))
            x_new    = x.copy()
            x_new[j] = 1.0 - x_new[j]
            cost_new  = float(x_new @ Q @ x_new)
            delta     = cost_new - cost
            if delta < 0.0 or rng.random() < np.exp(-delta / T):
                x, cost = x_new, cost_new

        if cost < best_cost:
            best_cost = cost
            best_x    = x.copy()

    return best_x.astype(int)


def _solve_qubo(Q: np.ndarray, seed: int) -> np.ndarray:
    """Dispatch to dwave-neal; fall back to the numpy SA implementation."""
    try:
        return _optimize_qubo_neal(Q, seed)
    except ImportError:
        return _optimize_qubo_numpy(Q, seed)


# ── MAIN FUNCTION ─────────────────────────────────────────────────────────────

def select_features(
    normalized_csv: str,          # Input: normalized dataset (output of preprocessing.py)
    reducedTrain_csv: str,        # Output: training dataset with only selected features
    reducedTest_csv: str,         # Output: test dataset with only selected features
    output_ottim_csv: str,        # Output: CSV with one row per alpha tried
    output_json: str,             # Output: JSON with stats and selected feature info
    target_column: str,           # Name of the binary target column
    percTest: float = 0.30,       # Fraction of dataset to use as test set
    percSelected: float = 0.20,   # Fraction of features to select
    allowance: int = 1,           # Tolerance: K ± allowance features is acceptable
    seed: int = 42,               # RNG seed for reproducibility
    alpha_computations: int = 100 # Max number of alpha values to try
) -> None:

    # ── LOAD NORMALIZED CSV (CHUNKED) ─────────────────────────────────────────
    print(f"[{datetime.now():%H:%M:%S}] Loading normalized dataset from '{normalized_csv}' ...")

    chunks = []
    for chunk in pd.read_csv(normalized_csv, chunksize=131072):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in '{normalized_csv}'. "
            f"Available columns: {list(df.columns)}"
        )

    total_samples = len(df)
    n_cols        = len(df.columns)
    print(
        f"[{datetime.now():%H:%M:%S}] Dataset loaded: "
        f"{total_samples} rows, {n_cols} columns (including target)."
    )

    # ── TRAINING / TEST HARD SPLIT ────────────────────────────────────────────
    # First M rows → training set; remaining rows → test set.
    M = total_samples - round(percTest * total_samples)

    df_train = df.iloc[:M].reset_index(drop=True)
    df_test  = df.iloc[M:].reset_index(drop=True)

    print(
        f"[{datetime.now():%H:%M:%S}] Split complete: "
        f"{len(df_train)} training samples, {len(df_test)} test samples "
        f"(cut at row index {M})."
    )

    # ── SPEARMAN CORRELATIONS (TRAINING SET ONLY, TIMED) ─────────────────────
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
    combined = np.column_stack([U_train, V_train])   # shape (M, n+1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress ConstantInputWarning for flat columns
        stat = spearmanr(combined).statistic

    # scipy returns a scalar when the input has exactly 2 columns (n == 1).
    if n == 1:
        val       = 0.0 if np.isnan(float(stat)) else float(stat)
        corr_full = np.array([[1.0, val], [val, 1.0]])
    else:
        corr_full = np.nan_to_num(stat, nan=0.0)

    rho_jk = np.abs(corr_full[:n, :n])   # feature–feature absolute correlations
    rho_Vj = np.abs(corr_full[:n,  n])   # feature–target  absolute correlations

    q_matrix_creation_time = time.perf_counter() - t_corr_start

    print(
        f"[{datetime.now():%H:%M:%S}] Spearman correlations computed in "
        f"{q_matrix_creation_time:.3f}s."
    )

    # ── ALPHA BISECTION SEARCH ────────────────────────────────────────────────
    # n_selected is monotonically non-decreasing with alpha:
    #   alpha → 0  ⟹  n_selected → 0   (independence term dominates)
    #   alpha → 1  ⟹  n_selected → n   (influence term dominates)

    tried = []
    best  = None   # (alpha, x_copy, n_selected, cost)

    lo, hi = 0.0, 1.0

    for iteration in range(alpha_computations):
        alpha = (lo + hi) / 2.0
        Q     = _build_Q(alpha, rho_Vj, rho_jk)

        t0       = time.perf_counter()
        x        = _solve_qubo(Q, seed + iteration)
        opt_time = time.perf_counter() - t0

        n_selected = int(x.sum())
        # Cost is evaluated with the full-matrix formula (consistent with spec).
        cost = float(x @ Q @ x)

        tried.append({
            "alpha":             alpha,
            "optimization_time": opt_time,
            "n_selected":        n_selected,
            "cost_value":        cost,
        })

        print(
            f"[{datetime.now():%H:%M:%S}] iter {iteration + 1:3d} | "
            f"alpha={alpha:.6f} | n_selected={n_selected} "
            f"(target {K}±{allowance}) | "
            f"cost={cost:.4f} | time={opt_time:.3f}s"
        )

        # Keep the candidate whose n_selected is closest to K.
        if best is None or abs(n_selected - K) < abs(best[2] - K):
            best = (alpha, x.copy(), n_selected, cost)

        # Stop as soon as tolerance is satisfied.
        if K - allowance <= n_selected <= K + allowance:
            print(
                f"[{datetime.now():%H:%M:%S}] Acceptable solution found: "
                f"{n_selected} feature(s) at alpha={alpha:.6f}."
            )
            break

        # Bisect: too few features → raise alpha; too many → lower alpha.
        if n_selected < K - allowance:
            lo = alpha
        else:
            hi = alpha

    # ── UNPACK BEST SOLUTION ──────────────────────────────────────────────────
    best_alpha, best_x, best_n_selected, best_cost = best

    opt_times     = [row["optimization_time"] for row in tried]
    mean_opt_time = float(np.mean(opt_times))
    std_opt_time  = float(np.std(opt_times, ddof=0))

    print(
        f"[{datetime.now():%H:%M:%S}] Search complete: {len(tried)} alpha(s) tried | "
        f"best n_selected={best_n_selected} | alpha={best_alpha:.6f}."
    )

    selected_vector        = best_x.tolist()
    selected_feature_names = [feature_cols[j] for j, v in enumerate(best_x) if v == 1]

    print(
        f"[{datetime.now():%H:%M:%S}] Selected {best_n_selected} feature(s): "
        f"{selected_feature_names}"
    )

    # ── SAVE reducedTrain_csv ──────────────────────────────────────────────────
    out_train_path = _to_output_path(reducedTrain_csv)
    df_train[selected_feature_names + [target_column]].to_csv(
        out_train_path, index=False
    )
    print(f"[{datetime.now():%H:%M:%S}] Saved training set → {out_train_path}")

    # ── SAVE reducedTest_csv ───────────────────────────────────────────────────
    out_test_path = _to_output_path(reducedTest_csv)
    df_test[selected_feature_names + [target_column]].to_csv(
        out_test_path, index=False
    )
    print(f"[{datetime.now():%H:%M:%S}] Saved test set     → {out_test_path}")

    # ── SAVE output_ottim_csv (sorted by alpha ascending) ─────────────────────
    out_ottim_path = _to_output_path(output_ottim_csv)
    (
        pd.DataFrame(tried, columns=["alpha", "optimization_time", "n_selected", "cost_value"])
        .sort_values("alpha")
        .to_csv(out_ottim_path, index=False)
    )
    print(f"[{datetime.now():%H:%M:%S}] Saved optimizations log → {out_ottim_path}")

    # ── SAVE output_json ───────────────────────────────────────────────────────
    out_json_path = _to_output_path(output_json)
    summary = {
        "n_features":                n,
        "target_ratio":              percSelected,
        "target_k":                  K,
        "allowance":                 allowance,
        "n_selected":                best_n_selected,
        "alpha":                     round(best_alpha, 6),
        "selected_vector":           selected_vector,
        "selected_feature_names":    selected_feature_names,
        "algorithm":                 "simulated_annealing",
        "seed":                      seed,
        "alpha_computations":        len(tried),
        "percTest":                  percTest,
        "training_dataset_size":     len(df_train),
        "test_dataset_size":         len(df_test),
        "q_matrix_creation_time":    round(q_matrix_creation_time, 6),
        "mean_optimization_time":    round(mean_opt_time, 6),
        "std_dev_optimization_time": round(std_opt_time, 6),
    }
    with open(out_json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[{datetime.now():%H:%M:%S}] Saved JSON summary  → {out_json_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QUBO-based feature selection for binary classification."
    )
    parser.add_argument("--in-normalized",     required=True,  help="Normalized input CSV")
    parser.add_argument("--out-train",         required=True,  help="Output training CSV")
    parser.add_argument("--out-test",          required=True,  help="Output test CSV")
    parser.add_argument("--out-optimizations", required=True,  help="Output optimizations CSV")
    parser.add_argument("--out-json",          required=True,  help="Output JSON file")
    parser.add_argument("--target",            required=True,  help="Name of the target column")
    parser.add_argument("--perc-selected",     type=float, default=0.20,
                        help="Fraction of features to select (default: 0.20)")
    parser.add_argument("--allowance",         type=int,   default=1,
                        help="Tolerance on the number of selected features (default: 1)")
    parser.add_argument("--perc-test",         type=float, default=0.30,
                        help="Fraction of dataset to use as test set (default: 0.30)")
    parser.add_argument("--seed",              type=int,   default=42,
                        help="RNG seed for reproducibility (default: 42)")
    parser.add_argument("--alpha-computations", type=int,  default=100,
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