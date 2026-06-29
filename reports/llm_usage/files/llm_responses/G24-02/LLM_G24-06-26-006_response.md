Now let me read the project PDF specification to understand the full requirements for `feature_selection.py`.Now I have a thorough understanding of the entire specification. Let me also check the existing `preprocessing.py` to align the prompt style perfectly.Good — the preprocessing.py isn't in the project directory, but I have all the context from memory. Now let me write the comprehensive prompt.Now I have everything I need. Let me compose the complete, self-contained prompt.Here's the prompt — ready to paste directly into a new chat with Claude Sonnet 4.6. A few notes on what makes it especially complete:

**QUBO math is spelled out exactly.** The specification's appendix describes the cost function in terms of Italian text and images that wouldn't survive a copy-paste. The prompt translates it into unambiguous Python-ready notation: diagonal `Q[j,j] = -α|ρ_Vj|`, off-diagonal `Q[j,k] = (1-α)|ρ_jk|`. This is critical for the **bonus point** — the evaluator will check the QUBO cost value, so the formula must be exactly right.

**Alpha search is bisection, not linear sweep.** Because feature count is monotonically non-decreasing in α, binary search finds the target K in O(log N) iterations instead of wasting the full `alpha_computations` budget on a linear scan. Fewer wasted iterations → lower cost values found → better chance at the bonus.

**Correlations are computed once, outside the loop.** This is the key performance insight: Spearman correlations don't change with α, only the Q values do. Pre-computing them makes the loop fast enough to handle 1.5M-row datasets.

**The style section prevents drift** from `preprocessing.py`'s conventions (chunked reads, `_to_output_path`, comment headers) so the two modules feel like they came from the same hand.


```md
# Prompt: Implement `feature_selection.py` for the QUBO Binary Classification Project

## Context

You are implementing the second module of a Python package called `qubo_project`, located at:

```
isw-qubo-classification-GXX/
└── src/
    └── qubo_project/
        ├── __init__.py
        ├── preprocessing.py   ← already implemented and tested
        ├── feature_selection.py  ← YOU MUST IMPLEMENT THIS
        ├── model.py
        └── gui.py
```

The project handles large-scale datasets (1.5M+ records) performing binary classification with QUBO-based feature reduction.

The first module (`preprocessing.py`) is already done. It uses:
- Chunked reads (128k rows/chunk) for memory safety at scale
- Welford's online algorithm for numerically stable z-score normalization
- A `_to_output_path(filename)` helper that routes all outputs to the `outputs/` directory (no absolute paths)
- `argparse` for the CLI, following a comment-header style for code sections
- Python 3.11+, `pandas >= 2.2.0`, `numpy >= 1.26.0`, `scikit-learn >= 1.4.0`

---

## What to implement

Implement `src/qubo_project/feature_selection.py` **completely**, including:
1. The mandatory `select_features()` function (see interface below)
2. The QUBO matrix construction using Spearman correlations (see math below)
3. The QUBO optimization loop varying `alpha` until the target feature count `K` is hit
4. The `argparse`-based CLI (see CLI spec below)
5. All required file outputs (2 CSVs, 1 optimizations CSV, 1 JSON)

---

## Mandatory function interface

```python
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
```

---

## Dataset split rule

The split is a **hard cut**: the **first** `M` samples → training set, the **remaining** samples → test set.  
`M = total_samples - round(percTest * total_samples)`  
*(i.e., `percTest` fraction goes to the test set, taken from the end of the file)*

---

## QUBO cost function — exact specification (bonus point criteria)

This is the formula prescribed by the specification. **Do not use any other formulation.**

### Variables
- `U`: feature matrix — shape `(m_samples, n_features)`, **from the training set only**
- `V`: binary target vector — shape `(m_samples,)`, **from the training set only**
- `n`: number of features after preprocessing
- `xj ∈ {0, 1}`: binary variable, 1 if feature `j` is selected
- `ρ_Vj`: **absolute value** of the Spearman correlation between feature column `j` and target `V`
- `ρ_jk`: **absolute value** of the Spearman correlation between feature column `j` and feature column `k`
- `α ∈ [0, 1]`: weighting parameter

### Objective to minimize
```
f(x) = -α * Σ_j( xj * |ρ_Vj| )   +   (1-α) * Σ_{j<k}( xj * xk * |ρ_jk| )
```

**Interpretation:**
- The first term (negative) rewards selecting features that correlate with the target → maximise influence
- The second term (positive) penalises selecting pairs of mutually correlated features → minimise redundancy

### Building the Q matrix

Exploit the binary property `xj² = xj` to write this as `x^T Q x` (minimized):

- **Diagonal**: `Q[j, j] = -α * |ρ_Vj|`
- **Off-diagonal** (upper triangle, `j < k`): `Q[j, k] = (1-α) * |ρ_jk|`
- The matrix is symmetric: `Q[k, j] = Q[j, k]`

The optimization problem is:
```
x* = argmin_{x ∈ {0,1}^n}  x^T Q x
```

### Important implementation notes on the Q matrix
- Compute **all** Spearman correlations **once** before the alpha search loop (they do not depend on alpha). Only the Q matrix values change with alpha.
- Use `scipy.stats.spearmanr` for Spearman correlations.
- The correlation matrix is dense (nearly all elements non-zero) — do not sparse-encode it.
- Use **absolute values** for all correlations, both diagonal and off-diagonal.
- Track and report `q_matrix_creation_time` (time to compute all Spearman correlations).

---

## Alpha search strategy

- Compute `K = round(percSelected * n)` (target feature count)
- A result is acceptable if `K - allowance <= count_ones(x*) <= K + allowance`
- Search `alpha` in `[0, 1]` range, at most `alpha_computations` iterations
- **Recommended strategy**: binary search or adaptive bisection on `alpha` (not a naive linear sweep), since the number of selected features is monotonically non-decreasing with `alpha`. This is more efficient and produces better results.
- If no alpha yields exactly K±allowance features within the allowed iterations, return the result with the closest feature count.
- Log every attempted alpha (in increasing order) to `output_ottim_csv`.

---

## QUBO optimizer choice

Use **Simulated Annealing** as the QUBO solver. Recommended library: `neal` (`dwave-neal`), which provides `neal.SimulatedAnnealingSampler`. If `neal` is not available, fall back to a custom simulated annealing implementation using `numpy`.

**Why SA**: It handles dense Q matrices efficiently, scales to 100+ variables, is reproducible via seed, and is widely available open-source.

Record `algorithm: "simulated_annealing"` in the output JSON.

Timing: measure the wall-clock time of each individual optimization call and collect mean + std dev.

---

## Output files

### 1. `output_ottim_csv` — one row per alpha tried (CSV)

Columns (in this order):
```
alpha, optimization_time, n_selected, cost_value
```
- `alpha`: the alpha value tried
- `optimization_time`: wall-clock seconds for that single optimization call
- `n_selected`: number of 1s in the solution vector `x*`
- `cost_value`: the value of the objective function `x^T Q x` for that solution

Rows must be sorted by `alpha` in **ascending** order.

### 2. `output_json` — summary statistics (JSON)

Exact required structure (values are illustrative):
```json
{
  "n_features": 95,
  "target_ratio": 0.20,
  "target_k": 19,
  "allowance": 1,
  "n_selected": 19,
  "alpha": 0.344,
  "selected_vector": [1, 0, 0, 1, 0],
  "selected_feature_names": ["feature_1", "feature_4"],
  "algorithm": "simulated_annealing",
  "seed": 42,
  "alpha_computations": 6,
  "percTest": 0.30,
  "training_dataset_size": 14000,
  "test_dataset_size": 6000,
  "q_matrix_creation_time": 2.53,
  "mean_optimization_time": 0.23,
  "std_dev_optimization_time": 0.044
}
```

All keys are mandatory. `selected_vector` and `selected_feature_names` reflect the **final** chosen solution.

### 3. `reducedTrain_csv` — training set with selected features only (CSV)

- Columns: selected feature columns + target column (target **last**)
- First row: original column header names
- Rows: first `M` samples of the normalized dataset

### 4. `reducedTest_csv` — test set with selected features only (CSV)

- Same column layout as training CSV
- Rows: remaining samples (from index `M` onward)

---

## Output path routing

All output files must be routed through a `_to_output_path(filename)` helper, identical in behaviour to the one in `preprocessing.py`:

```python
import os

def _to_output_path(filename: str) -> str:
    """Route a bare filename (or relative path) into the outputs/ directory."""
    os.makedirs("outputs", exist_ok=True)
    return os.path.join("outputs", os.path.basename(filename))
```

This ensures no absolute paths appear anywhere in the code.

---

## Mandatory CLI interface

The module must be runnable as a script. The `argparse` CLI must match this exact invocation:

```
python feature_selection.py \
  --in-normalized normalized.csv \
  --out-train training_reduced.csv \
  --out-test test_reduced.csv \
  --out-optimizations optimizations.csv \
  --out-json feature_selection_result.json \
  --target target \
  --perc-selected 0.20 \
  --allowance 1 \
  --perc-test 0.30 \
  --seed 42 \
  --alpha-computations 10
```

Expected printed output (paths only, one per line, in this order):
```
outputs/training_reduced.csv
outputs/test_reduced.csv
outputs/optimizations.csv
outputs/feature_selection_result.json
```

Use an `if __name__ == "__main__":` block to invoke `select_features()` with the parsed arguments.

---

## Code style requirements

Match the style of `preprocessing.py` exactly:
- Use `# ── SECTION NAME ──` comment headers to delimit logical sections
- No new third-party imports beyond what is strictly necessary (`pandas`, `numpy`, `scipy`, `neal`/`dwave-neal`, `sklearn` is available but not needed here)
- All file I/O goes through `_to_output_path()`
- Use chunked reading (`chunksize=131072`) when loading the normalized CSV, accumulating into a single DataFrame — the dataset can be 1.5M+ rows
- Validate that `target_column` exists in the input CSV; raise `ValueError` with a clear message if not
- Log progress to stdout with timestamps at key steps (loading complete, Q matrix done, each alpha tried, final result)

---

## Dependencies to add to `requirements.txt` (if not already present)

```
scipy>=1.12.0
dwave-neal>=0.6.0
```

---

## Summary checklist

- [ ] `_to_output_path()` helper defined
- [ ] Chunked loading of `normalized_csv` into a DataFrame
- [ ] Training/test hard split at row index `M`
- [ ] Spearman correlation computation using **training set only**, timed separately
- [ ] Q matrix construction: diagonal = `-alpha * |rho_Vj|`, off-diagonal = `(1-alpha) * |rho_jk|`
- [ ] Alpha bisection/binary search loop (max `alpha_computations` iterations)
- [ ] Per-iteration: optimize, count ones, record `(alpha, time, n_selected, cost)`, check vs K±allowance
- [ ] Final solution: build `selected_vector`, `selected_feature_names`
- [ ] Save `reducedTrain_csv` (selected cols + target, first M rows)
- [ ] Save `reducedTest_csv` (selected cols + target, remaining rows)
- [ ] Save `output_ottim_csv` sorted by alpha ascending
- [ ] Save `output_json` with all 14 required keys
- [ ] `argparse` CLI with all 10 flags, prints 4 output paths
- [ ] `if __name__ == "__main__":` block
- [ ] No absolute paths anywhere
```