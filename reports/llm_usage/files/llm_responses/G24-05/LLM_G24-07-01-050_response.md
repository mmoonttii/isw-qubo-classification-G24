Looking at your spec (§13) and your `test_pipeline.py`, here's what's actually going on and how to produce a proper `data/sample_test_dataset.csv`.

## Two important things to notice first

1. **§13 requirement**: the file must exist in the repo at `data/sample_test_dataset.csv`, contain a small dataset, and guarantee **both target values with ≥10% representation** each.
2. **Your `test_pipeline.py` already auto-generates it** — the `raw_csv` fixture (lines 135–197) builds a 120-row, 20-feature synthetic dataset every time you run `pytest`, and writes a copy to `data/sample_test_dataset.csv` automatically. So technically, just running the tests once will create the file for you.

However, for a graded deliverable it's better to have a file that's **deliberately committed**, not just a test side-effect — and ideally one that looks like a real (small) subset of your actual credit-risk dataset, not pure noise. Here's a tutorial covering both approaches.

---

## Option A (recommended): extract a real subset from `input_dataset.csv`

This matches what §13 literally suggests ("può essere estratto dal dataset originale").

**Step 1 — Load your real dataset**
```python
import pandas as pd
df = pd.read_csv("data/input_dataset.csv")
target_col = "target"  # use your actual target column name
```

**Step 2 — Stratified sample, keeping both classes ≥10%**
```python
pos = df[df[target_col] == 1]
neg = df[df[target_col] == 0]

n_pos = max(20, int(0.20 * 150))   # e.g. want ~20% positive in a ~150-row sample
n_neg = 150 - n_pos

sample = pd.concat([
    pos.sample(n=n_pos, random_state=42),
    neg.sample(n=n_neg, random_state=42),
]).sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
```

**Step 3 — Keep it small but representative**
Aim for ~100–200 rows and maybe 15–30 feature columns (you don't need all 145 — a subset is fine and makes tests faster). If you want to also test the "drop sparse columns" logic, deliberately keep a couple of near-empty columns in the sample.

**Step 4 — Sanity check before saving**
```python
assert sample[target_col].isin([0, 1]).all()
assert sample[target_col].mean() >= 0.10
assert (1 - sample[target_col]).mean() >= 0.10
```

**Step 5 — Save it**
```python
sample.to_csv("data/sample_test_dataset.csv", index=False)
```

**Step 6 — Verify it plugs into your pipeline**
```bash
python src/qubo_project/preprocessing.py \
  --input data/sample_test_dataset.csv \
  --target target \
  --out-data outputs/sample_normalized.csv \
  --out-json outputs/sample_preproc.json \
  --min-perc-valid 0.05
```
If this runs cleanly, your sample file is valid for the rest of the pipeline too.

---

## Option B: synthetic generation (what your test suite already does)

If the real dataset isn't ready yet, or you want a fully reproducible, seed-controlled sample independent of any external file, mirror the logic already in `test_pipeline.py`'s `raw_csv` fixture:

```python
import numpy as np, pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
n_samples, n_features, n_sparse = 120, 20, 3

X = rng.standard_normal((n_samples, n_features))
# inject some NaNs to test missing-value handling
nan_mask = rng.random((n_samples, n_features - n_sparse)) < 0.05
X[:, n_sparse:][nan_mask] = np.nan
# make a few columns near-all-zero to test the sparse-column drop logic
for i in range(n_sparse):
    col = np.zeros(n_samples)
    idx = rng.choice(n_samples, size=2, replace=False)
    col[idx] = rng.standard_normal(2)
    X[:, i] = col

y = np.zeros(n_samples, dtype=int)
y[rng.choice(n_samples, size=24, replace=False)] = 1  # 20% positive

df = pd.DataFrame(X, columns=[f"feature_{i:02d}" for i in range(n_features)])
df["target"] = y

Path("data").mkdir(exist_ok=True)
df.to_csv("data/sample_test_dataset.csv", index=False)
```

Save this as a small standalone script (e.g. `scripts/make_sample_dataset.py`) and run it once. This is exactly the kind of file that should also go into `reports/llm_usage/files/` since it's LLM-generated, per §16.

---

## A note on consistency with your test suite

Right now `test_pipeline.py` *regenerates and overwrites* `data/sample_test_dataset.csv` every time pytest runs (line 195), using its own fixed synthetic logic. If you commit a hand-curated real-data sample (Option A) but then run `pytest`, it will silently get overwritten by the synthetic one. You have two clean choices:

- **Decouple them**: change the fixture to write to a different temp-only location and stop touching `data/`, treating `data/sample_test_dataset.csv` as a fixed, committed artifact that tests *read* rather than *regenerate*.
- **Or embrace it**: keep the fixture as the single source of truth for the sample dataset, and don't hand-edit `data/sample_test_dataset.csv` separately — just document in the README that it's test-generated.

Either is defensible, but pick one and say so explicitly in `README.md`, since graders will check that §13 requirements (numeric-only, ≥10% per class, both errors and sparse columns represented) are visibly satisfied by whatever ends up in that file.