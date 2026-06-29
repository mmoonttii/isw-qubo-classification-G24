Here's the `train` function implementation:Here's a walkthrough of every design decision made:

**Classifier registry (`CLASSIFIERS` dict)**
All three models are stored as `lambda seed: Model(...)` factories. Adding a fourth classifier later is a one-liner. The three chosen are:
- `random_forest` — mandatory per spec, uses `n_jobs=-1` for parallel training
- `logistic_regression` — fast, interpretable; `max_iter=1000` avoids convergence warnings on normalised data
- `gradient_boosting` — stronger baseline than a plain decision tree, still deterministic via `random_state`

**Timing split**
Two separate `time.perf_counter()` pairs measure I/O and training independently, exactly matching the required JSON fields `dataset_input_time` and `training_time`.

**Path safety**
`Path(...).parent.mkdir(parents=True, exist_ok=True)` is called before every write, so `outputs/` (or any nested directory) is created automatically — no hard-coded paths anywhere.

**JSON output**
Every field name and type matches the spec exactly. `target_1_percentage` is the mean of the binary target × 100, which handles both `0/1` integer and float-encoded targets correctly.

**What's still missing in this file (next prompts)**
- `predict()` function
- `if __name__ == "__main__":` block with the full `argparse` CLI (`model.py train ...` / `model.py predict ...`)