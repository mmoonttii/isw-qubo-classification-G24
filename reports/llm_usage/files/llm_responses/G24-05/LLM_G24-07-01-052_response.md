Good news first: **52/58 passed** — preprocessing, normalization, feature selection, and CLI tests are all fine. The failures are isolated to one thing: `model.train()`.

## What's actually happening

Look at the captured stdout for each failure — it tells the whole story:

```
[train] Model saved  → /Users/monti/Projects/isw-qubo-classification-G24/outputs/rf_model.joblib
```

But the test passed in:
```
model_path = str(tmp_path / "rf_model.joblib")
# e.g. /private/tmp/pytest-of-monti/pytest-4/test_random_forest_trains_and_0/rf_model.joblib
```

**`train()` is not writing to the `model_path` the caller gave it.** It's silently redirecting the save into the project's own `outputs/` directory instead — presumably keeping only the basename (`rf_model.joblib`) and discarding the directory component the caller specified, then always writing under `<project_root>/outputs/`.

This is the same pattern already flagged in the test file's own comments for `preprocessing.py` and `feature_selection.py` — those two modules have a private `_to_output_path()` helper that strips any directory component from output paths and forces everything into `outputs/`. It looks like `model.py`'s `train()` (and possibly `predict()`) copies that same behavior. That's fine for those two modules because §12's CLI examples only ever pass bare filenames like `--out-data normalized.csv`. But `train()`/`predict()` per §11.3 need to honor whatever `model_path` / `metrics_json` the caller supplies literally — including full absolute paths outside the project — because the evaluator will call these functions directly with its own paths, not just via the CLI.

## Why 3 fail and not all 6 classifiers

`random_forest`, `logistic_regression`, and `gradient_boosting` all failed with this exact symptom. The other candidates in `_CANDIDATE_CLFS` were skipped (`s`), meaning they raised `ValueError`/`KeyError`/`NotImplementedError` — i.e. those classifier names simply aren't implemented yet, which is a separate (and less serious) issue, since the spec only requires 3 total classifiers and Random Forest is mandatory.

## The fix (conceptually, before I see the source)

In `src/qubo_project/model.py`, `train()` and `predict()` should save directly to the exact path strings passed in (`model_path`, `metrics_json`, `predictions_csv`, `classif_stats_json`) — e.g. `joblib.dump(model, model_path)` — without routing them through any "strip directory, force into outputs/" helper. If you want the *CLI* entry point to default relative filenames into `outputs/`, do that remapping only in the `argparse`/CLI wrapper, before calling `train()`/`predict()` — not inside the function bodies themselves.

Send over `model.py` when you're ready and I'll confirm exactly where the remapping is happening and patch it.