Confirmed exactly what I flagged last time. `gui.py` calls `train()`/`predict()` with **bare filenames** and relies on the internal `_to_output_path()` redirect to land them in `outputs/`:

- Line 1296–1297: `model_path = "model.joblib"`, `metrics_json = "training_metrics.json"`
- Line 1451–1452: `predictions_csv = "predictions.csv"`, `classif_stats_json = "classification_stats.json"`
- Then in each case it reads back via `_outputs_dir() / "the_same_name.ext"`.

So this confirms the situation exactly as flagged: **if you apply the fix as I described it last time (strip `_to_output_path()` out of `train()`/`predict()`), `gui.py` breaks** — bare filenames like `"model.joblib"` would then be written relative to whatever the process's cwd happens to be when Streamlit launches, not `outputs/`, and `_run_training`/`_run_prediction` would then fail to find the file at `_outputs_dir() / "model.joblib"`.

## The actual fix, updated for this

Don't remove the resolution logic — move it out of `train()`/`predict()` and apply it consistently at every call site (CLI **and** GUI), the same way the CLI block already does at lines 709–710/719–720.

**In `model.py`**, replace the two `_to_output_path()` calls inside `train()` (lines 339–340) and `predict()` (lines 422–423) with plain `Path(...)`, so the functions honor whatever path they're given, literally — this is the part from before, unchanged.

**In `gui.py`**, do the remapping yourself, right where the CLI already does it. Cleanest is to import `_to_output_path` (or reproduce it — it's already duplicated as `_outputs_dir()`/`_resolve_outputs_dir()` in `gui.py`) and wrap the bare filenames before calling `train()`/`predict()`:

```python
# _run_training(), replacing lines 1296-1297
model_path       = str(_outputs_dir() / "model.joblib"),
metrics_json     = str(_outputs_dir() / "training_metrics.json"),
```

```python
# _run_prediction(), replacing lines 1451-1452
predictions_csv    = str(_outputs_dir() / "predictions.csv"),
classif_stats_json = str(_outputs_dir() / "classification_stats.json"),
```

`gui.py` already has `_outputs_dir()` defined (line 437) and already reads results back from exactly those same paths afterward, so this is a one-line change per call site — no new helper needed, and behavior for the GUI stays identical to today.

## Net effect

| Caller | Before | After |
|---|---|---|
| `pytest` / evaluator, explicit path (e.g. `tmp_path/rf_model.joblib`) | silently redirected → test fails | honored literally → test passes |
| CLI (`--out-model model.joblib`) | resolved into `outputs/` at dispatch (line 709) | unchanged |
| `gui.py` (bare `"model.joblib"`) | resolved into `outputs/` inside `train()` | resolved into `outputs/` at the call site instead — same result |

Same pattern applies to `preprocessing.py` and `feature_selection.py` if they have equivalent internal `_to_output_path()` calls inside their public functions — worth checking those two next if you haven't already, since the comment block at the top of `model.py` (lines 82–94) says it's mirroring their exact helpers.