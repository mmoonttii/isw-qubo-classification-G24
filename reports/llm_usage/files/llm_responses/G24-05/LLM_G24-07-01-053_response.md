Found it. Lines 339–340 in `train()`:

```python
model_path = _to_output_path(model_path)
metrics_json = _to_output_path(metrics_json)
```

`_to_output_path()` (defined at line 118) does exactly what the traceback showed:

```python
def _to_output_path(user_path: str) -> Path:
    ...
    filename = Path(user_path).name   # ← keeps ONLY the basename
    return outputs_dir / filename     # ← always forces it into <repo>/outputs/
```

So when the test calls `train(..., model_path=str(tmp_path / "rf_model.joblib"), ...)`, `train()` throws away the `tmp_path` directory, keeps just `rf_model.joblib`, and writes it to `<project_root>/outputs/rf_model.joblib` instead — exactly matching the `[train] Model saved →` line in your log. The test then checks `tmp_path/rf_model.joblib`, finds nothing, and fails. Same root cause for all three failures (random_forest, logistic_regression, gradient_boosting) — it's not classifier-specific, it just happens to hit every classifier that reaches step 4 of `train()`.

Let me check `predict()` too, since it likely has the same pattern and would fail the same way if the evaluator calls it directly:Confirmed — `predict()` has the identical pattern at lines 422–423: `predictions_csv` and `classif_stats_json` also get silently rerouted into `outputs/` via `_to_output_path()`. It'll pass the CLI tests (which pass bare filenames anyway) but would fail the same way if the evaluator or a test calls `predict()` directly with an explicit path, e.g. a `tmp_path`.

## Root cause, in one sentence

`_to_output_path()` — meant only to make bare CLI filenames resolve into `outputs/` — is being applied unconditionally inside the library functions `train()` and `predict()` themselves, so **any** caller-supplied path (even a full absolute path from a test or the evaluator) gets silently reduced to its basename and forced into `<repo>/outputs/`.

## The fix

Move the `_to_output_path()` calls out of `train()`/`predict()` entirely — they're already applied correctly at the CLI dispatch site (lines 709–710, 719–720). Inside the functions, just use the paths exactly as given:

```python
# in train(), replace lines 339-340:
model_path = Path(model_path)
metrics_json = Path(metrics_json)
```
```python
# in predict(), replace lines 422-423:
predictions_csv = Path(predictions_csv)
classif_stats_json = Path(classif_stats_json)
```

That keeps the CLI behavior identical (still resolves bare filenames into `outputs/`, since that remapping now happens once, at the CLI boundary) while making the two functions honor whatever path a direct caller — pytest, the evaluator, or your own `gui.py` — actually passes in, satisfying §11.3's contract that `model_path`/`metrics_json`/`predictions_csv`/`classif_stats_json` are literal destination paths.

One thing worth double-checking once you apply this: if `gui.py` calls `train()`/`predict()` with bare filenames expecting them to land in `outputs/` automatically (per the comment at lines 82–94), you'll want `gui.py` itself to do the `_to_output_path()` remapping before calling `train()`/`predict()`, the same way the CLI block does — otherwise the GUI's bare-filename convenience breaks.