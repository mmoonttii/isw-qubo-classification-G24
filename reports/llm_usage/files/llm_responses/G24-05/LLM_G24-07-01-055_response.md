Yes — both files have the exact same bug as `model.py`, just currently invisible for a different reason.

## What I found

**`preprocessing.py`**: `fit_normalize()` itself calls `_to_output_path()` internally on `normalized_csv` and `outInitalRes_json` (lines 233–234), before ever touching the CSV/JSON writes. Same silent "keep only the basename, force into `outputs/`" behavior as `model.py`'s `train()`.

**`feature_selection.py`**: `select_features()` does the identical thing to all four of its output params — `reducedTrain_csv`, `reducedTest_csv`, `output_ottim_csv`, `output_json` (lines 415, 422, 429, 438).

So it's the same root cause across all three modules, not something specific to `model.py`.

## Why this didn't show up as a test failure yet

It's masked, not absent. Two different reasons:

1. **In `test_pipeline.py`**: the `preproc` and `feat_sel` fixtures deliberately pass **bare filenames** (`_TEST_PREFIX + "normalized.csv"`, etc.) instead of `tmp_path`-style full paths — the fixture docstrings even say so explicitly ("Path contract... output is remapped... Fix: pass bare filenames"). That was written as a workaround for this exact behavior, not proof the behavior is correct. Only `model.py`'s test used a real `tmp_path`, which is why only `train()` failed.
2. **In the CLI**: `preprocessing.py`'s and `feature_selection.py`'s `__main__` blocks pass the raw `args.out_data`/`args.out_train`/etc. straight into the function (lines 543–544, 488–493) and let the function do the internal remapping. This is different from `model.py`'s CLI, which already pre-resolves the path *before* calling `train()`/`predict()` (lines 709–710, 719–720) — that's actually the correct pattern, the other two just don't follow it.

## Confirmed in `gui.py` too

Same call pattern as `train()`/`predict()` — bare filenames, relying on internal resolution:
```python
# line 753-754
fit_normalize(..., normalized_csv="normalized.csv", outInitalRes_json="preprocessing_result.json")
# line 935-938
select_features(..., reducedTrain_csv="training_reduced.csv", reducedTest_csv="test_reduced.csv", ...)
```

## Net conclusion

This is one bug pattern, present in all three modules, currently hidden by workarounds in the test fixtures and CLI blocks rather than fixed. The moment the evaluator calls `fit_normalize()` or `select_features()` directly with its own path (exactly like it apparently does for `train()`), you'll get the same class of failure you just saw for the classifiers.

## Coordinated fix (same shape everywhere)

1. **In `fit_normalize()` and `select_features()`**: remove the internal `_to_output_path()` calls, use `Path(...)` directly — functions honor whatever path they're given, literally.
2. **In each module's CLI `__main__` block**: pre-resolve args through `_to_output_path()` *before* calling the function (mirroring what `model.py`'s CLI already does correctly) — `preprocessing.py` and `feature_selection.py` currently don't do this and need that line added.
3. **In `gui.py`**: resolve all four bare filenames (`normalized.csv`, `preprocessing_result.json`, `training_reduced.csv`, `test_reduced.csv`, `optimizations.csv`, `feature_selection_result.json`, plus the two from `train()`/`predict()`) through the local `_outputs_dir()` at each call site, same as I described for training/prediction last turn.
4. **In `test_pipeline.py`**: the `preproc`/`feat_sel` fixtures currently pass bare filenames *because* of this bug — once fixed, they should switch to passing fully resolved `_project_outputs_dir() / name` paths (matching what the `feat_sel`/`model_outputs` fixtures effectively need anyway), so the tests actually exercise the "honor the literal path" contract instead of routing around it.

Want me to make all four edits now? I can do it in one pass since it's the same fix applied consistently.