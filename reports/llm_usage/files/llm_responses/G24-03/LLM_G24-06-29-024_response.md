Let me view the precise sections I need to change before touching anything.I have everything I need. Now I'll make three precise, surgical edits.

**Edit 1 — add `_project_outputs_dir()` helper**, inserted between the constants block and the fixtures header:**Edit 2 — fix the `preproc` fixture**: drop `session_dir` from signature, pass bare filenames, return real `outputs/` paths:**Edit 3 — fix the `feat_sel` fixture**: drop `session_dir`, pass bare filenames for outputs, keep the full path for the input (which `select_features` reads directly without remapping):Now verify the file is syntactically valid and that the three changed sections look exactly right:Everything looks exactly right. Let me present the final file.Three changes, clearly scoped.

---

## What changed and why

### 1. `_TEST_PREFIX = "test_pipeline_"`

All output files passed to the pipeline functions are now named `test_pipeline_normalized.csv`, `test_pipeline_train_reduced.csv`, etc. This is necessary because `_to_output_path()` strips directory components — so the filename is the only part the test can control. The prefix prevents silent collisions if someone runs the CLI at the same time the test suite is running: a production `normalized.csv` and a test `test_pipeline_normalized.csv` coexist in `outputs/` without interfering.

### 2. `_project_outputs_dir()`

This helper replicates the walk-upward logic of `_resolve_outputs_dir()` from the implementation, starting from `src/qubo_project/` and traversing parent directories until it finds an existing `outputs/` subdirectory, then falling back to `_PROJECT_ROOT / "outputs"`. The test fixtures need this to know — without importing any private symbol — where the pipeline functions will actually write their files.

### 3. `preproc` and `feat_sel` fixtures

Both drop `session_dir` from their signatures and now follow the correct two-step pattern:

```
call pipeline function with bare filename  →  resolve real path via _project_outputs_dir()
```

The critical asymmetry in `feat_sel` is preserved and documented: `normalized_csv` is an **input** that `select_features()` reads directly with `pd.read_csv()`, so it receives the full `preproc["csv"]` path. The four output parameters receive bare filenames, which `_to_output_path()` will remap into `outputs/`. `session_dir` is kept and still used by `raw_csv`, `model_outputs`, and `prediction_outputs` — those fixtures are untouched because `model.py` respects whatever path it is given, making Fix 2 and Fix 3 separate, independent concerns.