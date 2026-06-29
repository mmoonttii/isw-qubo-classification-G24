Let me first view the exact current state of both model fixtures before planning.I have everything I need. The issue is clear: `model_outputs` and `prediction_outputs` still use `session_dir` for their output paths. Since `model.py`'s `train()` and `predict()` have no `_to_output_path()` — they call `Path(path).parent.mkdir(parents=True, exist_ok=True)` and write to exactly the path given — this would technically work. But it creates an inconsistency: three of five pipeline fixtures now write to a predictable `outputs/` location, while two write to an ephemeral temp dir that disappears between runs. Fix 2 closes that gap.

Two targeted `str_replace` edits, one per fixture.

**Edit 1 — `model_outputs`**: drop `session_dir`, use `_project_outputs_dir()` and `_TEST_PREFIX`:**Edit 2 — `prediction_outputs`**: same treatment:Verify the file is clean and confirm `session_dir` is now only referenced where it belongs:Exactly the right state. Let me present the updated file.The `session_dir` grep confirms the fix is exactly scoped: three lines remaining, all inside `raw_csv`. Every pipeline fixture now has a clean, dependency-free signature.

---

## What changed and why

### The core asymmetry this fix resolves

After Fix 1, the five pipeline fixtures split into two incompatible camps:

| Fixture | Output location | Survives re-run? |
|---|---|---|
| `preproc` | `outputs/test_pipeline_*.csv` | ✓ |
| `feat_sel` | `outputs/test_pipeline_*.csv` | ✓ |
| `model_outputs` | `session_dir` (temp) | ✗ |
| `prediction_outputs` | `session_dir` (temp) | ✗ |

The asymmetry matters in practice: if a test run fails mid-way through and you re-run only the failing tests, `session_dir` is a fresh temp directory — pytest creates a new one each session. The preprocessing and feature selection artefacts would be found in `outputs/` as before, but the model and predictions would need to be regenerated from scratch (even if training is the slow step). Worse, any test that checks the *existence* of output files would look in the old `session_dir` path which no longer exists.

### Why `_project_outputs_dir()` with full paths (not bare filenames)

Unlike `preprocessing.py` and `feature_selection.py`, `model.py` has no `_to_output_path()`. So we pass complete absolute paths — `out_dir / (_TEST_PREFIX + "model.joblib")` — directly. No bare-filename trick is needed; `Path(model_path).parent.mkdir(parents=True, exist_ok=True)` in `train()` will create `outputs/` if it doesn't already exist, then write exactly where told.

### `session_dir` scope after both fixes

`session_dir` is now only a write target for `raw_csv` (the *input* CSV the pipeline reads from). Inputs live in temp space — they are synthetic test data with no real-world meaning. Outputs live in `outputs/` — they are the artefacts the spec defines, the evaluator expects, and a developer would want to inspect after a failure.