Let me read every `_setup` fixture and the tests that depend on them before touching anything.Now I have a precise view of all five `_setup` fixtures. Let me catalogue the exact problem in each one:

| Class | What `_setup` does | What breaks |
|---|---|---|
| `TestPreprocessing` | `pd.read_csv(preproc["csv"])` | Crashes before `test_output_csv_is_created` can run its own `Path.exists()` assertion |
| `TestNormalization` | `pd.read_csv(preproc["csv"])` | FileNotFoundError in fixture setup → ERROR, not FAILED |
| `TestFeatureSelection` | `open(feat_sel["json"])` | Same — crashes before `test_all_output_files_are_created` can assert |
| `TestModelTraining` | `open(model_outputs["metrics_json"])` | Crashes before `test_training_metrics_json_is_created` can assert |
| `TestModelPrediction` | `pd.read_csv(...)` twice + `open(...)` | Same cascade |

The fix in each case is the same: add a `Path.exists()` guard that calls `pytest.fail()` with an explicit message **before** any I/O. An unhandled exception in a fixture produces an `ERROR`; `pytest.fail()` produces a `FAILED`. Every test in the class then reports `FAILED` with a clear prerequisite message instead of an opaque `ERROR: FileNotFoundError`.

Five targeted `str_replace` operations, one per class.Now verify the file and confirm every `_setup` contains a guard:All five guards are in place. Let me present the final file.All three fixes are now complete. Here is a precise account of what Fix 3 changed and why each decision was made.

---

## What changed

Five `_setup` autouse fixtures, each gained a `Path.exists()` guard with `pytest.fail()` immediately before the first I/O call. Everything else — the loaded DataFrames, the parsed JSON dicts, the attribute names — is identical.

---

## The mechanics of the fix

The distinction that matters to pytest is where an exception is raised:

| Exception raised in… | pytest status | Visible to developer as |
|---|---|---|
| Fixture body (unhandled) | `ERROR` | Infrastructure failure — test never ran |
| `pytest.fail()` in fixture | `FAILED` | Assertion failure — implementation is wrong |
| Test body (any exception) | `FAILED` | Assertion failure — implementation is wrong |

Before Fix 3, a missing output file caused `pd.read_csv()` or `open()` to raise `FileNotFoundError` inside the fixture — unhandled, so pytest marked every single test as `ERROR`. A student looking at an `ERROR` might suspect a problem with the test code or environment, not with their own implementation. With `pytest.fail()`, the same situation produces `FAILED` tests, which correctly communicate "the implementation did not produce the expected file."

---

## Why a guard rather than pure lazy loading

Pure lazy loading (removing all data loading from `_setup` and moving it into each test method) would also work, but it has two costs: every test that needs a DataFrame would need two to three extra lines of boilerplate for loading, and classes like `TestNormalization` — where every single test needs the same DataFrame — would reload it from disk on each of their four tests. The guard pattern retains the efficient single-load-per-test design while converting the failure mode from `ERROR` to `FAILED`.

---

## Why `TestModelPrediction` gets two guards

`predict()` writes two independent files: `predictions_csv` and `stats_json`. A partially broken implementation could produce one but not the other. Guarding only the first would mean a missing `stats_json` still crashes the fixture with an unhandled `FileNotFoundError`. The two independent guards make sure both failure modes convert to `FAILED` with messages that name the specific missing file and the specific existence-check test that should report the root cause.