Let me read all three files carefully before forming any plan.Now I have a complete picture. Let me map out every layer of the problem precisely.

---

## Layer 1 — The `_to_output_path` strip (root cause)

Both `preprocessing.py` and `feature_selection.py` share the same helper:

```python
def _to_output_path(user_path: str) -> Path:
    outputs_dir = _resolve_outputs_dir()
    filename = Path(user_path).name  # ← strips ALL directory components
    return outputs_dir / filename
```

`Path(user_path).name` on any path — including a full `tmp_path` absolute path — returns only the bare filename. The entire directory tree the caller passes in is silently discarded:

```
Path("/private/tmp/pytest-xxx/qubo_pipeline0/normalized.csv").name
→  "normalized.csv"
→  final write path: <project_root>/outputs/normalized.csv
```

Both `fit_normalize` and `select_features` run every output parameter through this function immediately on entry. `model.py` is architecturally different — `train()` and `predict()` have **no** `_to_output_path` at all; they call `Path(model_path).parent.mkdir(parents=True, exist_ok=True)` and then write to the exact path given.

---

## Layer 2 — The fixture return value is stale

The `preproc` session fixture returns:

```python
return {"csv": str(session_dir / "normalized.csv"), ...}
```

`fit_normalize()` never raised an exception, so pytest considers the fixture successful. But the actual file was written to `<project>/outputs/normalized.csv`. The path inside the returned dict points to a file that was **never created**. The fixture has no way to know this happened — it passes the path in and trusts the callee honoured it.

This stale path is then propagated through the entire session:
- `feat_sel` tries to read `session_dir/normalized.csv` as its input → would fail there too, but
- All `_setup` autouse fixtures also try to open it independently

---

## Layer 3 — Eager `pd.read_csv()` in every `_setup` converts fixture ERRORs into a suite-wide flood

Every test class has an `autouse` fixture that immediately calls `pd.read_csv()`:

```python
@pytest.fixture(autouse=True)
def _setup(self, preproc):
    self.df = pd.read_csv(preproc["csv"])   # ← raises FileNotFoundError here
    self.features = self.df.drop(columns=[TARGET])
```

`pd.read_csv()` raises at **fixture setup time**, before pytest enters any test body. This converts a missing-file situation — which should produce one clean `FAILED` in `test_output_csv_is_created` — into a fixture `ERROR` on every test in every class. Since this `_setup` is `autouse=True` and session-level fixtures feed it, the failure cascades across all 50 tests simultaneously.

This also creates a logical contradiction: `test_output_csv_is_created` checks whether the file exists by calling `assert Path(self.norm_csv).exists()`, but `_setup` has already crashed trying to open that very file before the test body runs.

---

## Summary of what needs to change

| Problem | Location in test file | Root trigger |
|---|---|---|
| Fixtures pass full `tmp_path` absolute paths to functions that strip directories | `preproc`, `feat_sel` fixtures | `_to_output_path` in implementation |
| Returned paths point to non-existent files | `preproc["csv"]`, `feat_sel[*]` dict values | Consequence of above |
| `pd.read_csv()` at fixture setup time crashes before any test runs | Every `_setup` autouse fixture | Consequence of stale paths |

### The three-part fix strategy

**Fix 1 — Path alignment for `preprocessing.py` and `feature_selection.py`:** The fixtures must pass bare filenames (`"test_normalized.csv"`, etc.) — no directory component — to `fit_normalize()` and `select_features()`. They must then compute where `_to_output_path` actually wrote the files (the project's `outputs/` directory) and return those real paths. This exactly mirrors how the CLI works.

**Fix 2 — Path handling for `model.py`:** Since `train()` and `predict()` respect whatever path they receive and create parent directories themselves, `model_outputs` and `prediction_outputs` fixtures can write to any path. The simplest consistent choice is to also write into `outputs/` with test-prefixed filenames.

**Fix 3 — Lazy loading in `_setup`:** `_setup` should store only **paths**, never call `pd.read_csv()`. Tests that need the DataFrame should load it themselves. Tests that only check file existence should use `Path.exists()`. This eliminates the cascade: if a prerequisite file is absent, exactly one test fails cleanly with a descriptive assertion error, not all fifty with an opaque `FileNotFoundError`.