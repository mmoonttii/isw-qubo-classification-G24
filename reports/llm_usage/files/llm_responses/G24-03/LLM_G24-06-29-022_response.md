This is a single root cause propagating to all 54 tests. Let me trace it precisely through the fixture dependency chain.

## Diagnosis

### Where the exception is actually thrown

Every error points to the same file and the same location in the code. The `_setup` autouse fixture inside every test class eagerly reads the output CSV on disk:

```python
@pytest.fixture(autouse=True)
def _setup(self, preproc):
    self.norm_csv  = preproc["csv"]
    self.json_path = preproc["json"]
    self.df        = pd.read_csv(preproc["csv"])   # ← FileNotFoundError here
    self.features  = self.df.drop(columns=[TARGET])
```

`pd.read_csv()` raises `FileNotFoundError` when the file is absent. Since `_setup` is `autouse=True`, it runs before every single test — so one missing file wipes out the entire suite.

### Why the file is missing

The `preproc` session-scoped fixture itself **did not raise an exception** (otherwise pytest would report a fixture setup error, not a `FileNotFoundError` inside the test). It returned its dict successfully. This means `fit_normalize()` was called and finished without crashing — but it did **not write its output to the absolute path the fixture passed in**.

The fixture passes a full absolute path rooted under pytest's temp dir:
```
/private/tmp/pytest-of-monti/pytest-0/qubo_pipeline0/normalized.csv
```

The implementation almost certainly writes to a **CWD-relative** path instead — most likely `outputs/normalized.csv` — by either hardcoding the `outputs/` prefix or by using `os.path.basename()` on the `normalized_csv` parameter rather than the full path. This is consistent with the CLI spec's `Output atteso: outputs/normalized.csv`, which students often replicate inside the function body instead of only in the CLI wrapper.

### Why all 54 tests error with identical messages

The cascade has two layers:

**Layer 1 — session scope amplification.** `preproc` is `scope="session"`, so once its product (the CSV file) is missing, every downstream fixture (`feat_sel`, `model_outputs`, `prediction_outputs`) also has an absent file to build on. Every fixture in the entire chain silently "succeeds" but returns paths to files that don't exist.

**Layer 2 — eager loading in `_setup`.** The autouse fixture in every test class immediately calls `pd.read_csv()` on the path rather than just storing it for later. This design means there is no test that can survive fixture setup — not even `test_output_csv_is_created`, whose entire job is to check whether the file exists. It errors during setup before its own `assert Path(...).exists()` body ever runs.

### Summary

| Problem | Location | Effect |
|---|---|---|
| `fit_normalize()` writes to wrong path | Student's `preprocessing.py` | `normalized.csv` absent at temp path |
| `_setup` calls `pd.read_csv()` eagerly | `_setup` autouse fixture in test file | Crashes before any test body runs |
| Session scope | `preproc`, `feat_sel`, `model_outputs` | One failure poisons the entire suite |

The architectural tension is this: `_setup` was designed to pre-load data as a convenience, but that convenience turns every missing-file situation from a clean test **FAIL** (one test reports "file not found, as expected") into a suite-wide test **ERROR** (pytest never reaches the assertion body). The existence-check test and the content-check tests have incompatible preconditions, but `_setup` treats them identically.