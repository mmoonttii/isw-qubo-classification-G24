Good — no dedicated pytest skill, so I'll go straight to writing the file. Let me now craft the full test suite.Let me do a quick sanity check on what we produced:Syntax is clean. 25 tests across 3 classes, 702 lines. Let me present the file.Here's what the file contains and why each decision was made:

---

## Structure at a glance

| Layer | Count | Purpose |
|---|---|---|
| Session-scoped fixtures | 4 | `session_dir`, `raw_csv`, `preproc`, `feat_sel` – each expensive step runs exactly once |
| `TestPreprocessing` | 9 tests | Spec §7 + §11.1 JSON schema |
| `TestNormalization` | 4 tests | Spec §7.2 z-score math |
| `TestFeatureSelection` | 11 tests | Spec §8 QUBO vector + §11.2 JSON + CSV consistency |
| **Total** | **25 tests** | |

---

## Key design decisions

**Deterministic fixture.** The dummy CSV uses `np.random.default_rng(SEED)` so the dataset is bit-identical every run. Sparse columns are engineered with exactly 2 non-zero values (98.3% zeros), which reliably exceeds the 95% removal threshold without being fragile.

**Session scope.** All four fixtures are `scope="session"`: `fit_normalize` and `select_features` run once per `pytest` invocation, not once per test. This keeps the full suite fast even when `alpha_computations` is higher.

**Tolerance design for z-score.** `_MEAN_TOL = 1e-9` (very tight) catches any un-centred output. `_STD_TOL = 0.05` is deliberately generous: it accepts both `ddof=0` (sklearn default → exactly 1.0) and `ddof=1` (numpy/pandas default → `sqrt((n-1)/n) ≈ 0.996`), while still rejecting min-max scaling (which would give std ≈ 0.28).

**Cross-referencing outputs.** Several tests compare two artefacts against each other (e.g. JSON `n_kept_features` vs CSV column count, JSON `selected_feature_names` vs train CSV columns, train vs test column order). This catches implementation bugs where different parts of the code fall out of sync.

**Informative failure messages.** Every `assert` includes a message that names the exact discrepancy, making it possible to diagnose failures without re-reading the spec.

---

Step 2 will cover `model.py` – `train()` and `predict()` – including checking that the saved `.joblib` model is loadable, that predictions produce the required CSV columns, and that all classification-metric keys are present in the JSON.