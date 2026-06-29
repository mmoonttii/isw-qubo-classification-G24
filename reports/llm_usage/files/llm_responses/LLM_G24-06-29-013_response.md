Now I'll write the corrected module. The changes are surgical and documented inline:All three tests pass. Let me present the file.Here's a summary of every change made and why, so you can document it in your LLM log.

---

## Changes from the previous version

### 1. Factor-of-2 fix in `_optimize_qubo_neal` (correctness)

The spec's double-sum $\sum_{j} \sum_{k \neq j} x_j x_k |\rho_{jk}|$ counts every off-diagonal pair **twice**. The full-matrix product `x @ Q @ x` (used by the numpy fallback) honours that factor naturally. D-Wave/neal's `sample_qubo` API, however, takes an upper-triangle dict and counts each pair **once**, silently halving the independence penalty.

The fix is a single multiplication in the dict construction:

```python
# old (bug): val = float(Q[i, j])
val = float(Q[i, j]) if i == j else 2.0 * float(Q[i, j])
```

The smoke-test verified numerically that the old path produced `−1.13` where the correct value was `+0.70` — a completely different sign, not just a scaling issue.

### 2. `num_reads` raised from 1 → 200 in `_optimize_qubo_neal`

SA is stochastic. A single run has high variance on a dense 100+ variable QUBO. `num_reads=200` runs 200 independent trajectories; neal automatically returns the lowest-energy sample. This is the primary lever for the bonus point (lowest cost).

### 3. Numpy SA fallback improved

`n_steps` raised from `max(10_000, 150n)` to `max(50_000, 500n)`, and the solver now runs **5 independent restarts** keeping the global best. This keeps the fallback competitive when dwave-neal is not installed.

### 4. Everything else is unchanged

All function signatures, output file formats, JSON field names, CLI arguments, and output paths are byte-for-byte identical to the spec requirements.