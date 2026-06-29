I have everything I need — the spec's Appendix is fully readable in the document provided. Here is a precise analysis.

---

## QUBO Cost Function Verification

### ✅ The matrix structure is correct

The spec defines the objective to minimize as:

$$f(\mathbf{x}) = -\left[\alpha \sum_j x_j|\rho_{Vj}| - (1-\alpha)\sum_{j \neq k} x_j x_k |\rho_{jk}|\right]$$

which is rewritten as $f(\mathbf{x}) = -\mathbf{x}^T Q_{spec} \mathbf{x}$ with:
- Diagonal of $Q_{spec}$: $+\alpha|\rho_{Vj}|$
- Off-diagonal of $Q_{spec}$: $-(1-\alpha)|\rho_{jk}|$

The code builds `Q_code = -Q_spec`:
- Diagonal of `Q_code`: $-\alpha|\rho_{Vj}|$ ✅
- Off-diagonal of `Q_code`: $+(1-\alpha)|\rho_{jk}|$ ✅

Since $\mathbf{x}^T Q_{code} \mathbf{x} = -\mathbf{x}^T Q_{spec} \mathbf{x} = f(\mathbf{x})$, **minimizing `x @ Q @ x` is exactly what the spec requires**. The numpy fallback does this correctly.

---

### 🐛 Critical bug: factor-of-2 error in the `neal` path

This is subtle but real. The spec's double sum $\sum_{j=1}^n \sum_{k \neq j}$ counts every pair $(j,k)$ **twice** (once as $(j,k)$ and once as $(k,j)$), so the correct full-matrix expansion is:

$$f(\mathbf{x}) = -\alpha \sum_j x_j |\rho_{Vj}| + \mathbf{2}(1-\alpha) \sum_{j < k} x_j x_k |\rho_{jk}|}$$

The **numpy fallback** computes `x @ Q @ x` using the full symmetric matrix, which correctly produces that factor of 2 on the off-diagonal. ✅

The **neal path** however does this:

```python
Q_dict = {
    (i, j): float(Q[i, j])
    for i in range(n)
    for j in range(i, n)   # ← upper triangle only
    ...
}
```

D-Wave/neal's `sample_qubo` computes $\sum_{i \leq j} Q_{dict}[i,j] \cdot x_i x_j$, which for the off-diagonal gives only **one** copy of each pair — effectively the formula:

$$E_{neal} = -\alpha \sum_j x_j |\rho_{Vj}| + \mathbf{1} \cdot (1-\alpha) \sum_{j < k} x_j x_k |\rho_{jk}|$$

The independence penalty is **half** what it should be. Neal is solving a subtly different problem from what the spec requires. The numpy SA is the correct implementation; neal is not.

**The fix is one line** in `_optimize_qubo_neal`:

```python
Q_dict = {
    (i, j): float(Q[i, j]) if i == j else float(2.0 * Q[i, j])
    for i in range(n)
    for j in range(i, n)
    if Q[i, j] != 0.0
}
```

This restores the factor of 2 so that `sample_qubo` minimizes the same objective as `x @ Q @ x`.

---

### ⚠️ Additional issue: `num_reads=1` is far too low

```python
response = neal.SimulatedAnnealingSampler().sample_qubo(
    Q_dict, num_reads=1, seed=seed   # ← only 1 SA run
)
```

Simulated annealing is stochastic — a single run has a high chance of getting stuck in a local minimum, especially for large dense Q matrices (n ≈ 145). This directly hurts the quality of the selected features.

---

### 💡 What to change to compete for the bonus point

The bonus is awarded for the **lowest QUBO cost value** on the professor's verification dataset. Here are the improvements in order of impact:

**1. Fix the factor-of-2 (required for correctness)**
As shown above — double the off-diagonal values passed to neal.

**2. Increase `num_reads` substantially**
```python
response = neal.SimulatedAnnealingSampler().sample_qubo(
    Q_dict, num_reads=200, seed=seed  # more independent restarts
)
```
Then pick the sample with the lowest energy: `response.first.sample` already does this, but only among the reads actually performed.

**3. Tune the SA schedule in the numpy fallback**
The current `n_steps = max(10_000, 150 * n)` is modest. For the verification dataset (1.5M+ rows, likely more features), increase it:
```python
n_steps = max(50_000, 500 * n)
```
Also consider starting from a warm start (flip bits greedily by gradient) rather than random initialization.

**4. (Optional, bigger impact) Use a better solver**
For dense QUBO of this size, `dwave-neal` with many reads is good, but if you want to push further, consider `qiskit-optimization` (QAOA), `scipy.optimize.differential_evolution`, or a tabu search. The spec says "the choice of algorithm must be documented" — changing it is explicitly allowed.

---

### Summary table

| Check | Result |
|---|---|
| Q matrix formula (diagonal) | ✅ Correct |
| Q matrix formula (off-diagonal) | ✅ Correct |
| Numpy SA objective | ✅ Correct — minimizes $f(\mathbf{x})$ as per spec |
| Neal QUBO objective | ❌ Off-diagonal halved — factor-of-2 bug |
| `num_reads` in neal | ⚠️ Set to 1, very poor optimization quality |
| Alpha bisection logic | ✅ Correct |
| Output files and JSON fields | ✅ Match spec exactly |