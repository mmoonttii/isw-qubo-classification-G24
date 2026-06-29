Good. The insertion point is clear: after the `total_rows == 0` guard (line 274) and before the variance finalisation (line 276). Now I'll apply the edit and verify it.Test 3 found a real edge case worth fixing: `0.5` rounds to `0` via `int()`, slipping through as a false valid. The fix is to check the raw float values directly instead of casting to `int` first.Also need to update the type annotation and the `sorted()` call in the logger so they work with floats too:5/5. Copy updated file to outputs.Here is the exact block to insert, followed by the explanation of every decision.

---

### Block to insert

Place it after the `if total_rows == 0` guard and before `# Finalise variance`:

```python
    # ------------------------------------------------------------------
    # TARGET VALIDATION — confirm binary {0, 1} values across all chunks
    # ------------------------------------------------------------------
    target_unique: set = set()  # may hold int or float before validation

    target_iter = pd.read_csv(
        input_path,
        usecols=[target_column],
        chunksize=_CHUNK_SIZE,
        low_memory=False,
    )

    for chunk in target_iter:
        coerced = pd.to_numeric(chunk[target_column], errors="coerce").dropna()
        target_unique.update(coerced.unique().tolist())

    # Accept only values that are numerically equal to 0 or 1 (not merely
    # int-truncatable to them, e.g. 0.5 must not silently pass).
    unexpected = {v for v in target_unique if v not in (0, 1)}
    if unexpected:
        raise ValueError(
            f"Target column '{target_column}' must contain only binary values "
            f"(0 and 1), but found unexpected values: {sorted(unexpected)}"
        )

    logger.info(
        "Target validation passed: unique values in '%s' = %s",
        target_column,
        sorted(int(v) for v in target_unique),
    )
```

---

### Design decisions and why each one matters

**Separate dedicated chunk iterator with `usecols=[target_column]`** — this reads only the single target column from disk, so pandas skips parsing all feature columns entirely. On a 1.5 M-row, 145-column file that is roughly a 145× I/O reduction for this pass. It also keeps the block self-contained and structurally parallel with Pass 1 and Pass 2.

**`coerced.unique().tolist()` instead of `int(v) for v in ...`** — the original draft cast each value to `int()` before accumulating, which meant `int(0.5) == 0` would silently pass. Keeping the raw Python floats in `target_unique` and then comparing with `v not in (0, 1)` delegates the equality check to Python's `==` operator, which correctly distinguishes `0.5 != 0`. The `int()` cast is deferred to the INFO log line only, where the values are already known valid.

**Set membership via `v not in (0, 1)` rather than `target_unique - {0, 1}`** — the set-difference form works perfectly for integers but fails silently for floats in Python because `{0.0, 1.0} - {0, 1}` is empty (Python's `int`/`float` equality). The comprehension form is unambiguous regardless of the numeric type.

**`sorted(unexpected)` in the error message** — produces a deterministic, human-readable list (`[0.5, 2]`) that makes the problem immediately actionable without the caller needing to inspect the raw data.