Good, I have a clear picture of the spec. Now let me build the complete solution.I've now thoroughly read the spec. Let me build the complete, production-ready solution.Now I have everything I need. Let me create the file.All green. Now copy the final file to outputs and present it.Everything tested and passing. Here is a full breakdown of all decisions made.

---

## 1. `requirements.txt` entries for this module

```
pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.4.0
```

`pandas` and `numpy` do all the heavy lifting here. `scikit-learn` is not used in `preprocessing.py` itself (Z-score is implemented from scratch for chunk-compatibility), but you'll need it for the later modules and its `StandardScaler` can be used as a consistency-check reference. Pin minor-version floors, not exact versions, to keep the project usable with future bugfix releases.

---

## 2. Key design decisions explained

### Two-pass architecture for 1.5 M+ rows

The function deliberately **never loads the full file into memory**. It instead streams the CSV in 128 000-row chunks:

- **Pass 1** — statistics only. For each chunk, accumulate valid-cell counts and update the mean/variance using the **parallel (batch) form of Welford's algorithm**. This is numerically identical to computing mean and population variance in a single pass over all rows, but uses O(chunk) memory instead of O(N).
- **Pass 2** — write. Apply the frozen `mean`/`std` to each chunk and append to the output CSV. The header is written only on the first chunk.

### Column filtering logic

A column is dropped when `(non-NaN AND non-zero values) / total_rows < minPercValid`. This implements the spec's "missing values or zeros" criterion exactly: a column with 94 % NaNs *and* a column with 94 % zeros are both considered "almost empty" and dropped.

### NaN imputation → mean (i.e. 0 in normalised space)

NaN cells are filled with the **training-set column mean** before normalisation. After z-scoring, those cells become exactly `0.0`. This is the correct, leak-free strategy: the same frozen `mean`/`std` can be serialised and reapplied to any future chunk without re-fitting.

### `_to_output_path` — no absolute paths

The helper walks up the directory tree from `__file__` to find an `outputs/` sibling directory (the repo root). If none is found it falls back to `./outputs`. Only the **basename** of the caller-supplied filename is used, so paths like `../../../etc/passwd` cannot escape the outputs directory.

### Z-score for constant columns

If `std == 0` (constant feature that survived the filter), the safe divisor is set to `1.0`, producing `(x − mean) / 1 = 0` for all rows. This avoids a `ZeroDivisionError` without silently dropping the column.