Looking at your three modules, the pipeline handles the ~1.5M-row target with three different strategies depending on the stage — worth understanding because they're not uniform.

## Stage 1 — `preprocessing.py`: true streaming (the only stage that really needs it)

This is the stage genuinely built for scale, because it's the only one that touches the *raw, unreduced* dataset (145+ columns × 1.5M rows).

- **Two-pass chunked I/O**: `pd.read_csv(..., chunksize=128_000)`. Pass 1 streams through the file to compute per-column statistics; pass 2 streams through again to apply normalization and write output incrementally (`mode="a"` append). At no point does the full raw CSV sit in memory — peak memory is bounded by chunk size (~128k rows), not dataset size.
- **Welford's online algorithm** for mean/variance: this is the key trick. Z-score normalization needs mean and std *before* you can normalize, but computing those normally requires holding all the data. Welford's algorithm updates mean/M2 incrementally per chunk (with the parallel/batch merge formula), so you get numerically stable statistics in one streaming pass instead of loading everything.
- Column-validity counts (for dropping sparse/zero columns) and target-value validation are accumulated the same way, chunk by chunk.

This is the part of the code doing the actual heavy lifting for 1.5M+ rows.

## Stage 2 — `feature_selection.py`: reads in chunks but materializes the full DataFrame

```python
chunks = []
for chunk in pd.read_csv(normalized_csv_path, chunksize=131072):
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)
```

This *reads* in chunks but then concatenates everything into one in-memory DataFrame. That's a deliberate (if implicit) trade-off, not an oversight: Spearman correlation and the QUBO cost matrix both need to see every row of every feature column simultaneously — you can't compute a rank correlation incrementally the way you can a mean. So this stage isn't streaming in the memory sense; it just avoids `pandas`' higher-overhead single-shot read.

The saving grace at scale: by this point the dataset is *already normalized* but still has all features (e.g. 1.5M × 145 float64 ≈ 1.7 GB) — large but generally workable on a normal machine, not append-only-file scale. The important thing to notice, though, is that the QUBO matrix Q itself is only **n × n where n = number of features** (~145), never m × m over samples — so the NP-hard optimization (simulated annealing via `neal` or the numpy fallback) doesn't scale with row count at all, only with feature count. Row count only affects how long the Spearman correlation computation takes.

## Stage 3 — `model.py`: operates on the already-reduced dataset

`train()` and `predict()` just do a plain `pd.read_csv()`. This is fine because by this point the feature-selection step has already cut columns down to ~20% of the original (e.g. 145 → ~29), so the in-memory footprint is a fraction of stage 2's. `RandomForestClassifier(..., n_jobs=-1)` parallelizes across CPU cores to keep training time reasonable at 1M+ rows.

## The honest picture

| Stage | Memory strategy | Bottleneck at 1.5M rows |
|---|---|---|
| preprocessing | true streaming, O(chunk) memory | none — scales indefinitely |
| feature_selection | chunked read → full in-memory concat | O(rows × features) memory; Spearman compute time |
| model train/predict | direct full read | O(rows × K) memory, K = selected features (small) |

So the project's scalability claim really rests on preprocessing being fully streaming, and on feature selection happening *after* normalization (so it isn't holding raw + normalized data simultaneously) and *before* the row count matters for the QUBO solve itself. If you wanted to push further (say, for the docente's 1.5M-row verification set), the main risk point to watch is stage 2's `pd.concat` — that's where you'd hit a memory ceiling first, not in preprocessing or training.