Here's what was implemented — exactly the first three checklist items, nothing more:

**① `_to_output_path(filename)`** — creates `outputs/` if needed and joins `os.path.basename(filename)` onto it, matching the helper contract from `preprocessing.py`.

**② Chunked loading** — reads the normalized CSV in 131 072-row chunks, concatenates them into one `DataFrame`, then validates that `target_column` exists (raises a clear `ValueError` if not).

**③ Training/test hard split** — applies the exact formula from the spec:
```
M = total_samples - round(percTest * total_samples)
df_train = df.iloc[:M]   # first M rows
df_test  = df.iloc[M:]   # remaining rows
```

The stub `# TODO` comments mark where the Spearman/Q-matrix and alpha-search sections will go in the next steps. The CLI skeleton and `if __name__ == "__main__":` block are also present so the file is immediately runnable without errors.