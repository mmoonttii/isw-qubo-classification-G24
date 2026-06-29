No dataset needed — the failure is fully explained by the fixture parameters and a specific mathematical identity. Let me trace the exact cause.

## What the failure is telling us

```
feature_19: std = 0.9487
```

`0.9487` is not a random number. It is `sqrt(0.9) = sqrt(9/10)` to four decimal places. That is a precise algebraic fingerprint of exactly what went wrong.

## Deriving the formula

When `preprocessing.py` normalises a column that contains `k` NaN values out of `n_total` rows, the pipeline does this:

1. Welford's algorithm accumulates mean `µ` and sample variance (`ddof=1`) from the `n_valid = n_total − k` non-NaN values only.
2. NaN cells are filled with `µ` before z-scoring → `(µ − µ) / σ = 0.0` in the output.
3. Non-NaN cells become `(xᵢ − µ) / σ` in the output.

The output column therefore has `k` exact zeros and `n_valid` properly scaled values. Computing its **population** std (`ddof=0`) with mean = 0:

```
var_ddof0 = (1/n_total) × Σ yᵢ²
          = (1/n_total) × Σ [(xᵢ - µ)² / σ²]   (sum over non-NaN only)
          = M₂ / (n_total × σ²)
```

Substituting `σ² = M₂ / (n_valid − 1)` (the sample variance Welford computed):

```
var_ddof0 = M₂ / (n_total × M₂ / (n_valid − 1))
           = (n_valid − 1) / n_total
```

Therefore:

```
std_ddof0(normalised column) = sqrt( (n_valid − 1) / n_total )
```

## Verifying with the observed value

```
0.9487² = 0.9000 = 9/10
→  (n_valid − 1) / n_total = 9/10 = 108/120
→  n_valid − 1 = 108
→  n_valid = 109
→  k = 120 − 109 = 11 NaN values in feature_19
```

The RNG with `SEED=42` produced exactly 11 NaN values in `feature_19`. That is unlucky but plausible: with `NAN_FRACTION = 0.05` and `N_SAMPLES = 120` the expected count is 6 with standard deviation `sqrt(120 × 0.05 × 0.95) ≈ 2.4`, so 11 is roughly two standard deviations above the mean.

## Why `_STD_TOL = 0.05` does not cover it

The tolerance was sized to accommodate the `ddof=0` vs `ddof=1` gap alone:

```
1 − sqrt((n−1)/n) = 1 − sqrt(119/120) ≈ 0.004
```

But the formula above shows NaN imputation adds an independent term. The combined deviation is:

```
1 − sqrt((n_valid − 1) / n_total) = 1 − sqrt(108/120) ≈ 0.051
```

That is `0.001` past the tolerance boundary — a boundary case caused by unusually high random NaN count in one column. The tolerance was never sized to absorb NaN-imputation variance.

## The fix

The tolerance needs to cover the worst-case NaN count that the fixture can realistically produce. At three standard deviations above the mean, `k_max ≈ 6 + 3 × 2.4 ≈ 13`, giving:

```
max deviation = 1 − sqrt((120 − 13 − 1) / 120) = 1 − sqrt(106/120) ≈ 0.060
```

The cleanest fix derives `_STD_TOL` analytically from the fixture constants rather than hardcoding a number that was never explained:

```python
# _STD_TOL must cover two sources of deviation from 1.0:
#   1. ddof=0 vs ddof=1 mismatch:   ≈ 1 − sqrt((n−1)/n)       ≈ 0.004
#   2. NaN imputation zero-fill:    ≈ 1 − sqrt((n·(1−f)−1)/n) ≈ 0.030
#      where f = NAN_FRACTION (expected NaN rate per column)
# A 2× safety factor absorbs RNG variance above the expected NaN count.
_n_valid_expected = N_SAMPLES * (1.0 - NAN_FRACTION)
_STD_TOL = 2.0 * (1.0 - float(np.sqrt((_n_valid_expected - 1) / N_SAMPLES)))
```

With our constants this evaluates to `2.0 × (1 − sqrt(113/120)) ≈ 2.0 × 0.030 ≈ 0.060`, giving a comfortable margin without being so loose that it would fail to detect genuinely broken normalisation such as min-max scaling.