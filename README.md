# NumCompute
A high-performance scientific computing toolkit built with Python and NumPy for Assignment 2.1.

## Sorting, ranking and statistics (`sort_search`, `rank`, `stats`)

These three modules provide vectorised sorting / selection / search, ranking with ties and percentiles, and descriptive statistics. They depend only on NumPy.

### Quick start

```bash
pip install numpy pytest
pytest                                   # run all unit tests
python benchmark/bench_member_b.py       # loop vs vectorised timings -> benchmark/results_member_b.md
```

The notebook `demo/demo_member_b.ipynb` walks through every function below.

```python
import numpy as np
from numcompute import sort_search, rank, stats

scores = np.array([85, 92, 78, 92, 60])

sort_search.stable_sort(scores, descending=True, return_indices=True)
# (array([92, 92, 85, 78, 60]), array([1, 3, 0, 2, 4]))   ties keep their original order

sort_search.topk(scores, 3, return_indices=False)        # array([92, 92, 85])
sort_search.quickselect(scores, 2)                      # 85  (3rd smallest, 0-based k)
sort_search.binary_search([60, 78, 85, 92, 92], 80)     # (2, False)  insertion index, found?

rank.rank(scores, method="average")                     # array([3. , 4.5, 2. , 4.5, 1. ])
rank.rank(scores, method="dense")                       # array([3, 4, 2, 4, 1])
rank.percentile(scores, [25, 50, 75])                   # array([78., 85., 92.])

stats.mean(scores)                                      # 81.4
stats.std(scores, ddof=1)                               # 13.297...
stats.histogram(scores, bins=4)
# (array([1, 0, 1, 3]), array([60., 68., 76., 84., 92.]))

grades = np.array([[85, 70, 90],
                   [92, np.nan, 88],
                   [78, 65, 95]])                       # students x subjects, one missing exam
stats.mean(grades, axis=0)                              # array([85., nan, 91.])
stats.mean(grades, axis=0, nan_policy="omit")           # array([85. , 67.5, 91. ])
```

### Conventions

**Axis.** `axis=None` reduces over all elements and returns a Python `float`. `axis=k` removes dimension `k`: for data of shape `(n_samples, n_features)`, `axis=0` gives one value per feature (shape `(n_features,)`). `rank` and `topk` work along an axis and keep the other dimensions.

**NaN policy.** Functions in `stats` and `rank.percentile` accept `nan_policy`:

| value | behaviour |
|---|---|
| `'propagate'` (default) | any NaN in a slice makes that result NaN |
| `'omit'` | NaNs are ignored; an all-NaN slice gives NaN (no warning) |
| `'raise'` | `ValueError` if any NaN is present |

`rank.rank` gives NaN a rank of NaN; `topk` never prefers NaN; `histogram` ignores NaN; `quickselect` and `RunningStats.update` raise `ValueError` on NaN.

**Errors.** Invalid input raises an informative exception: `TypeError` for wrong types (e.g. non-integer `k`, non-numeric data), `ValueError` for invalid values (e.g. `k` out of range, `q` outside [0, 100], empty input, unknown `method`), `AxisError` for an out-of-range `axis`.

### API reference

#### `numcompute.sort_search`

| Function | Description | Time |
|---|---|---|
| `stable_sort(values, axis=-1, descending=False, return_indices=False)` | Stable sort; equal values keep their original order, also when descending | O(n log n) |
| `multikey_sort(X, keys, descending=False, return_indices=False)` | Sort rows of a 2-D array by several columns (`keys[0]` is the primary key); `descending` may be a list, one flag per key | O(m · n log n) |
| `topk(values, k, largest=True, return_indices=True, axis=-1, sorted=True)` | k largest/smallest values via `np.argpartition`; `0 <= k <= n` | O(n + k log k) |
| `quickselect(values, k, seed=None)` | k-th smallest (0-based) of a 1-D array; random pivot, 3-way partition; input is not modified | expected O(n) |
| `binary_search(sorted_array, x, side='left', check_sorted=False)` | Insertion index and existence flag; vectorised over `x` | O(m log n) |
| `binary_search_py(sorted_array, x)` | Pure-Python loop version (reference for benchmarks) | O(log n) |

#### `numcompute.rank`

| Function | Description | Time |
|---|---|---|
| `rank(data, method='average', axis=-1, nan_policy='keep')` | Ranks starting at 1. Ties: `'average'`, `'min'`, `'max'`, `'dense'`, `'ordinal'`. Returns float64 for `'average'`, otherwise int64 | O(n log n) |
| `percentile(data, q, interpolation='linear', axis=None, nan_policy='propagate')` | q in [0, 100] (scalar or array). Interpolation: `'linear'`, `'lower'`, `'higher'`, `'midpoint'`, `'nearest'` — matches `np.percentile` | O(n log n) |

Tie handling example for `[10, 20, 20, 30]`:

| method | ranks |
|---|---|
| average | `[1, 2.5, 2.5, 4]` |
| min | `[1, 2, 2, 4]` |
| max | `[1, 3, 3, 4]` |
| dense | `[1, 2, 2, 3]` |
| ordinal | `[1, 2, 3, 4]` |

#### `numcompute.stats`

| Function | Description | Time |
|---|---|---|
| `mean`, `median`, `minimum`, `maximum` `(data, axis=None, nan_policy='propagate')` | Basic descriptive statistics | O(n) |
| `var`, `std` `(data, axis=None, ddof=0, nan_policy='propagate')` | Variance / standard deviation; `ddof=1` for the sample version | O(n) |
| `quantile(data, q, axis=None, interpolation='linear', nan_policy='propagate')` | q in [0, 1]; same engine as `rank.percentile` | O(n log n) |
| `describe(data, axis=0, ddof=1, nan_policy='propagate')` | Dict with `count`, `mean`, `std`, `min`, `25%`, `50%`, `75%`, `max` | O(n log n) |
| `histogram(data, bins=10, range=None, density=False)` | Counts per bin and bin edges; bins are `[a, b)` except the last, which is closed (same as `np.histogram`). `bins` may be an int or explicit edges | O(n) |
| `RunningStats(ddof=0)` | Streaming mean / var / std / min / max over batches (`.update(batch)`), using Welford / Chan's merge for numerical stability | O(n) per batch |

```python
rs = stats.RunningStats()
for chunk in np.array_split(np.arange(10.0), 3):
    rs.update(chunk)
rs.mean, rs.count                                       # (4.5, 10)
```

### Design notes

- **Vectorised ranking.** One stable `argsort`; group starts are found with `s[1:] != s[:-1]`, dense ranks with `cumsum`, and group first/last positions with `maximum.accumulate` / reversed `minimum.accumulate`. No Python loops.
- **Top-k** uses `np.argpartition` (O(n)) and only sorts the k selected items.
- **Quickselect** uses a 3-way split (`<`, `==`, `>` pivot), so duplicates and all-equal arrays do not degrade performance.
- **Stable descending sort** sorts the reversed array and maps indices back, instead of negating values (which overflows for unsigned ints and fails for strings).
- **Percentiles** snap positions such as `7.000000000000001` back to `7` before choosing neighbours, avoiding floating-point off-by-one errors.
- **Streaming variance** never uses `E[x²] − E[x]²`, which suffers catastrophic cancellation for data with a large offset.

### Tests

`tests/test_sort_search.py`, `tests/test_rank.py` and `tests/test_stats.py` contain 55 test cases covering empty arrays, all-equal values, duplicates/ties, extreme `k` (0, 1, n), NaNs, non-contiguous strided views, invalid arguments, and agreement with NumPy reference functions (`np.sort`, `np.percentile`, `np.histogram`, ...).
