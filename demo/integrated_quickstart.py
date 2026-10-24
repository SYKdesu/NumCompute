"""Integrated team demo. Run from any directory; data paths use this file.

The numbered sections are also exported to quickstart.ipynb by
``python demo/build_notebook.py``. Edit this script to update both versions.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import numcompute as nc
from benchmark.bench_integrated import format_table, run_benchmarks


class MeanThresholdClassifier:
    """Teaching estimator: learn the first feature's mean, then threshold it.

    Demonstrates Pipeline.fit/predict. The metrics use this same six-row
    example, not a held-out model evaluation.
    """

    def fit(self, X, y=None):
        self.threshold_ = float(np.mean(np.asarray(X)[:, 0]))
        return self

    def predict(self, X):
        return (np.asarray(X)[:, 0] >= self.threshold_).astype(int)


def main() -> None:
    np.set_printoptions(precision=3, suppress=True)

    # %% 1. CSV I/O
    print("\n1. CSV I/O - six students, one missing attendance")
    csv_path = ROOT / "demo" / "data" / "students.csv"
    schema = [("hours", "f8"), ("attendance", "f8"), ("score", "f8"),
              ("group", "U1"), ("passed", "i4")]
    data = nc.load_csv(csv_path, dtype=schema, skip_header=1)
    X = np.column_stack([data["hours"], data["attendance"]])
    scores = data["score"]
    y_true = data["passed"]
    categories = data["group"].reshape(-1, 1)
    print("Features [hours, attendance]:\n", X)
    print("Scores:", scores)
    print("Missing values:", int(np.isnan(X).sum()))
    assert data.shape == (6,) and np.isnan(X).sum() == 1

    # %% 2. Preprocessing and Pipeline
    print("\n2. Preprocessing and Pipeline")
    # 75 is a deliberate constant for this synthetic attendance example.
    # Targets/scores stay out of X. Fit once, reuse transform on new data.
    preprocess = nc.Pipeline([
        ("impute", nc.Imputer(fill_value=75.0)),
        ("scale", nc.StandardScaler()),
    ])
    prepared = preprocess.fit_transform(X)
    print("Standardised features:\n", prepared)
    print("Column means:", nc.mean(prepared, axis=0))
    print("Column standard deviations:", nc.std(prepared, axis=0))
    np.testing.assert_allclose(nc.mean(prepared, axis=0), [0, 0], atol=1e-12)
    np.testing.assert_allclose(nc.std(prepared, axis=0), [1, 1], atol=1e-12)
    learned_mean = preprocess["scale"].mean_.copy()
    new_data = np.array([[2.5, 70.0], [6.5, np.nan]])
    print("New rows, using the fitted pipeline:\n", preprocess.transform(new_data))
    np.testing.assert_array_equal(preprocess["scale"].mean_, learned_mean)
    clean = preprocess["impute"].transform(X)
    encoded = nc.OneHotEncoder().fit_transform(categories)
    minmax = nc.MinMaxScaler().fit_transform(clean)
    print("Groups A/B, one-hot:\n", encoded)
    print("Min-max feature range:", minmax.min(axis=0), minmax.max(axis=0))
    union = nc.make_union(nc.StandardScaler(), nc.MinMaxScaler())
    combined = np.column_stack([union.fit_transform(clean), encoded])
    print("FeatureUnion (two scalings) + one-hot shape:", combined.shape)
    assert combined.shape == (6, 6) and np.isfinite(combined).all()

    # %% 3. Sorting, searching and ranking
    print("\n3. Sorting, searching and ranking")
    sorted_scores, order = nc.stable_sort(scores, descending=True, return_indices=True)
    print("Stable descending scores / indices:", sorted_scores, order)
    print("Top-3 scores:", nc.topk(scores, 3, return_indices=False))
    print("Third smallest (zero-based k=2):", nc.quickselect(scores, 2, seed=42))
    print("Search 80 (insertion index, exists):", nc.binary_search(np.sort(scores), 80))
    average_ranks = nc.rank_values(scores, method="average")
    dense_ranks = nc.rank_values(scores, method="dense")
    print("Average ranks (smallest is 1):", average_ranks)
    print("Dense ranks:", dense_ranks)
    print("Score quartiles:", nc.percentile(scores, [25, 50, 75]))
    np.testing.assert_array_equal(average_ranks, [1, 2, 3, 4.5, 4.5, 6])
    np.testing.assert_array_equal(dense_ranks, [1, 2, 3, 4, 4, 5])
    assert list(order[1:3]) == [3, 4]  # equal scores retain original order

    # %% 4. Descriptive and streaming statistics
    print("\n4. Statistics and bounded CSV streaming")
    print("Score mean / population variance:", nc.mean(scores), nc.var(scores))
    print("Score summary:", nc.describe(scores))
    counts, edges = nc.histogram(scores, bins=[60, 70, 80, 90, 100])
    print("Histogram counts / edges:", counts, edges)
    print("Quantiles 0.25 / 0.5 / 0.75:", nc.quantile(scores, [0.25, 0.5, 0.75]))
    print("Raw feature means, omit NaNs:", nc.mean(X, axis=0, nan_policy="omit"))
    running = nc.RunningStats(ddof=0)
    reader = nc.CSVReader(csv_path, dtype=schema, skip_header=1)
    chunk_shapes = []
    for chunk in reader.iter_chunks(2):
        chunk_shapes.append(chunk.shape)
        running.update(chunk["score"].reshape(-1))
    print("Chunk shapes:", chunk_shapes)
    print("Streaming count / mean / variance:", running.count, running.mean, running.var)
    np.testing.assert_allclose([running.mean, running.var], [nc.mean(scores), nc.var(scores)])
    np.testing.assert_array_equal(counts, [1, 1, 1, 3])

    # %% 5. Estimator Pipeline and metrics
    print("\n5. Pipeline.fit / predict and metrics (illustrative, same data)")
    model = nc.Pipeline([
        ("impute", nc.Imputer(fill_value=75.0)),
        ("scale", nc.StandardScaler()),
        ("classifier", MeanThresholdClassifier()),
    ])
    model.fit(X, y_true)
    y_pred = model.predict(X)
    print("Actual / predicted labels:", y_true, y_pred)
    print("Accuracy:", nc.accuracy(y_true, y_pred))
    print("Precision / recall / F1:", nc.precision(y_true, y_pred),
          nc.recall(y_true, y_pred), nc.f1(y_true, y_pred))
    print("Confusion matrix (rows=true, columns=predicted):\n",
          nc.confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print("MSE example:", nc.mse([1, 2, 3], [1, 2, 4]))
    print("ROC AUC (continuous hours score):", nc.roc_auc_score(y_true, prepared[:, 0]))
    np.testing.assert_array_equal(y_pred, [0, 0, 0, 1, 1, 1])
    np.testing.assert_allclose(nc.accuracy(y_true, y_pred), 5 / 6)
    np.testing.assert_array_equal(nc.confusion_matrix(y_true, y_pred), [[2, 0], [1, 3]])

    # %% 6. Finite differences and numerical stability
    print("\n6. Gradients, Jacobian and stable utilities")
    point = np.array([1.0, 2.0])
    grad = nc.gradient(lambda z: np.sum(z ** 2), point, method="central")
    jac = nc.jacobian(lambda z: np.array([z.sum(), np.sum(z ** 2)]), point)
    print("Gradient of sum(z**2) at [1,2]:", grad)
    print("Jacobian of [sum(z), sum(z**2)]:\n", jac)
    logits = np.array([[1000.0, 1001.0, 1002.0]])
    probabilities = nc.softmax(logits, axis=1)
    print("Stable softmax at large logits:", probabilities)
    print("Stable logsumexp:", nc.logsumexp(logits, axis=1))
    print("Euclidean distance [0,0] to [3,4]:", nc.euclidean_distance([0, 0], [3, 4]))
    print("Batch shapes:", [batch.shape for batch in nc.batch_iterator(prepared, 4)])
    np.testing.assert_allclose(grad, [2, 4], atol=1e-7)
    np.testing.assert_allclose(jac, [[1, 1], [2, 4]], atol=1e-7)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), [1])

    # %% 7. Reproducible performance comparison
    print("\n7. Loop versus NumPy; correctness checked before timing")
    print(format_table(run_benchmarks()))
    print("Median per call; seed=42, 5 repeats, 3 calls, one warmup.")
    print("Data preparation and fitting excluded; top-k baseline uses a full Python sort.")
    print("\nDemo completed: all output checks passed.")


if __name__ == "__main__":
    main()
