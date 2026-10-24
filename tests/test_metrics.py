import numpy as np
import pytest

from numcompute.metrics import (
    accuracy,
    confusion_matrix,
    f1,
    mean_squared_error,
    precision,
    recall,
)


def test_accuracy_supports_numeric_and_string_labels():
    assert accuracy([1, 0, 1, 1], [1, 1, 1, 0]) == 0.5
    assert accuracy(["cat", "dog"], ["cat", "cat"]) == 0.5


def test_confusion_matrix_uses_true_rows_and_predicted_columns():
    result = confusion_matrix([0, 0, 1, 1], [0, 1, 0, 1])
    np.testing.assert_array_equal(result, [[1, 1], [1, 1]])


def test_confusion_matrix_honors_explicit_label_order():
    result = confusion_matrix(["a", "b", "a"], ["a", "a", "b"], labels=["b", "a"])
    np.testing.assert_array_equal(result, [[0, 1], [1, 1]])


def test_confusion_matrix_normalizes_rows_without_nan():
    result = confusion_matrix([0, 0, 1], [0, 1, 1], labels=[0, 1, 2], normalize="true")
    np.testing.assert_allclose(result, [[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])


def test_binary_precision_recall_and_f1():
    y_true = [1, 1, 1, 0]
    y_pred = [1, 1, 0, 1]
    assert precision(y_true, y_pred) == pytest.approx(2 / 3)
    assert recall(y_true, y_pred) == pytest.approx(2 / 3)
    assert f1(y_true, y_pred) == pytest.approx(2 / 3)


def test_multiclass_macro_and_per_class_values():
    y_true = [0, 1, 2, 2]
    y_pred = [0, 2, 2, 1]
    per_class = precision(y_true, y_pred, average=None)
    np.testing.assert_allclose(per_class, [1.0, 0.0, 0.5])
    assert precision(y_true, y_pred, average="macro") == pytest.approx(0.5)


def test_zero_division_is_configurable():
    assert precision([0, 0], [0, 0], pos_label=1, zero_division=1) == 1.0


def test_mean_squared_error_supports_multidimensional_arrays():
    assert mean_squared_error([[1, 2], [3, 4]], [[1, 1], [5, 4]]) == pytest.approx(1.25)


@pytest.mark.parametrize(
    "function",
    [accuracy, precision, recall, f1],
)
def test_classification_metrics_reject_mismatched_lengths(function):
    with pytest.raises(ValueError, match="same shape"):
        function([0, 1], [0])


def test_mse_rejects_non_finite_inputs():
    with pytest.raises(ValueError, match="finite"):
        mean_squared_error([1.0, np.nan], [1.0, 2.0])
