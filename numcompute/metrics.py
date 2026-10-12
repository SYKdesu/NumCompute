"""Evaluation metrics implemented with NumPy only.

The public API intentionally exposes both concise names (``accuracy``) and
scikit-learn-style names (``accuracy_score``) so that the module is pleasant
to use in teaching examples while remaining familiar to Python users.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, Optional

import numpy as np

Average = Optional[Literal["binary", "micro", "macro", "weighted"]]


def _as_1d(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _validate_classification_inputs(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    true = _as_1d(y_true, "y_true")
    pred = _as_1d(y_pred, "y_pred")
    if true.shape != pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape; "
            f"got {true.shape} and {pred.shape}"
        )
    return true, pred


def _unique_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    combined = np.concatenate((y_true, y_pred))
    try:
        return np.unique(combined)
    except TypeError:
        # Mixed, non-orderable object labels still receive a deterministic
        # first-seen order.
        unique: list[Any] = []
        for value in combined.tolist():
            if not any(value == existing for existing in unique):
                unique.append(value)
        return np.asarray(unique, dtype=object)


def _validate_labels(
    labels: Iterable[Any] | None, y_true: np.ndarray, y_pred: np.ndarray
) -> np.ndarray:
    if labels is None:
        return _unique_labels(y_true, y_pred)

    result = _as_1d(list(labels), "labels")
    if len(_unique_labels(result, result)) != len(result):
        raise ValueError("labels must not contain duplicates")

    observed = _unique_labels(y_true, y_pred)
    for value in observed:
        if not np.any(result == value):
            raise ValueError(f"observed label {value!r} is missing from labels")
    return result


def accuracy_score(y_true: Any, y_pred: Any) -> float:
    """Return the fraction of exactly matching predictions."""

    true, pred = _validate_classification_inputs(y_true, y_pred)
    return float(np.mean(true == pred))


def confusion_matrix(
    y_true: Any,
    y_pred: Any,
    *,
    labels: Iterable[Any] | None = None,
    normalize: Literal["true", "pred", "all"] | None = None,
) -> np.ndarray:
    """Build a confusion matrix with true labels on rows and predictions on columns.

    Parameters
    ----------
    normalize:
        ``"true"`` normalizes rows, ``"pred"`` normalizes columns and
        ``"all"`` normalizes the complete matrix. Empty rows/columns remain 0.
    """

    true, pred = _validate_classification_inputs(y_true, y_pred)
    class_labels = _validate_labels(labels, true, pred)
    matrix = np.zeros((len(class_labels), len(class_labels)), dtype=np.int64)

    for row, label_true in enumerate(class_labels):
        true_mask = true == label_true
        for column, label_pred in enumerate(class_labels):
            matrix[row, column] = np.count_nonzero(true_mask & (pred == label_pred))

    if normalize is None:
        return matrix
    if normalize not in {"true", "pred", "all"}:
        raise ValueError("normalize must be one of None, 'true', 'pred', or 'all'")

    result = matrix.astype(float)
    if normalize == "true":
        denominator = result.sum(axis=1, keepdims=True)
    elif normalize == "pred":
        denominator = result.sum(axis=0, keepdims=True)
    else:
        denominator = np.asarray(result.sum())
    return np.divide(result, denominator, out=np.zeros_like(result), where=denominator != 0)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray, zero_division: int) -> np.ndarray:
    if zero_division not in (0, 1):
        raise ValueError("zero_division must be 0 or 1")
    default = np.full_like(numerator, float(zero_division), dtype=float)
    return np.divide(numerator, denominator, out=default, where=denominator != 0)


def _classification_statistic(
    y_true: Any,
    y_pred: Any,
    *,
    metric: Literal["precision", "recall", "f1"],
    average: Average,
    labels: Iterable[Any] | None,
    pos_label: Any,
    zero_division: int,
) -> float | np.ndarray:
    true, pred = _validate_classification_inputs(y_true, y_pred)
    class_labels = _validate_labels(labels, true, pred)
    if average == "binary" and not np.any(class_labels == pos_label):
        if labels is not None:
            raise ValueError(f"pos_label={pos_label!r} is not present in labels")
        if len(class_labels) > 1:
            raise ValueError(f"pos_label={pos_label!r} is not a valid label")
        class_labels = np.append(class_labels, pos_label)
    matrix = confusion_matrix(true, pred, labels=class_labels)

    tp = np.diag(matrix).astype(float)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    support = matrix.sum(axis=1).astype(float)

    precision_values = _safe_divide(tp, tp + fp, zero_division)
    recall_values = _safe_divide(tp, tp + fn, zero_division)
    if metric == "precision":
        values = precision_values
    elif metric == "recall":
        values = recall_values
    else:
        values = _safe_divide(
            2.0 * precision_values * recall_values,
            precision_values + recall_values,
            zero_division,
        )

    if average is None:
        return values
    if average == "binary":
        if len(class_labels) > 2:
            raise ValueError("average='binary' is only valid for binary targets")
        positions = np.flatnonzero(class_labels == pos_label)
        if positions.size == 0:
            raise ValueError(f"pos_label={pos_label!r} is not a valid label")
        return float(values[int(positions[0])])
    if average == "macro":
        return float(np.mean(values))
    if average == "weighted":
        total = support.sum()
        if total == 0:
            return float(zero_division)
        return float(np.average(values, weights=support))
    if average == "micro":
        total_tp = np.asarray([tp.sum()])
        if metric == "precision":
            return float(_safe_divide(total_tp, np.asarray([tp.sum() + fp.sum()]), zero_division)[0])
        if metric == "recall":
            return float(_safe_divide(total_tp, np.asarray([tp.sum() + fn.sum()]), zero_division)[0])
        micro_precision = _safe_divide(
            total_tp, np.asarray([tp.sum() + fp.sum()]), zero_division
        )
        micro_recall = _safe_divide(
            total_tp, np.asarray([tp.sum() + fn.sum()]), zero_division
        )
        return float(
            _safe_divide(
                2.0 * micro_precision * micro_recall,
                micro_precision + micro_recall,
                zero_division,
            )[0]
        )
    raise ValueError("average must be one of 'binary', 'micro', 'macro', 'weighted', or None")


def precision_score(
    y_true: Any,
    y_pred: Any,
    *,
    average: Average = "binary",
    labels: Iterable[Any] | None = None,
    pos_label: Any = 1,
    zero_division: int = 0,
) -> float | np.ndarray:
    """Compute binary or multiclass precision."""

    return _classification_statistic(
        y_true,
        y_pred,
        metric="precision",
        average=average,
        labels=labels,
        pos_label=pos_label,
        zero_division=zero_division,
    )


def recall_score(
    y_true: Any,
    y_pred: Any,
    *,
    average: Average = "binary",
    labels: Iterable[Any] | None = None,
    pos_label: Any = 1,
    zero_division: int = 0,
) -> float | np.ndarray:
    """Compute binary or multiclass recall."""

    return _classification_statistic(
        y_true,
        y_pred,
        metric="recall",
        average=average,
        labels=labels,
        pos_label=pos_label,
        zero_division=zero_division,
    )


def f1_score(
    y_true: Any,
    y_pred: Any,
    *,
    average: Average = "binary",
    labels: Iterable[Any] | None = None,
    pos_label: Any = 1,
    zero_division: int = 0,
) -> float | np.ndarray:
    """Compute the harmonic mean of precision and recall."""

    return _classification_statistic(
        y_true,
        y_pred,
        metric="f1",
        average=average,
        labels=labels,
        pos_label=pos_label,
        zero_division=zero_division,
    )


def mean_squared_error(y_true: Any, y_pred: Any) -> float:
    """Return the mean squared difference between numeric arrays."""

    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape; "
            f"got {true.shape} and {pred.shape}"
        )
    if true.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    if not np.all(np.isfinite(true)) or not np.all(np.isfinite(pred)):
        raise ValueError("y_true and y_pred must contain only finite values")
    return float(np.mean(np.square(true - pred)))


# Concise aliases required by the assignment plan.
accuracy = accuracy_score
precision = precision_score
recall = recall_score
f1 = f1_score
mse = mean_squared_error


__all__ = [
    "accuracy",
    "accuracy_score",
    "confusion_matrix",
    "f1",
    "f1_score",
    "mean_squared_error",
    "mse",
    "precision",
    "precision_score",
    "recall",
    "recall_score",
]
