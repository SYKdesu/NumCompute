"""NumCompute public API for the Member C workstream."""

from .benchmarking import BenchmarkResult, benchmark, benchmark_many, format_results
from .metrics import (
    accuracy,
    accuracy_score,
    confusion_matrix,
    f1,
    f1_score,
    mean_squared_error,
    mse,
    precision,
    precision_score,
    recall,
    recall_score,
)
from .optim import gradient, jacobian, numerical_gradient, numerical_jacobian
from .pipeline import Pipeline, make_pipeline

__version__ = "0.1.0"

__all__ = [
    "BenchmarkResult",
    "Pipeline",
    "accuracy",
    "accuracy_score",
    "benchmark",
    "benchmark_many",
    "confusion_matrix",
    "f1",
    "f1_score",
    "format_results",
    "gradient",
    "jacobian",
    "make_pipeline",
    "mean_squared_error",
    "mse",
    "numerical_gradient",
    "numerical_jacobian",
    "precision",
    "precision_score",
    "recall",
    "recall_score",
]
