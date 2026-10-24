"""Loop versus NumPy benchmarks for the three team members' modules.

Run ``python benchmark/bench_integrated.py`` from the project root. Timings
measure computation on already prepared inputs; CSV I/O and list conversion
are excluded. Each baseline and NumCompute result is checked before timing.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numcompute as nc


def run_benchmarks(size: int = 20_000, repeat: int = 5, number: int = 3) -> list[dict]:
    """Check four tasks and return median per-call timings and speed ratios.

    ``size`` is the number of rows (four features for scaling/softmax).
    A fresh fixed-seed generator makes input data reproducible. No speed
    threshold is asserted because hardware and background load affect timing.
    """
    for name, value in (("size", size), ("repeat", repeat), ("number", number)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    rng = np.random.default_rng(42)
    X = rng.normal(size=(size, 4))
    rows = X.tolist()
    scaler = nc.StandardScaler().fit(X)
    means, scales = scaler.mean_.tolist(), scaler.scale_.tolist()
    values = X[:, 0].copy()
    value_list = values.tolist()
    k = min(10, size)
    y_true = rng.integers(0, 2, size=size)
    y_pred = rng.integers(0, 2, size=size)
    true_list, pred_list = y_true.tolist(), y_pred.tolist()

    def loop_scale():
        return [[(row[j] - means[j]) / scales[j] for j in range(4)] for row in rows]

    def loop_softmax():
        output = []
        for row in rows:
            maximum = max(row)
            exponentials = [math.exp(value - maximum) for value in row]
            total = sum(exponentials)
            output.append([value / total for value in exponentials])
        return output

    def loop_accuracy():
        correct = 0
        for actual, predicted in zip(true_list, pred_list):
            if actual == predicted:
                correct += 1
        return correct / size

    cases = [
        ("StandardScaler.transform", f"{size:,} x 4", loop_scale,
         lambda: scaler.transform(X)),
        ("stable softmax", f"{size:,} x 4", loop_softmax,
         lambda: nc.softmax(X, axis=1)),
        ("top-k", f"n={size:,}, k={k}",
         lambda: sorted(value_list, reverse=True)[:k],
         lambda: nc.topk(values, k, return_indices=False)),
        ("accuracy", f"n={size:,}", loop_accuracy,
         lambda: nc.accuracy(y_true, y_pred)),
    ]
    results = []
    for task, shape, baseline, vectorised in cases:
        np.testing.assert_allclose(baseline(), vectorised(), rtol=1e-12, atol=1e-12)
        loop_time = nc.benchmark(baseline, repeat=repeat, number=number).median
        numpy_time = nc.benchmark(vectorised, repeat=repeat, number=number).median
        results.append({
            "task": task, "size": shape, "loop_ms": loop_time * 1000,
            "numpy_ms": numpy_time * 1000, "speedup": loop_time / numpy_time,
            "same_result": True,
        })
    return results


def format_table(results: list[dict]) -> str:
    """Return a Markdown table suitable for the notebook and README."""
    lines = [
        "| Task | Data size | Python baseline (ms) | NumCompute (ms) | Speed-up | Same result |",
        "|---|---|---:|---:|---:|:---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['task']} | {row['size']} | {row['loop_ms']:.4f} | "
            f"{row['numpy_ms']:.4f} | {row['speedup']:.2f}x | "
            f"{'yes' if row['same_result'] else 'no'} |"
        )
    return "\n".join(lines)


def format_report(results: list[dict], repeat: int = 5, number: int = 3) -> str:
    return (
        "# Integrated benchmark results\n\n" + format_table(results) + "\n\n"
        "## Environment and method\n\n"
        f"- Measured at: {datetime.now(timezone.utc).isoformat()}\n"
        f"- Python: {platform.python_version()} ({platform.python_implementation()})\n"
        f"- NumPy: {np.__version__}\n"
        f"- OS: {platform.system()} {platform.release()} ({platform.machine()})\n"
        f"- CPU: {platform.processor() or 'unknown'}; logical cores: {os.cpu_count()}\n"
        f"- Random seed: 42; {repeat} repeats, {number} calls per repeat, one warmup call.\n"
        "- Reported time: median seconds per call, converted to milliseconds.\n"
        "- Data/list preparation, scaler fitting, CSV I/O and correctness checks are outside timing.\n"
        "- Scaling uses identical fitted means/scales. Both softmax versions subtract each row's maximum.\n"
        "- Top-k baseline uses Python's full sorted(); NumCompute uses partial selection.\n"
        "  This comparison includes an algorithm difference, not only loop/vectorisation overhead.\n"
        "- Equality is checked with rtol=atol=1e-12 before timing.\n"
        "- Speed-up = Python baseline / NumCompute. Results vary by machine and input size;\n"
        "  values below 1 mean the baseline was faster. No minimum speed-up is required.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=20_000)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--number", type=int, default=3)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "benchmark" / "results_integrated.md")
    args = parser.parse_args()
    if min(args.size, args.repeat, args.number) < 1:
        parser.error("size, repeat and number must be positive")
    results = run_benchmarks(args.size, args.repeat, args.number)
    report = format_report(results, args.repeat, args.number)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
