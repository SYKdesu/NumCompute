"""Generate quickstart.ipynb from the numbered executable demo sections.

Uses the standard library only. This clears notebook outputs; run
``python demo/verify_demo.py --write`` afterwards to save fresh outputs.
"""

from pathlib import Path
import ast
import json
import re
import textwrap
import uuid

ROOT = Path(__file__).resolve().parents[1]

NOTES = {
    1: ("CSV I/O",
        "A six-student synthetic CSV contains a blank attendance field, tied scores, "
        "numeric columns and a categorical group. Explicit dtypes preserve each column. "
        "Only hours and attendance enter X; score and passed are kept separate."),
    2: ("Preprocessing and Pipeline",
        "Impute the missing attendance with the demonstration constant 75, then standardise "
        "each feature. Each column has mean approximately 0 and population standard deviation 1. "
        "Fit once on the example data; transform new rows using those same fitted statistics."),
    3: ("Sorting, searching and ranking",
        "Top-3 scores are [95, 92, 92]. Stable sorting preserves the order of the two 92s. "
        "Ranking is ascending: average ranks share positions 4 and 5 as 4.5; dense ranks "
        "assign both scores rank 4. Binary search for 80 returns insertion index 2 and False. "
        "Quickselect uses zero-based k, so k=2 returns 85."),
    4: ("Statistics and bounded CSV streaming",
        "axis=0 reduces rows, returning one statistic per feature. nan_policy='omit' "
        "ignores NaNs for raw feature means. Score mean is 83.667 and population variance "
        "is 143.556. Histogram counts are [1, 1, 1, 3]; quantiles use fractions in [0,1], "
        "whereas percentiles use [0,100]. Reading two CSV rows at a time and updating "
        "RunningStats agrees with the batch mean and variance."),
    5: ("Estimator Pipeline and metrics",
        "The teaching estimator thresholds the first feature at its fitted mean. "
        "Pipeline.fit and predict connect preprocessing to the estimator. "
        "Five of six predictions match: accuracy=5/6, precision=1, recall=3/4, F1=6/7. "
        "The confusion matrix has true labels on rows and predictions on columns: "
        "[[2,0],[1,3]]. These are illustrative in-sample metrics. Continuous hours scores "
        "give ROC AUC=1 because they order all positive labels above negative labels, "
        "even though the particular hard threshold misses one positive."),
    6: ("Finite differences and numerical stability",
        "For f(z)=sum(z**2), the analytic gradient is 2z; at [1,2], the numerical result "
        "is approximately [2,4]. For F(z)=[sum(z),sum(z**2)], the Jacobian is "
        "[[1,1],[2,4]], with one row per output. Central differences use f(x+h) and "
        "f(x-h). Softmax subtracts the maximum before exponentiation, keeping logits "
        "near 1000 finite; its probabilities sum to 1."),
    7: ("Reproducible performance comparison",
        "Fixed seed 42; 20,000 rows; four features for scaling and softmax. "
        "Check equivalent results before timing. Report median time per call over "
        "five repeats, three calls per repeat and one warmup. Preparation and fitting "
        "are excluded. Speed-up is Python baseline / NumCompute; timings vary. "
        "Top-k compares a full Python sort to partial selection, so the algorithm "
        "also contributes to the difference. The standalone benchmark writes an "
        "environment-stamped performance table."),
}


def cell(kind: str, source: str, name: str) -> dict:
    result = {
        "cell_type": kind, "id": uuid.uuid5(uuid.NAMESPACE_URL, name).hex[:12],
        "metadata": {}, "source": source.strip() + "\n",
    }
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def main() -> None:
    source = (ROOT / "demo" / "integrated_quickstart.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    estimator = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    entry = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    lines = source.splitlines()
    body = textwrap.dedent("\n".join(lines[entry.lineno:entry.end_lineno]))
    sections = re.split(r"^# %% (.+)\n", body, flags=re.MULTILINE)
    cells = [cell("markdown", """# NumCompute: integrated quickstart

Run from the project root or demo directory. Select **Python (NumCompute)**, then
**Restart Kernel and Run All Cells**. The examples use the included synthetic
student CSV and NumCompute's Python/NumPy implementations.

Flow: CSV with missing values -> preprocessing -> Pipeline -> sorting/ranks ->
statistics/streaming -> metrics -> gradients/Jacobian -> performance comparison.

The code is also available in `demo/integrated_quickstart.py`.
Assertions check the main numerical results.
""", "intro")]
    setup = '''from pathlib import Path
import sys
import numpy as np

# Supports launching Jupyter from the project root or the demo folder.
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "numcompute" / "__init__.py").is_file()
             and (p / "demo" / "data" / "students.csv").is_file()), Path.cwd())
if not (ROOT / "numcompute" / "__init__.py").is_file():
    raise RuntimeError("Open Jupyter from the NumCompute project root or demo folder.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numcompute as nc  # type: ignore
from benchmark.bench_integrated import format_table, run_benchmarks  # type: ignore
from IPython.display import Markdown, display

np.set_printoptions(precision=3, suppress=True)
print("Python:", sys.version.split()[0], "| NumPy:", np.__version__)
print("NumCompute:", nc.__version__)
'''
    cells.append(cell("code", setup, "setup"))
    for i in range(1, len(sections), 2):
        number = int(sections[i].split(".")[0])
        title, explanation = NOTES[number]
        code = sections[i + 1].strip()
        cells.append(cell("markdown", f"## {number}. {title}\n\n{explanation}", f"note-{number}"))
        if number == 2:
            core, additional = code.split('clean = preprocess["impute"].transform(X)', 1)
            cells.append(cell("code", core, "section-2-core"))
            cells.append(cell("markdown", "### Additional preprocessing examples\n\n"
                              "OneHotEncoder maps A/B to two indicator columns. MinMaxScaler "
                              "maps this example to [0,1]. FeatureUnion joins two transformed "
                              "branches; appending one-hot columns gives shape (6,6).",
                              "note-2-extra"))
            code = 'clean = preprocess["impute"].transform(X)' + additional
        if number == 5:
            code = ast.get_source_segment(source, estimator) + "\n\n" + code
        if number == 7:
            code = code.replace("print(format_table(run_benchmarks()))",
                                "display(Markdown(format_table(run_benchmarks())))")
        cells.append(cell("code", code, f"section-{number}"))
    cells.append(cell("markdown", "## Tests\n\n"
                      "Run `python -m pytest -q` for the unit and integration tests. "
                      "`python demo/verify_demo.py` executes this notebook with a "
                      "fresh kernel from the project root and demo directory.", "end"))
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python (NumCompute)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "file_extension": ".py"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    path = ROOT / "demo" / "quickstart.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Generated {path} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
