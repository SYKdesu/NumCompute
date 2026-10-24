"""Execute quickstart with fresh Jupyter kernels from root and demo folders.

Use --write to persist the final execution's outputs to quickstart.ipynb.
Nothing is saved when execution fails. Requires the optional demo dependencies.
"""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / "demo" / "quickstart.ipynb"
    for cwd in (root, root / "demo"):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        client = NotebookClient(notebook, timeout=120, kernel_name="python3",
                                resources={"metadata": {"path": str(cwd)}})
        executed = client.execute()
        count = sum(cell.cell_type == "code" for cell in executed.cells)
        print(f"PASS: {count} code cells, fresh kernel, working directory {cwd}")
    if args.write:
        nbformat.write(executed, path)
        print(f"Saved executed outputs to {path}")


if __name__ == "__main__":
    main()
