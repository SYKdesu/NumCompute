from io import StringIO
from pathlib import Path
from typing import Iterator, Union

import numpy as np


PathLike = Union[str, Path]


def _validate_path(filepath: PathLike) -> Path:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a file, got: {path}")
    return path


def _validate_options(delimiter: str, skip_header: int) -> None:
    if not isinstance(delimiter, str) or delimiter == "":
        raise ValueError("delimiter must be a non-empty string")
    if not isinstance(skip_header, int) or skip_header < 0:
        raise ValueError("skip_header must be a non-negative integer")


def load_csv(
    filepath: PathLike,
    delimiter: str = ",",
    dtype=float,
    missing_values=None,
    filling_values=np.nan,
    skip_header: int = 0,
) -> np.ndarray:

    """Load numeric or structured CSV data into a NumPy array.

    delimiter, dtype, missing_values, filling_values and skip_header are
    passed to numpy.genfromtxt. The returned shape follows that reader.
    Raises FileNotFoundError for a missing path and ValueError for invalid
    options or unparseable data. Time and storage are O(rows * columns)."""
    path = _validate_path(filepath)
    _validate_options(delimiter, skip_header)

    try:
        data = np.genfromtxt(
            path,
            delimiter=delimiter,
            dtype=dtype,
            missing_values=missing_values,
            filling_values=filling_values,
            skip_header=skip_header,
        )

    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Could not parse file '{path}': {exc}"
        ) from exc

    return np.asarray(data)


def iter_csv(
    filepath: PathLike,
    chunk_size: int,
    delimiter: str = ",",
    dtype=float,
    missing_values=None,
    filling_values=np.nan,
    skip_header: int = 0,
) -> Iterator[np.ndarray]:
    """Yield a CSV file in bounded-size NumPy chunks.

    Only ``chunk_size`` data rows are buffered at a time, so this is suitable
    for inputs that should not be loaded fully into memory.
    """

    path = _validate_path(filepath)
    _validate_options(delimiter, skip_header)
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    def parse(lines: list[str]) -> np.ndarray:
        try:
            return np.asarray(
                np.genfromtxt(
                    StringIO("".join(lines)),
                    delimiter=delimiter,
                    dtype=dtype,
                    missing_values=missing_values,
                    filling_values=filling_values,
                    ndmin=2,
                )
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Could not parse file '{path}': {exc}") from exc

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for _ in range(skip_header):
            next(handle, None)
        rows: list[str] = []
        for line in handle:
            rows.append(line)
            if len(rows) == chunk_size:
                yield parse(rows)
                rows = []
        if rows:
            yield parse(rows)


class CSVReader:
    """Reusable CSV reader offering full and chunked reads."""

    def __init__(self, filepath: PathLike, **options):
        self.filepath = Path(filepath)
        self.options = dict(options)

    def read(self) -> np.ndarray:
        """Read the complete file using the options supplied at construction."""
        return load_csv(self.filepath, **self.options)

    def iter_chunks(self, chunk_size: int) -> Iterator[np.ndarray]:
        """Yield arrays containing at most chunk_size data rows at a time."""
        return iter_csv(self.filepath, chunk_size, **self.options)


__all__ = ["CSVReader", "iter_csv", "load_csv"]
