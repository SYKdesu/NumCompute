from pathlib import Path
from typing import Union

import numpy as np


PathLike = Union[str, Path]


def load_csv(
    filepath: PathLike,
    delimiter: str = ",",
    dtype=float,
    missing_values=None,
    filling_values=np.nan,
    skip_header: int = 0,
) -> np.ndarray:

    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file, got: {path}")

    if not isinstance(delimiter, str) or delimiter == "":
        raise ValueError("delimiter must be a non-empty string")

    if not isinstance(skip_header, int) or skip_header < 0:
        raise ValueError("skip_header must be a non-negative integer")

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