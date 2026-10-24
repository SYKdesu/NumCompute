# tests/test_io.py

import numpy as np

from numcompute.io import load_csv


def test_load_basic_csv(tmp_path):
    file = tmp_path / "data.csv"
    file.write_text(
        "1,2,3\n"
        "4,5,6\n"
    )

    result = load_csv(file)

    expected = np.array([
        [1, 2, 3],
        [4, 5, 6],
    ])

    assert np.array_equal(result, expected)


def test_custom_delimiter(tmp_path):
    file = tmp_path / "data.tsv"
    file.write_text(
        "1\t2\n"
        "3\t4\n"
    )

    result = load_csv(file, delimiter="\t")

    expected = np.array([
        [1, 2],
        [3, 4],
    ])

    assert np.array_equal(result, expected)


def test_missing_values(tmp_path):
    file = tmp_path / "data.csv"
    file.write_text(
        "1,2\n"
        "3,\n"
    )

    result = load_csv(file)

    assert np.isnan(result[1, 1])


def test_skip_header(tmp_path):
    file = tmp_path / "data.csv"
    file.write_text(
        "x,y\n"
        "1,2\n"
        "3,4\n"
    )

    result = load_csv(
        file,
        skip_header=1
    )

    expected = np.array([
        [1, 2],
        [3, 4],
    ])

    assert np.array_equal(result, expected)


def test_missing_file():
    try:
        load_csv("does_not_exist.csv")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError(
            "Expected FileNotFoundError"
        )