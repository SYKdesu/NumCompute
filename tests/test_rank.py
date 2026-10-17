"""Unit tests for numcompute.rank."""

import numpy as np
import pytest

from numcompute.rank import percentile, rank


@pytest.mark.parametrize(
    "method, expected",
    [
        ("average", [1, 2.5, 2.5, 4]),
        ("min", [1, 2, 2, 4]),
        ("max", [1, 3, 3, 4]),
        ("dense", [1, 2, 2, 3]),
        ("ordinal", [1, 2, 3, 4]),
    ],
)
def test_rank_methods_with_ties(method, expected):
    np.testing.assert_array_equal(rank([10, 20, 20, 30], method=method), expected)


def test_rank_unsorted_input_and_dtype():
    r = rank([30, 10, 20, 10], method="average")
    np.testing.assert_array_equal(r, [4, 1.5, 3, 1.5])
    assert r.dtype == np.float64
    assert rank([3, 1, 2], method="dense").dtype == np.int64


def test_rank_all_equal():
    np.testing.assert_array_equal(rank([7, 7, 7], "average"), [2, 2, 2])
    np.testing.assert_array_equal(rank([7, 7, 7], "dense"), [1, 1, 1])
    np.testing.assert_array_equal(rank([7, 7, 7], "ordinal"), [1, 2, 3])


def test_rank_nan_keep_and_raise():
    r = rank([3.0, np.nan, 1.0, 3.0], "average")
    np.testing.assert_array_equal(r, [2.5, np.nan, 1.0, 2.5])
    with pytest.raises(ValueError):
        rank([1.0, np.nan], nan_policy="raise")


def test_rank_2d_axis():
    X = np.array([[3, 1, 2], [1, 1, 5]])
    np.testing.assert_array_equal(rank(X, "min", axis=1), [[3, 1, 2], [1, 1, 3]])
    np.testing.assert_array_equal(rank(X, "min", axis=0), [[2, 1, 1], [1, 1, 2]])
    assert rank(X, axis=None).shape == (6,)


def test_rank_empty_and_bad_method():
    assert rank(np.array([])).shape == (0,)
    with pytest.raises(ValueError):
        rank([1, 2], method="best")
    with pytest.raises(ValueError):
        rank(5)


@pytest.mark.parametrize("method", ["linear", "lower", "higher", "midpoint", "nearest"])
def test_percentile_matches_numpy(method):
    rng = np.random.default_rng(3)
    a = rng.normal(size=(6, 11))
    q = [0, 10, 25, 30, 50, 90, 100]
    for axis in (None, 0, 1):
        np.testing.assert_allclose(
            percentile(a, q, interpolation=method, axis=axis),
            np.percentile(a, q, method=method, axis=axis),
        )


def test_percentile_scalar_and_single_element():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([42], 73) == 42.0


def test_percentile_nan_policies():
    a = [1.0, np.nan, 3.0, 2.0]
    assert np.isnan(percentile(a, 50))
    assert percentile(a, 50, nan_policy="omit") == 2.0
    with pytest.raises(ValueError):
        percentile(a, 50, nan_policy="raise")
    X = np.array([[1.0, np.nan, 3.0], [np.nan, np.nan, np.nan]])
    np.testing.assert_allclose(
        percentile(X, 50, axis=1, nan_policy="omit"), [2.0, np.nan]
    )


def test_percentile_invalid_q_and_empty():
    with pytest.raises(ValueError):
        percentile([1, 2], 101)
    with pytest.raises(ValueError):
        percentile([1, 2], -1)
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1, 2], 50, interpolation="cubic")
