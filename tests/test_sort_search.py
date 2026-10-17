"""Unit tests for numcompute.sort_search."""

import numpy as np
import pytest

from numcompute.sort_search import (
    binary_search,
    binary_search_py,
    multikey_sort,
    quickselect,
    stable_sort,
    topk,
)


# ---------------------------------------------------------------- stable_sort
def test_stable_sort_matches_numpy_2d():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 5, size=(4, 7))
    for axis in (0, 1, None):
        np.testing.assert_array_equal(stable_sort(a, axis=axis), np.sort(a, axis=axis))


def test_stable_sort_descending_keeps_tie_order():
    a = np.array([1, 3, 1, 3, 2])
    out, idx = stable_sort(a, descending=True, return_indices=True)
    np.testing.assert_array_equal(out, [3, 3, 2, 1, 1])
    np.testing.assert_array_equal(idx, [1, 3, 4, 0, 2])  # ties in original order


def test_stable_sort_non_contiguous_view():
    base = np.arange(20)[::-1]
    view = base[::3]  # non-contiguous strided view
    assert not view.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(stable_sort(view), np.sort(view))


def test_stable_sort_empty():
    assert stable_sort(np.array([])).size == 0


# -------------------------------------------------------------- multikey_sort
def test_multikey_sort_primary_then_secondary():
    X = np.array([[2, 9], [1, 5], [2, 1], [1, 7]])
    out, order = multikey_sort(X, keys=[0, 1], return_indices=True)
    np.testing.assert_array_equal(out, [[1, 5], [1, 7], [2, 1], [2, 9]])
    np.testing.assert_array_equal(order, [1, 3, 2, 0])


def test_multikey_sort_mixed_directions():
    X = np.array([[2, 9], [1, 5], [2, 1], [1, 7]])
    out = multikey_sort(X, keys=[0, 1], descending=[False, True])
    np.testing.assert_array_equal(out, [[1, 7], [1, 5], [2, 9], [2, 1]])


def test_multikey_sort_errors():
    with pytest.raises(ValueError):
        multikey_sort(np.arange(5), keys=[0])
    with pytest.raises(IndexError):
        multikey_sort(np.ones((3, 2)), keys=[5])
    with pytest.raises(ValueError):
        multikey_sort(np.ones((3, 2)), keys=[0, 1], descending=[True])


# ----------------------------------------------------------------------- topk
def test_topk_largest_and_smallest():
    a = np.array([5, 1, 9, 3, 7])
    v, i = topk(a, 3)
    np.testing.assert_array_equal(v, [9, 7, 5])
    np.testing.assert_array_equal(a[i], v)
    v, _ = topk(a, 2, largest=False)
    np.testing.assert_array_equal(v, [1, 3])


@pytest.mark.parametrize("k", [0, 1, 6])
def test_topk_extreme_k(k):
    a = np.array([4, 2, 8, 6, 1, 3])
    v = topk(a, k, return_indices=False)
    np.testing.assert_array_equal(v, np.sort(a)[::-1][:k])


def test_topk_invalid_k():
    with pytest.raises(ValueError):
        topk([1, 2, 3], 4)
    with pytest.raises(ValueError):
        topk([1, 2, 3], -1)
    with pytest.raises(TypeError):
        topk([1, 2, 3], 1.5)


def test_topk_ties_and_all_equal():
    v, i = topk(np.array([2, 2, 2, 2]), 2)
    np.testing.assert_array_equal(v, [2, 2])
    assert len(set(i.tolist())) == 2  # two distinct positions


def test_topk_nan_is_never_preferred():
    a = np.array([1.0, np.nan, 3.0, 2.0])
    np.testing.assert_array_equal(topk(a, 2, return_indices=False), [3.0, 2.0])
    np.testing.assert_array_equal(
        topk(a, 2, largest=False, return_indices=False), [1.0, 2.0]
    )


def test_topk_along_axis0():
    a = np.array([[1, 9], [5, 2], [3, 7]])
    v = topk(a, 2, axis=0, return_indices=False)
    np.testing.assert_array_equal(v, [[5, 9], [3, 7]])


# ---------------------------------------------------------------- quickselect
def test_quickselect_matches_sort_random():
    rng = np.random.default_rng(1)
    a = rng.normal(size=101)
    s = np.sort(a)
    for k in (0, 50, 100):
        assert quickselect(a, k, seed=0) == s[k]


def test_quickselect_duplicates_and_all_equal():
    a = np.array([3, 1, 3, 3, 2, 1])
    assert [quickselect(a, k, seed=0) for k in range(6)] == sorted(a.tolist())
    assert quickselect(np.full(50, 7), 25) == 7


def test_quickselect_does_not_modify_input():
    a = np.array([5, 4, 3, 2, 1])
    before = a.copy()
    quickselect(a, 2)
    np.testing.assert_array_equal(a, before)


def test_quickselect_errors():
    with pytest.raises(ValueError):
        quickselect([], 0)
    with pytest.raises(ValueError):
        quickselect([1, 2, 3], 3)
    with pytest.raises(ValueError):
        quickselect([1.0, np.nan], 0)
    with pytest.raises(ValueError):
        quickselect(np.ones((2, 2)), 0)


# -------------------------------------------------------------- binary_search
def test_binary_search_found_and_missing():
    a = np.array([1, 3, 3, 5, 9])
    assert binary_search(a, 3) == (1, True)
    assert binary_search(a, 3, side="right") == (3, True)
    assert binary_search(a, 4) == (3, False)
    assert binary_search(a, 0) == (0, False)
    assert binary_search(a, 10) == (5, False)


def test_binary_search_vectorised_matches_loop_version():
    rng = np.random.default_rng(2)
    a = np.sort(rng.integers(0, 50, 40))
    queries = np.arange(-2, 53)
    idx, found = binary_search(a, queries)
    for q, i, f in zip(queries, idx, found):
        assert binary_search_py(a, q) == (i, f)


def test_binary_search_empty_and_unsorted():
    assert binary_search(np.array([]), 1.0) == (0, False)
    with pytest.raises(ValueError):
        binary_search([3, 1, 2], 1, check_sorted=True)
    with pytest.raises(ValueError):
        binary_search([1, 2], 1, side="middle")
