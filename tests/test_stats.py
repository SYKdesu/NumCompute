"""Unit tests for numcompute.stats."""

import numpy as np
import pytest

from numcompute import stats
from numcompute.rank import percentile


@pytest.fixture
def X():
    return np.random.default_rng(4).normal(size=(8, 3))


def test_basic_stats_match_numpy(X):
    for axis in (None, 0, 1):
        np.testing.assert_allclose(stats.mean(X, axis), np.mean(X, axis=axis))
        np.testing.assert_allclose(stats.median(X, axis), np.median(X, axis=axis))
        np.testing.assert_allclose(stats.std(X, axis, ddof=1), np.std(X, axis=axis, ddof=1))
        np.testing.assert_allclose(stats.minimum(X, axis), np.min(X, axis=axis))
        np.testing.assert_allclose(stats.maximum(X, axis), np.max(X, axis=axis))


def test_axis_shapes_and_scalar_type(X):
    assert isinstance(stats.mean(X), float)
    assert stats.mean(X, axis=0).shape == (3,)
    assert stats.mean(X, axis=1).shape == (8,)
    assert stats.quantile(X, [0.1, 0.9], axis=0).shape == (2, 3)


def test_nan_policies():
    a = np.array([[1.0, np.nan], [3.0, 4.0]])
    np.testing.assert_allclose(stats.mean(a, axis=0), [2.0, np.nan])
    np.testing.assert_allclose(stats.mean(a, axis=0, nan_policy="omit"), [2.0, 4.0])
    with pytest.raises(ValueError):
        stats.mean(a, nan_policy="raise")


def test_all_nan_slice_with_omit_gives_nan_without_warning():
    a = np.array([[np.nan, 1.0], [np.nan, 2.0]])
    with np.testing.assert_no_warnings():
        out = stats.mean(a, axis=0, nan_policy="omit")
    np.testing.assert_allclose(out, [np.nan, 1.5])


def test_zero_variance_and_ddof():
    a = np.full(5, 3.0)
    assert stats.var(a) == 0.0
    assert stats.std(a, ddof=1) == 0.0
    assert np.isnan(stats.var([1.0], ddof=1))  # n - ddof = 0


def test_empty_and_bad_input():
    with pytest.raises(ValueError):
        stats.mean([])
    with pytest.raises(ValueError):
        stats.mean(np.empty((0, 3)), axis=0)
    with pytest.raises(TypeError):
        stats.mean(["a", "b"])
    with pytest.raises(np.exceptions.AxisError):
        stats.mean(np.ones((2, 2)), axis=2)


def test_quantile_agrees_with_percentile(X):
    np.testing.assert_allclose(
        stats.quantile(X, [0.25, 0.75], axis=0), percentile(X, [25, 75], axis=0)
    )
    with pytest.raises(ValueError):
        stats.quantile(X, 1.5)


def test_describe(X):
    d = stats.describe(X, axis=0)
    np.testing.assert_allclose(d["50%"], np.median(X, axis=0))
    np.testing.assert_allclose(d["std"], np.std(X, axis=0, ddof=1))
    np.testing.assert_array_equal(d["count"], [8, 8, 8])


@pytest.mark.parametrize("bins", [1, 5, 13])
def test_histogram_matches_numpy(bins):
    a = np.random.default_rng(5).normal(size=500)
    c, e = stats.histogram(a, bins)
    c_np, e_np = np.histogram(a, bins)
    np.testing.assert_array_equal(c, c_np)
    np.testing.assert_allclose(e, e_np)


def test_histogram_custom_edges_range_density():
    a = np.array([0, 1, 1, 2, 3, 3, 3, 4.0])
    edges = [0, 1, 2, 4]
    np.testing.assert_array_equal(stats.histogram(a, edges)[0], np.histogram(a, edges)[0])
    c, _ = stats.histogram(a, 4, range=(1, 3))
    np.testing.assert_array_equal(c, np.histogram(a, 4, range=(1, 3))[0])
    d, e = stats.histogram(a, 4, density=True)
    assert np.isclose(np.sum(d * np.diff(e)), 1.0)


def test_histogram_all_equal_nan_and_errors():
    c, e = stats.histogram([2.0, 2.0, np.nan], 3)
    assert c.sum() == 2
    np.testing.assert_allclose([e[0], e[-1]], [1.5, 2.5])
    with pytest.raises(ValueError):
        stats.histogram([1, 2], 0)
    with pytest.raises(ValueError):
        stats.histogram([1, 2], [0, 2, 1])


def test_running_stats_matches_batch():
    rng = np.random.default_rng(6)
    data = rng.normal(loc=1e6, scale=1.0, size=(1000, 2))  # large offset: stability
    rs = stats.RunningStats(ddof=1)
    for chunk in np.array_split(data, 7):
        rs.update(chunk)
    assert rs.count == 1000
    np.testing.assert_allclose(rs.mean, data.mean(axis=0))
    np.testing.assert_allclose(rs.var, data.var(axis=0, ddof=1), rtol=1e-9)
    np.testing.assert_allclose(rs.min, data.min(axis=0))


def test_running_stats_errors():
    rs = stats.RunningStats()
    with pytest.raises(ValueError):
        rs.mean
    rs.update([1.0, 2.0])
    with pytest.raises(ValueError):
        rs.update(np.ones((2, 3)))
    with pytest.raises(ValueError):
        rs.update([np.nan])
