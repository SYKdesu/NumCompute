"""Descriptive statistics, histograms, quantiles and streaming statistics.

Axis convention (same for every function here)
----------------------------------------------
``axis=None`` reduces over all elements and returns a Python ``float``.
``axis=k`` reduces over dimension ``k`` and returns an array whose shape is
``data.shape`` with dimension ``k`` removed. For a data matrix of shape
``(n_samples, n_features)``, ``axis=0`` gives one value per feature.

NaN policy (same for every function that accepts ``nan_policy``)
-----------------------------------------------------------------
'propagate' (default): any NaN in a slice makes that result NaN.
'omit'               : NaNs are ignored; an all-NaN slice gives NaN.
'raise'              : ValueError if any NaN is present.

Functions / classes
-------------------
mean, median, var, std, minimum, maximum, describe, quantile, histogram,
RunningStats (Welford / Chan streaming mean and variance).
"""

from __future__ import annotations

import warnings

import numpy as np

from .rank import _quantile

__all__ = [
    "mean",
    "median",
    "var",
    "std",
    "minimum",
    "maximum",
    "describe",
    "quantile",
    "histogram",
    "RunningStats",
]

_NAN_POLICIES = ("propagate", "omit", "raise")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _prepare(data, axis, nan_policy):
    """Validate input and return a float64 array."""
    if nan_policy not in _NAN_POLICIES:
        raise ValueError(f"nan_policy must be one of {_NAN_POLICIES}, got {nan_policy!r}")
    a = np.asarray(data)
    if not (np.issubdtype(a.dtype, np.number) or a.dtype == bool):
        raise TypeError(f"data must be numeric, got dtype {a.dtype}")
    if np.iscomplexobj(a):
        raise TypeError("complex data is not supported")
    a = a.astype(np.float64)
    if axis is not None:
        if a.ndim == 0:
            raise ValueError("axis given but data is a scalar")
        if not -a.ndim <= axis < a.ndim:
            raise np.exceptions.AxisError(axis, a.ndim)
        if a.shape[axis] == 0:
            raise ValueError(f"cannot reduce over empty axis {axis} (shape {a.shape})")
    elif a.size == 0:
        raise ValueError("cannot compute statistics of an empty array")
    if nan_policy == "raise" and np.isnan(a).any():
        raise ValueError("data contains NaN (nan_policy='raise')")
    return a


def _finish(res):
    """Return a Python float for 0-d results, otherwise a float64 ndarray."""
    res = np.asarray(res, dtype=np.float64)
    return float(res) if res.ndim == 0 else res


def _reduce(func, nanfunc, data, axis, nan_policy, **kw):
    a = _prepare(data, axis, nan_policy)
    f = nanfunc if nan_policy == "omit" else func
    # All-NaN slices with 'omit' and ddof >= n produce NaN on purpose;
    # silence NumPy's RuntimeWarnings for those documented cases.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _finish(f(a, axis=axis, **kw))


# --------------------------------------------------------------------------
# descriptive statistics
# --------------------------------------------------------------------------
def mean(data, axis=None, nan_policy="propagate"):
    """Arithmetic mean.

    Parameters
    ----------
    data : array_like (numeric)
    axis : int or None, default None
    nan_policy : {'propagate', 'omit', 'raise'}

    Returns
    -------
    float or ndarray (shape = data.shape without ``axis``)

    Raises
    ------
    ValueError Empty input / empty axis, bad nan_policy, NaN with 'raise'.
    TypeError  Non-numeric input.
    AxisError  Invalid axis.

    Notes
    -----
    Time O(n), space O(1) extra beyond the float64 copy.
    """
    return _reduce(np.mean, np.nanmean, data, axis, nan_policy)


def median(data, axis=None, nan_policy="propagate"):
    """Median (50th percentile, linear interpolation for even n).

    Same parameters, returns and exceptions as :func:`mean`.
    Time O(n) average (NumPy uses partitioning), space O(n).
    """
    return _reduce(np.median, np.nanmedian, data, axis, nan_policy)


def var(data, axis=None, ddof=0, nan_policy="propagate"):
    """Variance ``sum((x - mean)**2) / (n - ddof)``.

    Parameters
    ----------
    ddof : int, default 0
        0 gives the population variance, 1 the sample variance.
        If ``n - ddof <= 0`` the result is NaN.

    Other parameters, returns and exceptions as :func:`mean`.
    Time O(n), space O(n).
    """
    if ddof < 0:
        raise ValueError(f"ddof must be >= 0, got {ddof}")
    return _reduce(np.var, np.nanvar, data, axis, nan_policy, ddof=ddof)


def std(data, axis=None, ddof=0, nan_policy="propagate"):
    """Standard deviation, ``sqrt(var(data, ddof))``. See :func:`var`."""
    if ddof < 0:
        raise ValueError(f"ddof must be >= 0, got {ddof}")
    return _reduce(np.std, np.nanstd, data, axis, nan_policy, ddof=ddof)


def minimum(data, axis=None, nan_policy="propagate"):
    """Smallest value. See :func:`mean`. Time O(n)."""
    return _reduce(np.min, np.nanmin, data, axis, nan_policy)


def maximum(data, axis=None, nan_policy="propagate"):
    """Largest value. See :func:`mean`. Time O(n)."""
    return _reduce(np.max, np.nanmax, data, axis, nan_policy)


def quantile(data, q, axis=None, interpolation="linear", nan_policy="propagate"):
    """q-th quantile, with ``q`` in [0, 1].

    ``quantile(x, 0.25)`` equals ``rank.percentile(x, 25)``; both use the
    same engine, so the interpolation rules are identical
    ('linear', 'lower', 'higher', 'midpoint', 'nearest').

    Returns
    -------
    float, or ndarray of shape ``q.shape + reduced_shape``.

    Raises
    ------
    ValueError If any q is outside [0, 1], input is empty, an option is
               invalid, or NaN is found with ``nan_policy='raise'``.

    Notes
    -----
    Time O(n log n) per slice, space O(n * len(q)).
    """
    return _quantile(data, q, interpolation, axis, nan_policy)


def describe(data, axis=0, ddof=1, nan_policy="propagate"):
    """Summary statistics in one call.

    Parameters
    ----------
    data : array_like
        Typically shape (n_samples, n_features) with ``axis=0``.
    axis : int or None, default 0
    ddof : int, default 1 (sample standard deviation)
    nan_policy : {'propagate', 'omit', 'raise'}

    Returns
    -------
    dict with keys 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'.
    'count' is the number of non-NaN values; the others are floats or
    arrays of shape ``data.shape`` without ``axis``.
    """
    a = _prepare(data, axis, nan_policy)
    count = np.sum(~np.isnan(a), axis=axis)
    q = quantile(a, [0.25, 0.5, 0.75], axis=axis, nan_policy=nan_policy)
    return {
        "count": int(count) if np.ndim(count) == 0 else count,
        "mean": mean(a, axis, nan_policy),
        "std": std(a, axis, ddof, nan_policy),
        "min": minimum(a, axis, nan_policy),
        "25%": _finish(q[0]),
        "50%": _finish(q[1]),
        "75%": _finish(q[2]),
        "max": maximum(a, axis, nan_policy),
    }


# --------------------------------------------------------------------------
# histogram
# --------------------------------------------------------------------------
def histogram(data, bins=10, range=None, density=False):
    """Count how many values fall into each bin.

    Bins are half-open ``[e_i, e_{i+1})`` except the last, which also
    includes its right edge (same convention as ``np.histogram``).

    Parameters
    ----------
    data : array_like (any shape, flattened)
        NaNs are always ignored.
    bins : int or array_like of float, default 10
        Number of equal-width bins, or the bin edges (1-D, increasing,
        at least 2 values).
    range : (float, float) or None
        Lower and upper edge when ``bins`` is an int. Default is
        ``(min, max)`` of the data. Values outside are ignored.
    density : bool, default False
        If True, return a probability density (area sums to 1).

    Returns
    -------
    counts : ndarray, shape (n_bins,)
        int64, or float64 if ``density``.
    edges : ndarray of float64, shape (n_bins + 1,)

    Raises
    ------
    ValueError Invalid ``bins``/``range``, or infinite values when the
               range must be inferred.
    TypeError  Non-numeric input.

    Notes
    -----
    If all values are equal the range is widened to ``(v - 0.5, v + 0.5)``.
    Bin indices are computed arithmetically for equal-width bins (O(n)) or
    with ``np.searchsorted`` for custom edges (O(n log b)); counts come
    from ``np.bincount``. Space O(n + b).
    """
    a = np.asarray(data)
    if not (np.issubdtype(a.dtype, np.number) or a.dtype == bool):
        raise TypeError(f"data must be numeric, got dtype {a.dtype}")
    a = a.astype(np.float64).ravel()
    a = a[~np.isnan(a)]

    if isinstance(bins, (int, np.integer)) and not isinstance(bins, bool):
        nb = int(bins)
        if nb < 1:
            raise ValueError(f"bins must be >= 1, got {nb}")
        if range is None:
            if a.size and not np.isfinite(a).all():
                raise ValueError("data contains inf; pass an explicit range")
            lo, hi = (a.min(), a.max()) if a.size else (0.0, 1.0)
        else:
            lo, hi = float(range[0]), float(range[1])
            if not (np.isfinite(lo) and np.isfinite(hi)) or lo > hi:
                raise ValueError(f"range must be finite with min <= max, got {range}")
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        edges = np.linspace(lo, hi, nb + 1)
        x = a[(a >= lo) & (a <= hi)]
        idx = ((x - lo) / (hi - lo) * nb).astype(np.intp)
        idx = np.clip(idx, 0, nb - 1)
        # correct floating-point rounding at bin boundaries
        idx -= (x < edges[idx]).astype(np.intp)
        idx += ((x >= edges[idx + 1]) & (idx != nb - 1)).astype(np.intp)
    else:
        edges = np.asarray(bins, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("bin edges must be a 1-D array with at least 2 values")
        if np.any(np.diff(edges) <= 0):
            raise ValueError("bin edges must be strictly increasing")
        nb = edges.size - 1
        x = a[(a >= edges[0]) & (a <= edges[-1])]
        idx = np.searchsorted(edges, x, side="right") - 1
        idx[idx == nb] = nb - 1  # right edge belongs to the last bin

    counts = np.bincount(idx, minlength=nb).astype(np.int64)
    if density:
        total = counts.sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            return counts / (total * np.diff(edges)), edges
    return counts, edges


# --------------------------------------------------------------------------
# streaming statistics
# --------------------------------------------------------------------------
class RunningStats:
    """Streaming mean / variance / min / max (Welford + Chan's merge).

    Feed data in batches with :meth:`update`; the statistics equal those of
    all data seen so far, without keeping the data in memory. Each batch is
    processed with vectorised NumPy, then merged with the running totals:

        delta = mean_b - mean
        mean  = mean + delta * n_b / (n + n_b)
        M2    = M2 + M2_b + delta**2 * n * n_b / (n + n_b)

    This is numerically stable (no ``sum(x**2) - n*mean**2`` cancellation).

    Parameters
    ----------
    ddof : int, default 0
        Delta degrees of freedom used by :attr:`var` and :attr:`std`.

    Examples
    --------
    >>> rs = RunningStats()
    >>> for chunk in np.array_split(np.arange(10.0), 3):
    ...     rs.update(chunk)
    >>> rs.mean, rs.count
    (4.5, 10)

    Notes
    -----
    A batch of shape (n_b,) is n_b scalar samples. A batch of shape
    (n_b, d1, ...) is n_b samples of shape (d1, ...); statistics then have
    shape (d1, ...). Time O(n_b) per update, space O(d).
    """

    def __init__(self, ddof=0):
        if ddof < 0:
            raise ValueError(f"ddof must be >= 0, got {ddof}")
        self.ddof = ddof
        self._n = 0
        self._mean = None
        self._m2 = None
        self._min = None
        self._max = None

    def update(self, batch):
        """Add a batch of samples (first axis = samples). Returns ``self``.

        Raises
        ------
        ValueError If the batch contains NaN or its feature shape differs
                   from earlier batches.
        TypeError  If the batch is not numeric.
        """
        x = np.asarray(batch)
        if not (np.issubdtype(x.dtype, np.number) or x.dtype == bool):
            raise TypeError(f"batch must be numeric, got dtype {x.dtype}")
        x = np.atleast_1d(x.astype(np.float64))
        if np.isnan(x).any():
            raise ValueError("batch contains NaN; impute or drop NaNs first")
        nb = x.shape[0]
        if nb == 0:
            return self
        if self._n and x.shape[1:] != self._mean.shape:
            raise ValueError(
                f"feature shape {x.shape[1:]} does not match earlier {self._mean.shape}"
            )
        b_mean = x.mean(axis=0)
        b_m2 = ((x - b_mean) ** 2).sum(axis=0)
        b_min, b_max = x.min(axis=0), x.max(axis=0)
        if self._n == 0:
            self._mean, self._m2, self._min, self._max = b_mean, b_m2, b_min, b_max
        else:
            tot = self._n + nb
            delta = b_mean - self._mean
            self._mean = self._mean + delta * (nb / tot)
            self._m2 = self._m2 + b_m2 + delta**2 * (self._n * nb / tot)
            self._min = np.minimum(self._min, b_min)
            self._max = np.maximum(self._max, b_max)
        self._n += nb
        return self

    def _check(self):
        if self._n == 0:
            raise ValueError("no data seen yet; call update() first")

    @property
    def count(self):
        """Number of samples seen."""
        return self._n

    @property
    def mean(self):
        self._check()
        return _finish(self._mean)

    @property
    def var(self):
        self._check()
        denom = self._n - self.ddof
        return _finish(self._m2 / denom if denom > 0 else np.full_like(self._m2, np.nan))

    @property
    def std(self):
        return _finish(np.sqrt(self.var))

    @property
    def min(self):
        self._check()
        return _finish(self._min)

    @property
    def max(self):
        self._check()
        return _finish(self._max)
