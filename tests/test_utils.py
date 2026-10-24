import numpy as np

from numcompute.utils import logsumexp, softmax

def test_logsumexp_basic():
    x = np.array([1.0, 2.0, 3.0])

    result = logsumexp(x)

    expected = np.log(
        np.exp(1.0) +
        np.exp(2.0) +
        np.exp(3.0)
    )

    assert np.allclose(result, expected)

def test_logsumexp_large_values():
    x = np.array([1000.0, 1001.0, 1002.0])

    result = logsumexp(x)

    assert np.isfinite(result)

def test_softmax_sums_to_one():
    x = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])

    result = softmax(x, axis=1)

    assert np.allclose(
        np.sum(result, axis=1),
        [1.0, 1.0]
    )