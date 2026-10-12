import numpy as np

def logsumexp(x, axis=None, keepdims=False):
    x = np.asarray(x, dtype=float)

    max_x = np.max(x, axis=axis, keepdims=True)

    result = max_x + np.log(
        np.sum(
            np.exp(x - max_x),
            axis=axis,
            keepdims=True
        )
    )

    if not keepdims:
        result = np.squeeze(result, axis=axis)

    return result

def softmax(x, axis=-1):
    x = np.asarray(x, dtype=float)

    shifted = x - np.max(
        x,
        axis=axis,
        keepdims=True
    )

    exp_x = np.exp(shifted)

    return exp_x / np.sum(
        exp_x,
        axis=axis,
        keepdims=True
    )

def euclidean_distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    return np.sqrt(
        np.sum((a - b) ** 2, axis=-1)
    )

def batch_iterator(X, batch_size):
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than 0."
        )

    for start in range(0, len(X), batch_size):
        yield X[start:start + batch_size]