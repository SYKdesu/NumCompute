import numpy as np
import pytest

from numcompute.optim import gradient, jacobian


def test_gradient_matches_quadratic_derivative():
    point = np.array([1.5, -2.0, 0.25])
    np.testing.assert_allclose(gradient(lambda x: np.sum(x**2), point), 2 * point, rtol=1e-6, atol=1e-7)


def test_forward_gradient_accepts_extra_arguments():
    point = np.array([2.0, -1.0])
    result = gradient(lambda x, scale: scale * np.sum(x**2), point, 3.0, method="forward")
    np.testing.assert_allclose(result, 6 * point, rtol=1e-5, atol=1e-5)


def test_jacobian_matches_vector_function():
    point = np.array([2.0, 3.0])
    result = jacobian(lambda x: np.array([x[0] * x[1], x[0] + x[1] ** 2]), point)
    np.testing.assert_allclose(result, [[3.0, 2.0], [1.0, 6.0]], rtol=1e-6, atol=1e-6)


def test_scalar_input_and_output_produce_scalar_shaped_array():
    result = gradient(lambda x: x**3, np.asarray(2.0))
    assert result.shape == ()
    assert result == pytest.approx(12.0)


def test_jacobian_supports_matrix_input_shape():
    point = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = jacobian(lambda x: np.array([x.sum(), np.sum(x**2)]), point)
    assert result.shape == (2, 2, 2)
    np.testing.assert_allclose(result[0], np.ones((2, 2)))
    np.testing.assert_allclose(result[1], 2 * point)


def test_gradient_rejects_vector_output():
    with pytest.raises(ValueError, match="scalar-valued"):
        gradient(lambda x: x**2, [1.0, 2.0])


def test_differentiation_does_not_mutate_input():
    point = np.array([1.0, 2.0])
    original = point.copy()
    jacobian(lambda x: x**2, point)
    np.testing.assert_array_equal(point, original)


@pytest.mark.parametrize("epsilon", [0.0, -1.0, np.inf, np.nan])
def test_invalid_epsilon_is_rejected(epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        gradient(lambda x: np.sum(x), [1.0], epsilon=epsilon)
