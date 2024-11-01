import numpy as np
import pytest
from linreg import PolyRegression

def test_MSE():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    coeffs = np.array([1, 1])
    l1 = 0.1
    l2 = 0.2

    poly_reg = PolyRegression(x, y)
    mse = poly_reg.MSE(x, y, coeffs, l1, l2)

    expected_mse = 0.0
    assert np.isclose(mse, expected_mse), f"Expected MSE: {expected_mse}, but got: {mse}"

def test_grad_descent():
    x = np.array([1, 2, 3, 4, 5]).astype(np.float64)
    print(x)
    y = np.array([2, 4, 6, 8, 10]).astype(np.float64)
    coeffs = np.array([0, 0])
    max_iter = 10000
    learning_rate = 0.01
    tolerance = 0.0001

    poly_reg = PolyRegression(x, y)
    updated_coeffs = poly_reg.grad_descent(x, y, coeffs, max_iter, learning_rate, tolerance)

    expected_coeffs = np.array([1.999, 0.0])
    assert np.allclose(updated_coeffs, expected_coeffs), f"Expected coefficients: {expected_coeffs}, but got: {updated_coeffs}"

test_grad_descent()