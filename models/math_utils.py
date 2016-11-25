"""
Collection of utilities for the models
Author: Eli Ben-Michael
"""

import numpy as np


def sigmoid(x):
    """Compute the sigmoid function"""
    return(1 / (1 + np.exp(-x)))


def sigmoid_prime(x):
    """Compute the first derivative of the sigmoid function"""
    return(sigmoid(x) * (1 - sigmoid(x)))


def sigmoid_double_prime(x):
    """Compute the second derivative of the sigmoid function"""
    return(sigmoid_prime(x) * (1 - sigmoid(x)) - sigmoid(x) * sigmoid_prime(x))


def gradient_descent(x0, gradient, tol=1e-5):
    """Quick and dirty gradient descent implementation
    Args:
        x0: ndarray, starting position
        gradient: func, function to compute the gradient at a value
        tol: float, tolerance for convergence, default 1e-5
    Returns:
       value: ndarray, optimal x through gradient descent
    """
    converged = False
    value = x0
    iter_num = 0
    while not converged:
        iter_num += 1
        grad = gradient(value)
        learning_rate = .1 / iter_num
        value = value - learning_rate * grad

        converged = np.max(np.abs(grad)) <= tol
        if iter_num > 500:
            iter_num = 0
    return(value)
