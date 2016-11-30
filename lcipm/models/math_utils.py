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


def gradient_descent(x0, gradient, tol=1e-5, max_iter=100, print_grad=False,
                     learning_rate=1):
    """Quick and dirty gradient descent implementation
    Args:
        x0: ndarray, starting position
        gradient: func, function to compute the gradient at a value
        tol: float, tolerance for convergence, default 1e-5
        max_iter: int, max number of iterations
        print_grad: boolean, whether to print the gradient at each iteration
        learning_rate: float, step size is n ** learning_rate
    Returns:
       value: ndarray, optimal x through gradient descent
    """
    converged = False
    value = x0
    iter_num = 0
    prev_value = x0 - .1
    prev_grad = np.ones(x0.shape)
    while not converged:
        grad = gradient(value) 
        # step = .1 / (iter_num + 1) ** learning_rate
        # compute the step size
        if iter_num == 0:
            step = 1e-5
        else:
            deltax = value - prev_value
            deltag = grad - prev_grad
            if iter_num % 2 == 0:
                step = (np.dot(deltag, deltax) / np.dot(deltag, deltag))
            else:
                step = np.dot(deltax, deltax) / np.dot(deltax, deltag)
        step = np.abs(step) 
        prev_value = value
        value = value - step * grad
        prev_grad = grad

        converged = np.max(np.abs(grad)) <= tol
        if print_grad:
            print("----")
            #print(grad)
            print(np.max(np.abs(grad)))
            print(np.sum(np.abs(grad) > tol))
            print(iter_num)
            print(step)
            #print(np.linalg.norm(deltax), np.linalg.norm(deltag))
        iter_num += 1
        if iter_num >= max_iter:
            converged = True

    return(value, grad)
