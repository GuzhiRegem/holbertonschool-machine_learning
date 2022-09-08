#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ l2 gradient """
    m = Y.shape[1]
    W_copy = weights.copy()
    for i in reversed(range(1, L + 1)):
        ka0 = "A{}".format(i - 1)
        ka1 = "A{}".format(i)
        kw0 = "W{}".format(i)
        kw1 = "W{}".format(i + 1)
        kb = "b{}".format(i)
        if i == L:
            dZ = cache[ka1] - Y
            dW = (np.matmul(cache[ka0], dZ.T) / m).T
        else:
            dW2 = np.matmul(W_copy[kw1].T, dZ2)
            A = cache[ka1]
            tanh = 1 - (A * A)
            dZ = dW2 * tanh
            dW = np.matmul(dZ, cache[ka0].T) / m
        dW_L2 = dW + (lambtha / m) * W_copy[kw0]
        db = np.sum(dZ, axis=1, keepdims=True) / m
        weights[kw0] -= alpha * dW_L2
        weights[kb] -= alpha * db
        dZ2 = dZ
