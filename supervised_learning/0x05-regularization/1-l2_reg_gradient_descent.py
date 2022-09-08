#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ l2 gradient """
    m = Y.shape[1]
    for i in range(L, 0, -1):
        kw0 = "W" + str(i)
        kw1 = "W" + str(i + 1)
        kb = "b" + str(i)
        A = cache["A" + str(i)]
        prev = cache["A" + str(i - 1)]
        if i == L:
            dZ = A - Y
            dW = (np.matmul(prev, dZ.T) / m).T
        else:
            dW2 = np.matmul(weights[kw1].T, dZ2)
            tanh = 1 - (A * A)
            dZ = dW2 * tanh
            dW = np.matmul(dZ, prev.T) / m
        dW_L2 = dW + (lambtha / m) * weights[kw0]
        db = np.sum(dZ, axis=1, keepdims=True) / m
        weights[kw0] -= alpha * dW_L2
        weights[kb] -= alpha * db
        dZ2 = dZ