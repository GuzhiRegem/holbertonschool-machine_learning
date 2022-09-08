#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ l2 gradient """
    m = Y.shape[1]
    dZ = cache["A" + str(L)] - Y
    for lay in range(L, 0, -1):
        A = cache["A" + str(lay - 1)]
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dW = np.dot(dZ, A.T) / m
        dZ = np.dot(weights["W" + str(lay)].T, dZ) * (1 - (A ** 2))
        dW_L2 = dW + (lambtha / m) * weights["W" + str(lay)]
        weights["W" + str(lay)] -= dW_L2 * alpha
        weights["b" + str(lay)] -= db * alpha
