#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """ back prop """
    m = Y.shape[1]
    dZ = cache["A" + str(L)] - Y
    for lay in range(L, 0, -1):
        A = cache["A" + str(lay - 1)]
        db = np.sum(dZ, axis=1, keepdims=True) / m
        if lay != L:
            d_l = cache["D" + str(lay)]
            dZ = (dZ * d_l) / keep_prob
        dW = np.dot(dZ, A.T) / m
        dZ = np.dot(weights["W" + str(lay)].T, dZ) * (1 - (A ** 2))
        weights["W" + str(lay)] -= dW * alpha
        weights["b" + str(lay)] -= db * alpha
