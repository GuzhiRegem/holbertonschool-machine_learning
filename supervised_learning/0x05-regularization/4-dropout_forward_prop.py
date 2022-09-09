#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """ dropout """
    cache = {"A0": X}
    for lay in range(1, L + 1):
        prev = cache["A" + str(lay - 1)]
        w = weights["W" + str(lay)]
        b = weights["b" + str(lay)]
        Z = np.dot(w, prev) + b
        if lay == L:
            t = np.exp(Z)
            A = t / sum(t)
        else:
            A = (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
        cache["A" + str(lay)] = A
    return cache