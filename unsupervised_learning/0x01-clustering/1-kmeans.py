#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def get_clusters(X, C):
    clus = C[:, np.newaxis]
    return np.argmin(np.sqrt(np.sum(np.square(X - clus), axis=2)), axis=0)


def kmeans(X, k, iterations=1000):
    """ initialize """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(k) is not int or k <= 0:
        return None
    n, d = X.shape
    a_min = np.min(X, 0)
    a_max = np.max(X, 0)
    out = np.random.uniform(a_min, a_max, size=(k, d))
    idxs = get_clusters(X, out)
    for ite in range(iterations):
        for c in range(k):
            eq = idxs == c
            if X[eq].size == 0:
                out[c] = np.random.uniform(a_min, a_max, size=(1, d))
            else:
                out[c] = X[eq].mean(axis=0)
        idxs = get_clusters(X, out)
    return out, idxs
