#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """ kmeans """
    n, d = X.shape
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    C = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))
    for _ in range(iterations):
        clss = np.argmin(np.sum((X[:, None, :] - C)**2, axis=2), axis=1)
        C_new = np.zeros(shape=(k, X.shape[1]))
        for c in range(k):
            if not len(X[clss == c]):
                C_new[c] = np.random.uniform(low=min_vals, high=max_vals)
            else:
                C_new[c] = X[clss == c].mean(axis=0)
        if np.all(C == C_new):
            break
        C = C_new
    return C, clss
