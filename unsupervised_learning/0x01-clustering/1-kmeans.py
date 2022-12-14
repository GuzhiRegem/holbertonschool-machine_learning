#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def get_clusters(X, C):
    clus = C[:, np.newaxis]
    return np.argmin(np.sqrt(np.sum(np.square(X - clus), axis=2)), axis=0)


def kmeans(X, k, iterations=1000):
    n, d = X.shape
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    C = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))
    for _ in range(iterations):
        clss = np.argmin(np.sum((X[:, None, :] - C)**2, axis=2), axis=1)
        C_new = np.array([
            X[clss == c].mean(axis=0)
            if np.any(clss == c)
            else min_vals + (max_vals - min_vals) * np.random.random(d)
            for c in range(k)
        ])
        if np.all(C == C_new):
            break
        C = C_new
    return C, clss
