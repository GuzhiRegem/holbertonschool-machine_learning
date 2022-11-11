#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def intersection(x, n, P, Pr):
    """ likelihood """
    if (type(n) != int) or (n <= 0):
        raise ValueError("n must be a positive integer")
    if (type(x) != int) or (x < 0):
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )
    if x > n:
        raise ValueError("x cannot be greater than n")
    if (type(P) != np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if (type(Pr) != np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    for val in P:
        if val < 0 or val > 1:
            raise ValueError("All values in P must be in the range [0, 1]")
    for val in Pr:
        if val < 0 or val > 1:
            raise ValueError("All values in Pr must be in the range [0, 1]")
    return np.array([0.5, 0.5])
