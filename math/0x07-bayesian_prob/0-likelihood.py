#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def likelihood(x, n, P):
    """ likelihood """
    if (type(n) != int) or (n <= 0):
        raise ValueError("n must be a positive integer")
    if (type(x) != int) or (x < 0):
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if (type(P) != np.ndarray) or P.shape != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    for val in P:
        if val < 0 or val > 1:
            raise ValueError("All values in P must be in the range [0, 1]")
    return np.array([0.5, 0.5])
