#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def initialize(X, k):
    """ initialize """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(k) is not int or k <= 0:
        return None
    n, d = X.shape
    a_min = np.min(X, 0)
    a_max = np.max(X, 0)
    return np.random.uniform(a_min, a_max, size=(k, d))
