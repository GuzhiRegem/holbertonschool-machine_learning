#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def initialize(X, k):
    """ initialize """
    n, d = X.shape
    a_min = X.min()
    a_max = X.max()
    return np.random.uniform(a_min, a_max, size=(k, d))
