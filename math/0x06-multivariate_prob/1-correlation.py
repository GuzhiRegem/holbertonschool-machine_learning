#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def correlation(C):
    """ correlation """
    if type(C) is not np.ndarray:
        raise TypeError("X must be a 2D numpy.ndarray")
    d = C.shape[0]
    if len(C.shape) != 2 or d != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    std = np.sqrt(np.diag(C))
    return C / np.outer(std, std)
