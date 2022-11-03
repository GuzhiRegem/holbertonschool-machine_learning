#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def definiteness(matrix):
    """ definiteness """
    if type(matrix) != np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")
    n = matrix.shape[0]
    out = []
    try:
        out.append(np.linalg.det(matrix))
    except Exception as e:
        return None
    for i in range(1, n):
        out.append(np.linalg.det(matrix[i:, i:]))
    true = False
    false = False
    zeros = False
    for i in out:
        if i > 0:
            true = True
        if i < 0:
            false = True
        if i == 0:
            zeros = True
    if true and not (false or zeros):
        return "Positive definite"
    if false and not (true or zeros):
        return "Negative definite"
    if zeros:
        if true and not false:
            return "Positive semi-definite"
        if false and not true:
            return "Negative semi-definite"
    return "Indefinite"