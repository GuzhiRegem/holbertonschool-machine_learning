#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def regular(P):
    """ markov_chain regular """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    n, n_check = P.shape
    if n != n_check:
        return None
    if not (P > 0).all():
        return None
    Q = P - np.identity(n)
    ones = np.ones((n,))
    Qe = np.c_[Q, ones]
    QTQ = np.matmul(Qe, Qe.T)
    res = np.linalg.solve(QTQ, ones)
    return np.expand_dims(res, axis=0)
