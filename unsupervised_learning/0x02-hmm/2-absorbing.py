#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def absorbing(P):
    """ markov_chain regular """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return False
    # save value of n and check that P is square
    n, n_check = P.shape
    if n != n_check:
        return False
    return True
