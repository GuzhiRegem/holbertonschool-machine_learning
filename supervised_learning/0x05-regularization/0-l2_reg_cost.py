#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ l2 cost """
    n = 0
    for i in range(L):
        w = weights["W" + str(i + 1)]
        n += np.linalg.norm(w) ** 2
    out = (lambtha / (2 * m)) * n
    return cost + out