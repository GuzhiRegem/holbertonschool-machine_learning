#!/usr/bin/env python3
"""
    module
"""


def matrix_shape(matrix):
    """ matrix shape """
    out = []
    act = matrix
    while (type(act) != int):
        out.append(len(act))
        act = act[0]
    return out
