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


def add_matrices(mat1, mat2):
    """ add matrices """
    sh1 = matrix_shape(mat1)
    sh2 = matrix_shape(mat2)
    if sh1 != sh2:
        return None
    out = []
    if len(sh1) > 1:
        for i in range(sh1[0]):
            out.append(add_matrices(mat1[i], mat2[i]))
        return out
    for i in range(sh1[0]):
        out.append(mat1[i] + mat2[i])
    return out
