#!/usr/bin/env python3
"""
    module
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """ cat matrix """
    sh1 = [len(mat1), len(mat1[0])]
    sh2 = [len(mat2), len(mat2[0])]
    if sh1[1 - axis] != sh2[1 - axis]:
        return None
    if axis == 0:
        out = []
        for r in mat1:
            out.append(r[:])
        for r in mat2:
            out.append(r[:])
    if axis == 1:
        out = []
        for r in range(sh1[0]):
            out.append(mat1[r] + mat2[r])
    return out
