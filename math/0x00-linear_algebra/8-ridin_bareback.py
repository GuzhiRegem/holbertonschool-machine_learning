#!/usr/bin/env python3
"""
    module
"""


def mat_mul(mat1, mat2):
    """ mat mul """
    sh1 = [len(mat1), len(mat1[0])]
    sh2 = [len(mat2), len(mat2[0])]
    if sh1[1] != sh2[0]:
        return None
    out = [[0 for i in range(sh2[1])] for j in range(sh1[0])]
    for y in range(len(out)):
        for x in range(len(out[0])):
            for i in range(sh1[1]):
                out[y][x] += mat1[y][i] * mat2[i][x]
    return out
