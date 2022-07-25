#!/usr/bin/env python3
"""
    module
"""


def matrix_transpose(matrix):
    """ matrix transpose """
    sh = [len(matrix), len(matrix[0])]
    out = [[0 for i in range(sh[0])] for j in range(sh[1])]
    for x in range(sh[0]):
        for y in range(sh[1]):
            out[y][x] = matrix[x][y]
    return out
