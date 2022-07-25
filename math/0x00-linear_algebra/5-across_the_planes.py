#!/usr/bin/env python3
"""
    module
"""


def add_matrices2D(mat1, mat2):
    if len(mat1) != len(mat2):
        return None
    if len(mat1[0]) != len(mat2[0]):
        return None
    out = []
    for y in range(len(mat1)):
        row = []
        for x in range(len(mat1[0])):
            row.append(mat1[y][x] + mat2[y][x])
        out.append(row)
    return out
