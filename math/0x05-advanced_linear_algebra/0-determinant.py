#!/usr/bin/env python3
"""
    module
"""


def copy_matrix(matrix):
    """ copy func """
    out = []
    for val in matrix:
        out.append(val[:])
    return out


def determinant_fast(A):
    """ determinant fast """
    n = len(A)
    AM = copy_matrix(A)
    for fd in range(n):
        for i in range(fd+1, n):
            if AM[fd][fd] == 0:
                AM[fd][fd] == 1.0e-18
            crScaler = AM[i][fd] / AM[fd][fd]
            for j in range(n):
                AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
    product = 1.0
    for i in range(n):
        product *= AM[i][i]
    return product


def determinant(matrix):
    """ determinant """
    if (type(matrix) != list):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if type(matrix[0]) != list:
        raise TypeError("matrix must be a list of lists")
    if matrix[0] == []:
        return 1
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")
    res = determinant_fast(matrix)
    return int(res)
