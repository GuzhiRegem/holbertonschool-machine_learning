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
    product = 1
    for i in range(n):
        product *= AM[i][i]
    return product


def determinant(matrix):
    """ determinant """
    if (type(matrix) != list):
        raise TypeError("matrix must be a list of lists")
    n = len(matrix)
    if n == 0:
        raise TypeError("matrix must be a list of lists")
    if n == 1:
        if matrix[0] == []:
            return 1
        if type(matrix[0]) != list:
            raise TypeError("matrix must be a list of lists")
        return matrix[0][0]
    for val in matrix:
        if type(val) != list:
            raise TypeError("matrix must be a list of lists")
        if len(val) != n:
            raise ValueError("matrix must be a square matrix")
    res = determinant_fast(matrix)
    return int(res) if int(res) == res else res
