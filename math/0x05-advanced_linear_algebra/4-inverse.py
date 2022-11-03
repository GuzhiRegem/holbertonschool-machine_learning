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
    return round(res)


def minor(matrix):
    """minor"""
    if (type(matrix) != list):
        raise TypeError("matrix must be a list of lists")
    n = len(matrix)
    if n == 0:
        raise TypeError("matrix must be a list of lists")
    for val in matrix:
        if type(val) != list:
            raise TypeError("matrix must be a list of lists")
        if len(val) != n:
            raise ValueError("matrix must be a non-empty square matrix")
    if n == 1:
        return [[1]]
    out = copy_matrix(matrix)
    for y in range(n):
        for x in range(n):
            tmp = []
            for y2 in range(n):
                if y2 == y:
                    continue
                row = []
                for x2 in range(n):
                    if x2 == x:
                        continue
                    row.append(matrix[y2][x2])
                tmp.append(row)
            out[y][x] = determinant(tmp)
    return out


def cofactor(matrix):
    """ cofactor """
    minor_matrix = minor(matrix)
    n = len(matrix)
    for y in range(n):
        for x in range(n):
            minor_matrix[y][x] *= pow(-1, y + x + 2)
    return minor_matrix


def matrix_transpose(matrix):
    """ matrix transpose """
    sh = [len(matrix), len(matrix[0])]
    out = [[0 for i in range(sh[0])] for j in range(sh[1])]
    for x in range(sh[0]):
        for y in range(sh[1]):
            out[y][x] = matrix[x][y]
    return out


def inverse(matrix):
    """ inverse """
    adjugate = matrix_transpose(cofactor(matrix))
    det = 1 / determinant(matrix)
    n = len(matrix)
    for y in range(n):
        for x in range(n):
            adjugate[y][x] *= det
    return adjugate
