#!/usr/bin/env python3
"""
    module
"""


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
    n = len(matrix)
    temp = [0]*n  # temporary array for storing row
    total = 1
    det = 1
    for i in range(0, n):
        index = i  # initialize the index
        while(index < n and matrix[index][i] == 0):
            index += 1
        if(index == n):  # if there is non zero element
            continue
        if(index != i):
            for j in range(0, n):
                matrix[index][j], matrix[i][j] = matrix[i][j], matrix[index][j]
            det = det*int(pow(-1, index-i))
        for j in range(0, n):
            temp[j] = matrix[i][j]
        for j in range(i+1, n):
            num1 = temp[i]     # value of diagonal element
            num2 = matrix[j][i]   # value of next row element
            for k in range(0, n):
                matrix[j][k] = (num1*matrix[j][k]) - (num2*temp[k])
            total = total * num1  # Det(kA)=kDet(A);
    for i in range(0, n):
        det = det*matrix[i][i]

    return int(det/total)  # Det(kA)/k=Det(A);
