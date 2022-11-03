#!/usr/bin/env python3
"""
    module
"""

def copy_matrix(matrix):
    out = []
    for val in matrix:
        out.append(val[:])
    return out

def determinant_fast(A):
    # Section 1: Establish n parameter and copy A
    n = len(A)
    AM = copy_matrix(A)
 
    # Section 2: Row ops on A to get in upper triangle form
    for fd in range(n): # A) fd stands for focus diagonal
        for i in range(fd+1,n): # B) only use rows below fd row
            if AM[fd][fd] == 0: # C) if diagonal is zero ...
                AM[fd][fd] == 1.0e-18 # change to ~zero
            # D) cr stands for "current row"
            crScaler = AM[i][fd] / AM[fd][fd] 
            # E) cr - crScaler * fdRow, one element at a time
            for j in range(n): 
                AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
    # Section 3: Once AM is in upper triangle form ...
    product = 1.0
    for i in range(n):
        # ... product of diagonals is determinant
        product *= AM[i][i] 
 
    return product


def determinant_recursive(A, total=0):
    indices = list(range(len(A)))
    if len(A) == 2 and len(A[0]) == 2:
        val = A[0][0] * A[1][1] - A[1][0] * A[0][1]
        return val
    for fc in indices: # A) for each focus column, ...
        As = copy_matrix(A) # B) make a copy, and ...
        As = As[1:] # ... C) remove the first row
        height = len(As) # D) 
 
        for i in range(height): 
            As[i] = As[i][0:fc] + As[i][fc+1:] 
 
        sign = (-1) ** (fc % 2) # F) 
        sub_det = determinant_recursive(As)
        total += sign * A[0][fc] * sub_det 
 
    return total


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
    return int(res) if int(res) == res else res
