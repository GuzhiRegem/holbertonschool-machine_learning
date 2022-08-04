#!/usr/bin/env python3
"""
    module
"""


def summation_i_squared(n):
    """ sum """
    if type(n) != int:
        return None
    if n < 1:
        return None
    if n == 1:
        return 1
    return (n * n) + summation_i_squared(n - 1)
