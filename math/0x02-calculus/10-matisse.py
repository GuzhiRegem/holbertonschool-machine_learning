#!/usr/bin/env python3
"""
    module
"""


def poly_derivative(poly):
    """ derivative """
    if poly == [] or type(poly) != list:
        return None
    out = []
    for i in range(1, len(poly + 1)):
        out.append(i * poly[i])
    return out
