#!/usr/bin/env python3
"""
    module
"""


def poly_integral(poly, C=0):
    """ poly integral """
    if type(C) != int or type(poly) != list:
        return None
    out = []
    for i in range(len(poly)):
        out.append(poly[i] / (i + 1))
    return out
