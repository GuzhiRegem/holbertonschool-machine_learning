#!/usr/bin/env python3
"""
    module
"""


def poly_integral(poly, C=0):
    """ poly integral """
    if not type(C) in [int, float] or type(poly) != list:
        return None
    out = [C]
    for i in range(len(poly)):
        v = poly[i] / (i + 1)
        out.append(int(v) if int(v) == v else v)
    return out
