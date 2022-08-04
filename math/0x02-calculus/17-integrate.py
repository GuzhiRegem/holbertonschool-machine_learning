#!/usr/bin/env python3
"""
    module
"""


def poly_integral(poly, C=0):
    """ poly integral """
    if type(C) != int or type(poly) != list or poly == 0:
        return None
    out = [C]
    if poly == [0]:
        return out
    try:
        for i in range(len(poly)):
            v = poly[i] / (i + 1)
            out.append(int(v) if int(v) == v else v)
    except Exception:
        return None
    return out
