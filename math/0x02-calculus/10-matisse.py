#!/usr/bin/env python3
"""
    module
"""


def poly_derivative(poly):
    """ derivative """
    if poly == []:
        return None
    out = []
    try:
        for i in range(1, len(poly)):
            out.append(i * poly[i])
    except Exception:
        return None
    return out
