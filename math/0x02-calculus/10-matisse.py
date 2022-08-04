#!/usr/bin/env python3
"""
    module
"""


def poly_derivative(poly):
    """ derivative """
    out = []
    try:
        for i in range(1, len(poly)):
            out.append(i * poly[i])
    except Exception:
        return None
    if out == []:
        out = [0]
    return out
