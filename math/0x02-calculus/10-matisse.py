#!/usr/bin/env python3
"""
    module
"""


def poly_derivative(poly):
    """ derivative """
    try:
        out = []
        for idx, val in enumerate(poly):
            if idx == 0:
                continue
            if type(val) not in [float, int]:
                return None
            out.append(val * idx)
    except TypeError:
        return None
    except ValueError:
        return None
    if out == []:
        out = [0]
    return out
