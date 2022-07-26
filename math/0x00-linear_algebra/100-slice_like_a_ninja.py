#!/usr/bin/env python3
"""
    module
"""


def np_slice(matrix, axes={}):
    """ slice """
    ax = [slice(None)] * len(matrix.shape)
    for key, value in axes.items():
        ax[key] = slice(*value)
    return matrix[tuple(ax)]
