#!/usr/bin/env python3
"""
    module
"""


def normalization_constants(X):
    """ normalization """
    return X.mean(axis=0), X.std(axis=0)
