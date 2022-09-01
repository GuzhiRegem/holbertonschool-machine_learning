#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def normalization_constants(X):
    """ normalization """
    return np.mean(X, axis=0), np.std(X, axis=0)
