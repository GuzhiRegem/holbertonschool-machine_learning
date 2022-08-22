#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def one_hot_decode(one_hot):
    """ one hot decode """
    try:
        out = np.zeros(shape=(one_hot.shape[1],))
        inc = np.array(range(one_hot.shape[0]))
        val = (one_hot.T * inc).T
        for row in val:
            out += row
        return out.astype(np.int)
    except Exception as e:
        return None