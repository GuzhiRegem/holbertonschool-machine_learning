#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """ conv forward """
    dA_prev = np.zeros(shape=A_prev.shape)
    dW = np.zeros(shape=W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    return dA_prev, dW, db
