#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """ rnn """
    t, m, i = X.shape
    _, h = h_0.shape

    H = np.zeros((t+1, m, h))
    Y = np.zeros((t, m, rnn_cell.by.shape[1]))

    H[0] = h_0
    for i in range(t):
        H[i+1], Y[i] = rnn_cell.forward(H[i], X[i])

    return H, Y