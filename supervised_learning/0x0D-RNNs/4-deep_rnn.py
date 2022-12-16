#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """ deep rnn """
    t, m, i = X.shape
    l, _, h = h_0.shape

    H = np.zeros((t+1, l, m, h))
    H[0] = h_0

    Y = []
    for t in range(t):
        x = X[t]
        for l, rnn_cell in enumerate(rnn_cells):
            h_prev = H[t, l]
            h_next, y = rnn_cell.forward(x, h_prev)
            H[t+1, l] = h_next
            x = h_next
        Y.append(y)

    return H, np.array(Y)
