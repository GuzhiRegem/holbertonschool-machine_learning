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
    Y = np.zeros((t, m, rnn_cells[-1].by.shape[1]))

    H[0] = h_0

    for layer in range(l):
        for step in range(t):
            if layer == 0:
                H[step+1][layer], _ = rnn_cells[layer].forward(H[step][layer], X[step])
            else:
                H[step+1][layer], _ = rnn_cells[layer].forward(H[step][layer], H[step][layer-1])
        Y[step], _ = rnn_cells[-1].forward(H[-1][-1], X[step])

    return H, Y
