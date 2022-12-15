#!/usr/bin/env python3
"""
    module
"""
import numpy as np


class GRUCell:
    """ GRU """
    def __init__(self, i, h, o):
        """ init """
        self.Wz = np.random.normal(size=(i+h, h))
        self.Wr = np.random.normal(size=(i+h, h))
        self.Wh = np.random.normal(size=(i+h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ forward """
        concat = np.concatenate((h_prev, x_t), axis=1)
        z = 1 / (1 + np.exp(-(np.dot(concat, self.Wz) + self.bz)))
        r = 1 / (1 + np.exp(-(np.dot(concat, self.Wr) + self.br)))
        concat = np.concatenate((r * h_prev, x_t), axis=1)
        h_hat = np.tanh(np.dot(concat, self.Wh) + self.bh)
        h_next = (1 - z) * h_prev + z * h_hat
        div = np.sum(np.exp(np.dot(h_next, self.Wy) + self.by),
                     axis=1, keepdims=True)
        y = np.exp(np.dot(h_next, self.Wy) + self.by) / div
        return h_next, y
