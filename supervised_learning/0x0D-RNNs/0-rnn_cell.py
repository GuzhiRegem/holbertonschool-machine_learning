#!/usr/bin/env python3
"""
    module
"""
import numpy as np


class RNNCell:
    """ RNNCell """
    def __init__(self, i, h, o):
        """ init """
        self.Wh = np.random.randn(h+i, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ forward """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(concat, self.Wh) + self.bh)
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)
        return h_next, y

    def softmax(self, x):
        """ softmax """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
