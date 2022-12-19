#!/usr/bin/env python3
"""
    module
"""
import numpy as np


class BidirectionalCell:
    """Bidirectional cell class"""
    def __init__(self, i, h, o):
        """ init """
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h * 2, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """forward"""
        concat = np.concatenate((h_prev, x_t), 1)
        return np.tanh(np.matmul(concat, self.Whf) + self.bhf)

    def backward(self, h_prev, x_t):
        """back"""
        concat = np.concatenate((h_prev, x_t), 1)
        return np.tanh(np.matmul(concat, self.Whb) + self.bhb)

    def output(self, H):
        """out"""
        out = np.exp(np.matmul(H, self.Wy) + self.by,)
        return out / out.sum(axis=2, keepdims=True)
