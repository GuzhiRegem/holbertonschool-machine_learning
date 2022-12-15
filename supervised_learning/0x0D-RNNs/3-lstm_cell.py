#!/usr/bin/env python3
"""
    module
"""
import numpy as np

def sigmoid(x):
    """ sigmoid """
    return 1 / (1 + np.exp(-x))


def tanh(x):
    """ sigmoid """
    return np.tanh(x)


def softmax(x):
    """ sigmoid """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


class LSTMCell:
    """ LSTMCell """
    def __init__(self, i, h, o):
        """ init """
        self.Wf = np.random.randn(h + i, h)
        self.Wu = np.random.randn(h + i, h)
        self.Wc = np.random.randn(h + i, h)
        self.Wo = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """ forward """
        concat = np.concatenate((h_prev, x_t), axis=1)

        forget_gate = self.Wf @ concat
        update_gate = self.Wu @ concat
        intermediate_state = self.Wc @ concat
        output_gate = self.Wo @ concat

        f = sigmoid(forget_gate + self.bf)
        u = sigmoid(update_gate + self.bu)
        c = tanh(intermediate_state + self.bc)
        c_next = f * c_prev + u * c
        o = sigmoid(output_gate + self.bo)
        h_next = o * tanh(c_next)

        y = softmax(self.Wy @ h_next + self.by)

        return h_next, c_next, y
