#!/usr/bin/env python3
"""
    module
"""
import numpy as np

def sigmoid(x):
    """ sigmoid """
    return 1 / (1 + np.exp(-x))

def softmax(x):
    """ softmax """
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class LSTMCell:
    """ LSTMCell """
    def __init__(self, i, h, o):
        """ init """
        self.Wf = np.random.randn(h, i+h)
        self.Wu = np.random.randn(h, i+h)
        self.Wc = np.random.randn(h, i+h)
        self.Wo = np.random.randn(h, i+h)
        self.Wy = np.random.randn(o, h)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, c_prev, x_t):
        concat = np.concatenate((h_prev, x_t), axis=1)
        f = self.sigmoid(np.dot(self.Wf, concat) + self.bf)
        u = self.sigmoid(np.dot(self.Wu, concat) + self.bu)
        c_bar = np.tanh(np.dot(self.Wc, concat) + self.bc)
        c_next = f * c_prev + u * c_bar
        o = self.sigmoid(np.dot(self.Wo, concat) + self.bo)
        h_next = o * np.tanh(c_next)
        y = softmax(np.dot(self.Wy, h_next) + self.by)
        
        return h_next, c_next, y
