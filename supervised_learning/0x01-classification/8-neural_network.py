#!/usr/bin/env python3
"""
    module
"""
import numpy as np


class NeuralNetwork:
    """ Neural Network """
    def __init__(self, nx, nodes):
        """ init """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) != int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.W1 = np.random.normal(0.0, 1.0, (nodes, nx))
        self.W2 = np.random.normal(0.0, 1.0, (1, nodes))
        self.b1 = np.zeros((nodes, 1))
        self.b2 = 0
        self.A1 = 0
        self.A2 = 0
