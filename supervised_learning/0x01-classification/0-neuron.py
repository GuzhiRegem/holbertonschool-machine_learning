#!/usr/bin/env python3
"""
    module
"""
import numpy as np


class Neuron:
    """ Neuron """
    def __init__(self, nx):
        """ init """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.normal(0.0, 1.0, (1, nx))
        self.b = 0
        self.A = 0
