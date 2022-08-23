#!/usr/bin/env python3
"""
    module
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Deep Neural Network
    """

    def __init__(self, nx, layers):
        """ init function """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) != list:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for lay in range(self.L):
            if type(layers[lay]) != int or layers[lay] < 0:
                raise TypeError("layers must be a list of positive integers")
            s_l = str(lay + 1)
            self.weights["b" + s_l] = np.zeros((layers[lay], 1))
            prev = nx
            if (lay > 0):
                prev = layers[lay - 1]
            self.weights["W" + s_l] = np.random.randn(layers[lay], prev)
            self.weights["W" + s_l] *= np.sqrt(2/prev)
