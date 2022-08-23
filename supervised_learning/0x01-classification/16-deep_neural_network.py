#!/usr/bin/env python3
"""
    module
"""
import numpy as np


class DeepNeuralNetwork:
    """ Deep Neural Network """
    def __init__(self, nx, layers):
        """ init """
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
            if type(layers[lay]) != int or layers[lay] <= 0:
                raise TypeError("layers must be a list of positive integers")
            self.weights[f"b{lay + 1}"] = np.zeros((layers[lay], 1))
            prev = layers[lay - 1] if (lay > 0) else nx
            self.weights[f"W{lay + 1}"] = np.random.randn(layers[lay], prev)
            self.weights[f"W{lay + 1}"] *= np.sqrt(2/prev)
