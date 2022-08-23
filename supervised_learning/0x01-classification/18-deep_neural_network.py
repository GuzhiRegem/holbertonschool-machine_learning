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
        for val in layers:
            if type(val) != int or val <= 0:
                raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for lay in range(self.L):
            self.weights[f"b{lay + 1}"] = np.zeros((layers[lay], 1))
            prev = layers[lay - 1] if (lay > 0) else nx
            self.weights[f"W{lay + 1}"] = np.random.randn(layers[lay], prev)
            self.weights[f"W{lay + 1}"] *= np.sqrt(2/prev)

    @property
    def L(self):
        """ L getter """
        return self.__L

    @property
    def cache(self):
        """ cache getter """
        return self.__cache

    @property
    def weights(self):
        """ weights getter """
        return self.__weights

    def act_func(self, X):
        """ act """
        return 1/(1 + np.exp(-X))

    def forward_prop(self, X):
        """ forward prop """
        self.__cache["A0"] = X
        for lay in range(1, self.L + 1):
            prev = self.cache[f"A{lay - 1}"]
            Z = np.dot(self.weights[f"W{lay}"], prev) + self.weights[f"b{lay}"]
            self.__cache[f"A{lay}"] = self.act_func(Z)
        return self.cache[f"A{lay}"], self.cache