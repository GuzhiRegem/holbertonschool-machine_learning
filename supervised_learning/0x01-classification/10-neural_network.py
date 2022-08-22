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
        self.__W1 = np.random.normal(0.0, 1.0, (nodes, nx))
        self.__W2 = np.random.normal(0.0, 1.0, (1, nodes))
        self.__b1 = np.zeros((nodes, 1))
        self.__b2 = 0
        self.__A1 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ W1 getter """
        return self.__W1

    @property
    def W2(self):
        """ W2 getter """
        return self.__W2

    @property
    def b1(self):
        """ b1 getter """
        return self.__b1

    @property
    def b2(self):
        """ b2 getter """
        return self.__b2

    @property
    def A1(self):
        """ A1 getter """
        return self.__A1

    @property
    def A2(self):
        """ A2 getter """
        return self.__A2

    def act_func(self, X):
        """ act """
        return 1/(1 + np.exp(-X))

    def forward_prop(self, X):
        """ forward_prop """
        self.__A1 = self.act_func(np.dot(self.W1, X) + self.b1)
        self.__A2 = self.act_func(np.dot(self.W2, self.A1) + self.b2)
        return self.__A1, self.__A2
