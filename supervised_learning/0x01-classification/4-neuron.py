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
        self.__W = np.random.normal(0.0, 1.0, (1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W
    
    @property
    def b(self):
        return self.__b
    
    @property
    def A(self):
        return self.__A
    def act_func(self, X):
    
        return 1/(1 + np.exp(-X))
    
    def forward_prop(self, X):
        """ forward prop """
        self.__A = self.act_func(np.dot(self.__W, X) + self.__b)
        return self.__A
    
    def cost(self, Y, A):
        """ cost """
        return -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)).mean()
    
    def evaluate(self, X, Y):
        """ evaluate """
        A = self.forward_prop(X)
        err = self.cost(Y, A)
        return np.round(A).astype(int), err