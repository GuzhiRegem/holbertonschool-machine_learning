#!/usr/bin/env python3
"""
    module
"""
import numpy as np


class DeepNeuralNetwork:
    """ Deep Neural Network """
    def __init__(self, nx, layers):
        """ init function """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) != list or layers == []:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for lay in range(self.__L):
            if type(layers[lay]) != int or layers[lay] < 0:
                raise TypeError("layers must be a list of positive integers")
            s_l = str(lay + 1)
            self.__weights["b" + s_l] = np.zeros((layers[lay], 1))
            prev = nx
            if (lay > 0):
                prev = layers[lay - 1]
            self.__weights["W" + s_l] = np.random.randn(layers[lay], prev)
            self.__weights["W" + s_l] *= np.sqrt(2/prev)

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
            prev = self.cache["A" + str(lay - 1)]
            Z = np.dot(self.weights["W" + str(lay)], prev)
            Z += self.weights["b" + str(lay)]
            self.__cache["A" + str(lay)] = self.act_func(Z)
        return self.cache["A" + str(lay)], self.cache

    def cost(self, Y, A):
        """ cost """
        return -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)).mean()

    def evaluate(self, X, Y):
        """ eval """
        A, _ = self.forward_prop(X)
        err = self.cost(Y, A)
        return np.round(A).astype(int), err

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ gradient """
        m = Y.shape[1]
        dZ = cache["A" + str(self.L)] - Y
        for lay in range(self.L, 0, -1):
            A = self.cache["A" + str(lay - 1)]
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dW = np.dot(dZ, A.T) / m
            dZ = np.dot(self.weights["W" + str(lay)].T, dZ) * (A * (1 - A))
            self.__weights["W" + str(lay)] -= dW * alpha
            self.__weights["b" + str(lay)] -= db * alpha
