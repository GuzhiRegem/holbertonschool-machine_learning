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
