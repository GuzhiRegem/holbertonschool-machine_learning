#!/usr/bin/env python3
"""
    module
"""
import numpy as np


class MultiNormal:
    """ multinormal """
    def __init__(self, data):
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1, keepdims=True)
        self.cov = np.dot(data - self.mean, data.T - self.mean.T) / (n - 1)

    def pdf(self, x):
        """ pdf """
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if len(x.shape) != 2 or x.shape[0] != d or x.shape[1] != 1:
            raise ValueError("x must have the shape ({}, 1)".format(d))
        inv = np.linalg.inv(self.cov)
        mul = np.dot(np.dot((x - self.mean).T, inv), (x - self.mean))
        pdf = np.exp(-0.5 * mul)
        pdf /= np.sqrt(((2 * np.pi) ** d) * np.linalg.det(self.cov))
        return pdf[0][0]
