#!/usr/bin/env python3
"""
    module
"""
import numpy as np


class GaussianProcess:
    """ GaussianProcess """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """ init """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = 0
    
    def kernel(self, X1, X2):
        """ kernel """
        dis = np.linalg.norm(X1 - X2, keepdims=True)
        return np.exp(dis / (2 * self.l))