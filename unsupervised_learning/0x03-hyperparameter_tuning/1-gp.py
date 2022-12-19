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
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """ kernel """
        if type(X1) is not np.ndarray or len(X1.shape) != 2:
            raise TypeError("X1 must be numpy.ndarray of shape (m, 1)")
        m, one = X1.shape
        if one != 1:
            raise TypeError("X1 must be numpy.ndarray of shape (m, 1)")
        if type(X2) is not np.ndarray or len(X2.shape) != 2:
            raise TypeError("X2 must be numpy.ndarray of shape (n, 1)")
        n, one = X2.shape
        if one != 1:
            raise TypeError("X2 must be numpy.ndarray of shape (n, 1)")
        X1_sum = np.sum(X1 ** 2, 1).reshape(-1, 1)
        X2_sum = np.sum(X2 ** 2, 1)
        dis = X1_sum + X2_sum - 2 * np.dot(X1, X2.T)
        cov = (self.sigma_f ** 2) * np.exp(-0.5 / (self.l ** 2) * dis)
        return cov

    def predict(self, X_s):
        """Predict mean and standard deviation of points in Gaussian Process"""
        K_s = self.kernel(X_s, self.X)
        K_i = np.linalg.inv(self.K)
        mu = np.matmul(np.matmul(K_s, K_i), self.Y)[:, 0]
        K_s2 = self.kernel(X_s, X_s)
        sigma = K_s2 - np.matmul(np.matmul(K_s, K_i), K_s.T)
        return mu, np.diagonal(sigma)
