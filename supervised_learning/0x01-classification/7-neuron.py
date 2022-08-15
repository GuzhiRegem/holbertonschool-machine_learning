#!/usr/bin/env python3
"""
    module
"""
import numpy as np
import matplotlib.pyplot as plt


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
        """ W getter """
        return self.__W

    @property
    def b(self):
        """ b getter """
        return self.__b

    @property
    def A(self):
        """ A getter """
        return self.__A

    def act_func(self, X):
        """ act """
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

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ gradient descent """
        self.__b -= alpha * (np.sum(A - Y) / X.shape[1])
        self.__W -= alpha * (np.dot(X, (A - Y).T).T / X.shape[1])

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """ train """
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) != int:
                raise TypeError("step must be an integer")
            if (step <= 0) or (step > iterations):
                raise ValueError("step must be positive and <= iterations")
        g_vals = []
        for ite in range(iterations):
            if (ite % step) == 0:
                c = self.cost(Y, self.forward_prop(X))
                g_vals.append([c, ite])
                if verbose:
                    print(f"Cost after {ite} iterations: {c}")
            self.gradient_descent(X, Y, self.forward_prop(X), alpha)
        out, err = self.evaluate(X, Y)
        if verbose:
            print(f"Cost after {iterations} iterations: {err}")
        if graph:
            arr = np.array(g_vals)
            plt.plot(arr[:, 1], arr[:, 0])
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return out, err

