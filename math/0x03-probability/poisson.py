#!/usr/bin/env python3
"""
    module
"""
E = 2.7182818285
PI = 3.1415926536


class Poisson:
    """ poisson class """
    def __init__(self, data=None, lambtha=1.):
        """ init """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
            return
        if type(data) != list:
            raise TypeError("data must be a list")
        if len(data) < 2:
            raise ValueError("data must contain multiple values")
        self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """ pmf """
        k = int(k)
        if k < 0:
            return 0
        fac = 1
        for i in range(1, k + 1):
            fac *= i
        lam = self.lambtha
        return ((lam ** k) * (E ** (-lam))) / fac

    def cdf(self, k):
        """ cdf """
        k = int(k)
        if k < 0:
            return 0
        val = 0
        for i in range(k + 1):
            fac = 1
            for j in range(1, i + 1):
                fac *= j
            val += (self.lambtha ** i) / fac
        return val * (E ** (-self.lambtha))
