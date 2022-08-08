#!/usr/bin/env python3
"""
    module
"""
E = 2.7182818285
PI = 3.1415926536

def fact(n):
    """ fact """
    out = 1
    for val in range(n):
        out *= val + 1
    return out

class Binomial:
    """ binomial class """
    def __init__(self, data=None, n=1, p=0.5):
        """ init """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
            return
        if type(data) != list:
            raise TypeError("data must be a list")
        if len(data) < 2:
            raise ValueError("data must contain multiple values")
        mean = float(sum(data) / len(data))
        var = float((sum(map(lambda n: (n - mean)**2, data)) / len(data)))
        self.n = round(mean / (-(var / mean) + 1))
        self.p = mean / self.n

    def pmf(self, k):
        """ calculates pmf of k """
        k = int(k)
        if k < 0:
            return 0
        mul = fact(self.n) / (fact(k)*fact(self.n-k))
        return mul*(self.p**k)*((1-self.p)**(self.n - k))

    def cdf(self, k):
        """ calculates cdf of k """
        k = int(k)
        if k < 0:
            return 0
        out = 0
        for i in range(k + 1):
            out += self.pmf(i)
        return out
