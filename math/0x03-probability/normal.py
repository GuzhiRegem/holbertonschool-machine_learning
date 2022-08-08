#!/usr/bin/env python3
"""
    module
"""
E = 2.7182818285
PI = 3.1415926536
def err(x):
    """ error func """
    err_vals = [[3, -3], [5, 10], [7, -42], [9, 216]]
    s = x
    for i in err_vals:
        s += (x ** i[0]) / i[1]
    return (2 * s) / (PI ** 0.5)


class Normal:
    """ normal class """
    def __init__(self, data=None, mean=0., stddev=1.):
        """ init """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
            return
        if type(data) != list:
            raise TypeError("data must be a list")
        if len(data) < 2:
            raise ValueError("data must contain multiple values")
        self.mean = float(sum(data) / len(data))
        deviation_data = []
        for value in data:
            deviation_data.append((value - self.mean) ** 2)
        self.stddev = (sum(deviation_data) / (len(deviation_data))) ** 0.5

    def z_score(self, x):
        """ z score """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """ x score """
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """ pdf """
        m = self.mean
        sd = self.stddev
        exp = -0.5 * (self.z_score(x) ** 2)
        mul = 1 / (sd * ((2 * PI) ** 0.5))
        return mul * (E ** exp)

    def cdf(self, x):
        """ cdf """
        return 0.5 * (1 + err((x - self.mean) / (self.stddev * (2 ** 0.5))))
