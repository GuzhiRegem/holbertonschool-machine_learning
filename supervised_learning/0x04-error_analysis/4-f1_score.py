#!/usr/bin/env python3
"""
    module
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """ function """
    rec = sensitivity(confusion)
    pre = precision(confusion)
    return 2 * (rec * pre) / (rec + pre)
