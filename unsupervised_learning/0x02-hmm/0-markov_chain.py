#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def markov_chain(P, s, t=1):
    """ markov_chain """
    out = s
    for i in range(t):
        out = np.dot(out, P)
    return out
