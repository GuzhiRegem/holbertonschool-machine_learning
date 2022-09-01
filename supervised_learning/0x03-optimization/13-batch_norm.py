#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """ function """
    mean = Z.mean(axis=0)
    var = Z.var(axis=0)
    norm = (Z - mean) / np.sqrt(var + epsilon)
    return (gamma * norm) + beta
