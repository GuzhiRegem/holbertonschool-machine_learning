#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ np cat """
    return np.concatenate((mat1, mat2), axis=axis)
