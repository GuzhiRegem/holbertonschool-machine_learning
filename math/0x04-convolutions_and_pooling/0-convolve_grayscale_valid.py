#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ convolve """
    m = images.shape[0]
    pad = kernel.shape
    siz = images.shape[1:]
    out = np.zeros(shape=(m, siz[0] - pad[0] + 1, siz[1] - pad[1] + 1))
    for x in range(out.shape[1]):
        for y in range(out.shape[2]):
            sp = images[:, x: x + pad[0], y: y + pad[1]]
            out[:, x, y] = np.sum(sp * kernel, axis=(1, 2))
    return out
