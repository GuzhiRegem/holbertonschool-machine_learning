#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ convolve """
    m = images.shape[0]
    pad = [kernel.shape[0] // 2, kernel.shape[1] // 2]
    out = np.zeros(shape=(m, images.shape[1] - pad[0] * 2, images.shape[2] - pad[1] * 2))
    for x in range(out.shape[1]):
        for y in range(out.shape[2]):
            sp = images[:, x: x + 2 + pad[0], y: y + 2 + pad[1]]
            out[:, x, y] = np.sum(np.sum(sp * kernel, axis=1), axis=1)
    return out
