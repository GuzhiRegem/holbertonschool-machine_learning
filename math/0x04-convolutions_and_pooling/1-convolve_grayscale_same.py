#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    img = np.pad(images, ((0, 0), (1, 1), (1, 1)))
    pad = kernel.shape
    out = np.zeros(shape=images.shape)
    for x in range(out.shape[1]):
        for y in range(out.shape[2]):
            sp = img[:, x: x + pad[0], y: y + pad[1]]
            out[:, x, y] = np.sum(sp * kernel, axis=(1, 2))
    return out