#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """ convolve """
    ks = np.array(kernel.shape)
    pad = np.ceil(ks - 1).astype(int)
    img = np.pad(images, ((0, 0), (pad[0], pad[0]), (pad[1], pad[1])),
                 'constant', constant_values=0)
    out = np.zeros(shape=images.shape)
    for x in range(out.shape[1]):
        for y in range(out.shape[2]):
            sp = img[:, x: x + ks[0], y: y + ks[1]]
            out[:, x, y] = np.sum(sp * kernel, axis=(1, 2))
    return out
