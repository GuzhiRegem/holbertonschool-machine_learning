#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """ convolve """
    ks = np.array(kernel.shape)
    pad = np.ceil((ks - 1) / 2).astype(int)
    img = np.pad(images, ((0, 0), (padding[0], padding[0]),
                 (padding[1], padding[1])),
                 'constant', constant_values=0)
    sh = np.array(img.shape) - np.pad(pad * 2, (1, 0))
    out = np.zeros(shape=(sh))
    for x in range(out.shape[1]):
        for y in range(out.shape[2]):
            sp = img[:, x: x + ks[0], y: y + ks[1]]
            out[:, x, y] = np.sum(sp * kernel, axis=(1, 2))
    return out
