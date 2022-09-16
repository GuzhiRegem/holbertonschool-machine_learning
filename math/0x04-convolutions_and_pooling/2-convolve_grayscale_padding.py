#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """ convolve """
    ks = np.array(kernel.shape)
    ph, pw = padding
    img = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                 'constant', constant_values=0)
    sh = np.array(images.shape[1:]) + (np.array(padding) * 2) - ks + 1
    out = np.zeros(shape=(images.shape[0], sh[0], sh[1]))
    for x in range(out.shape[1]):
        for y in range(out.shape[2]):
            sp = img[:, x: x + ks[0], y: y + ks[1]]
            out[:, x, y] = np.sum(sp * kernel, axis=(1, 2))
    return out
