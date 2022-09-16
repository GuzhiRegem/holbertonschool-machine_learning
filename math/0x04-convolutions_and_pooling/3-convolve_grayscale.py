#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """ convolve """
    ks = np.array(kernel.shape)
    if type(padding) == tuple:
        ph, pw = padding
    if padding == "valid":
        ph, pw = (0, 0)
    if padding == "same":
        ph, pw = tuple(np.ceil((ks - 1) / 2).astype(int))
    img = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                 'constant', constant_values=0)
    sh = np.array(images.shape[1:]) + (np.array((ph, pw)) * 2) - ks + 1
    sh = (sh // np.array(stride)).astype(int)
    out = np.zeros(shape=(images.shape[0], sh[0], sh[1]))
    for x in range(out.shape[1]):
        for y in range(out.shape[2]):
            _x, _y = (x * 2, y * 2)
            sp = img[:, _x: _x + ks[0], _y: _y + ks[1]]
            out[:, x, y] = np.sum(sp * kernel, axis=(1, 2))
    return out
