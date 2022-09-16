#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """ convolve """
    ks = np.array(kernel.shape)
    siz = np.array(images.shape[1:])
    st = np.array(stride)
    if padding == "valid":
        ph, pw = (0, 0)
    elif padding == "same":
        ph, pw = ((((siz - 1) * st) + ks - siz) // 2) + 1
    else:
        ph, pw = padding
    img = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                 'constant', constant_values=0)
    sh = ((siz + (2 * np.array((ph, pw))) - ks) // st) + 1
    out = np.zeros(shape=(images.shape[0], sh[0], sh[1]))
    for x in range(out.shape[1]):
        for y in range(out.shape[2]):
            _x, _y = (x * 2, y * 2)
            sp = img[:, _x: _x + ks[0], _y: _y + ks[1]]
            out[:, x, y] = np.sum(sp * kernel, axis=(1, 2))
    return out
