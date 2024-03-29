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
    ite = (siz + (2 * ph) - ks + 1)
    for i, x in enumerate(list(range(0, ite[0], st[0]))):
        for j, y in enumerate(list(range(0, ite[1], st[1]))):
            sp = img[:, x: x + ks[0], y: y + ks[1]]
            out[:, i, j] = np.sum(sp * kernel, axis=(1, 2))
    return out
