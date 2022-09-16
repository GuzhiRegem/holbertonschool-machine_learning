#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """ convolve """
    ks = np.array(kernel_shape)
    siz = np.array(images.shape[1:3])
    st = np.array(stride)
    sh = ((siz - ks) // st) + 1
    out = np.zeros((images.shape[0], sh[0], sh[1], images.shape[3]))
    ite = (siz - ks + 1)
    for i, x in enumerate(list(range(0, ite[0], st[0]))):
        for j, y in enumerate(list(range(0, ite[1], st[1]))):
            sp = images[:, x: x + ks[0], y: y + ks[1], :]
            if mode == "max":
                res = np.max(sp, axis=(1, 2))
            if mode == "avg":
                res = np.average(sp, axis=(1, 2))
            out[:, i, j, :] = res
    return out
