#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ conv forward """
    ks = np.array(kernel_shape)
    siz = np.array(A_prev.shape[1:3])
    st = np.array(stride)
    sh = ((siz - ks) // st) + 1
    out = np.zeros((A_prev.shape[0], sh[0], sh[1], A_prev.shape[3]))
    ite = (siz - ks + 1)
    for i, x in enumerate(list(range(0, ite[0], st[0]))):
        for j, y in enumerate(list(range(0, ite[1], st[1]))):
            sp = A_prev[:, x: x + ks[0], y: y + ks[1], :]
            if mode == "max":
                res = np.max(sp, axis=(1, 2))
            if mode == "avg":
                res = np.average(sp, axis=(1, 2))
            out[:, i, j, :] = res
    return out
