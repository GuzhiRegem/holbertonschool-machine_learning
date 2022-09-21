#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """ conv forward """
    ks = np.array(W.shape[:2])
    nc = W.shape[3]
    siz = np.array(A_prev.shape[1:3])
    st = np.array(stride)

    if padding == "valid":
        ph, pw = (0, 0)
    elif padding == "same":
        ph, pw = ((((siz - 1) * st) + ks[:2] - siz) // 2) + 1
    else:
        ph, pw = padding
    img = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                 'constant', constant_values=0)
    pad = np.array((ph, pw))
    sh = ((siz + (2 * pad) - ks[:2]) // st) + 1
    out = np.zeros(shape=(A_prev.shape[0], sh[0], sh[1], nc))
    ite = (siz + (2 * pad) - ks[:2] + 1)
    for k_idx in range(nc):
        kernel = W[:, :, :, k_idx]
        for i, x in enumerate(list(range(0, ite[0], st[0]))):
            for j, y in enumerate(list(range(0, ite[1], st[1]))):
                sp = img[:, x: x + ks[0], y: y + ks[1], :]
                out[:, i, j, k_idx] = np.sum(sp * kernel, axis=(1, 2, 3))
    return activation(out + b)
