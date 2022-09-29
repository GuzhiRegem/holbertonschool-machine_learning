#!/usr/bin/env python3
"""
    module
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """ inception_block """
    init = K.initializers.he_normal()
    act = {
            "activation": K.activations.relu,
            "kernel_initializer":
            init, "padding": "same"
    }
    L1 = K.layers.Conv2D(filters[0], 1, **act)(A_prev)
    L2 = K.layers.Conv2D(filters[1], 1, **act)(A_prev)
    L3 = K.layers.Conv2D(filters[2], 3, **act)(L2)
    L4 = K.layers.Conv2D(filters[3], 1, **act)(A_prev)
    L5 = K.layers.Conv2D(filters[4], 5, **act)(L4)
    L6 = K.layers.MaxPooling2D(pool_size=3, strides=1, padding="same")(A_prev)
    L7 = K.layers.Conv2D(filters[5], 1, **act)(L6)
    return K.layers.concatenate([L1, L3, L5, L7])
