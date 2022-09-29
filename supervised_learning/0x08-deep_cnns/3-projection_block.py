#!/usr/bin/env python3
"""
    module
"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """ identity_block """
    init = K.initializers.he_normal()
    act = {
            "kernel_initializer": init,
            "padding": "same"
    }
    L = K.layers

    def ac(x):
        return L.Activation(K.activations.relu)(x)
    out = L.Conv2D(filters[0], 1, strides=s,  **act)(A_prev)
    out = ac(L.BatchNormalization(axis=3)(out))
    out = L.Conv2D(filters[1], 3, **act)(out)
    out = ac(L.BatchNormalization(axis=3)(out))
    out = L.Conv2D(filters[2], 1, **act)(out)
    out = L.BatchNormalization(axis=3)(out)
    inp = L.Conv2D(filters[2], 1, strides=s, **act)(A_prev)
    inp = L.BatchNormalization(axis=3)(inp)
    return ac(L.Add()([out, inp]))
