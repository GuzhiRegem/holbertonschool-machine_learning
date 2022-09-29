#!/usr/bin/env python3
"""
    module
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """ identity_block """
    init = K.initializers.he_normal()
    act = {
            "kernel_initializer": init,
            "padding": "same"
    }
    L = K.layers

    def ac(x):
        return L.Activation(K.activations.relu)(x)
    for lay in range(layers):
        out = ac(L.BatchNormalization(axis=3)(X))
        out = L.Conv2D(4 * growth_rate, 1, **act)(out)
        out = ac(L.BatchNormalization(axis=3)(out))
        out = L.Conv2D(growth_rate, 3, **act)(out)
        X = L.concatenate([X, out])
        nb_filters += growth_rate
    return X, nb_filters
