#!/usr/bin/env python3
"""
    module
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ function """
    reg = K.regularizers.L2(lambtha)
    first = K.Input(shape=(nx, ))
    out = first
    for idx, layer in enumerate(layers):
        out = K.layers.Dense(
            layer,
            activation=activations[idx],
            input_shape=(nx,),
            kernel_regularizer=reg
        )(out)
        if idx < (len(layers) - 1):
            out = K.layers.Dropout(1 - keep_prob)(out)
    return K.Model(first, out)
