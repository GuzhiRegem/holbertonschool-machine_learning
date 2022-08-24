#!/usr/bin/env python3
"""
    module
"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """ function """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    out = tf.keras.layers.Dense(
        n,
        activation=activation,
        kernel_initializer=init,
        name="layer"
    )
    return out(prev)
