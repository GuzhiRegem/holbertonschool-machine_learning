#!/usr/bin/env python3
"""
    module
"""
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """ function """
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode=("fan_avg"))
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=init)(prev)
    return tf.layers.Dropout(keep_prob)(layer)
