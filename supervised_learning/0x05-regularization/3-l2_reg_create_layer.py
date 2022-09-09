#!/usr/bin/env python3
"""
    module
"""
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ function """
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode=("fan_avg"))
    reg = tf.keras.regularizers.L2(l2=lambtha)
    return tf.layers.Dense(
        n,
        activation=activation,
        kernel_initializer=init,
        kernel_regularizer=reg
    )(prev)
