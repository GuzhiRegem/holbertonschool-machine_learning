#!/usr/bin/env python3
"""
    module
"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """ conv forward """
    init = tf.keras.initializers.VarianceScaling(scale=2.0)
    act = {"activation": tf.nn.relu, "kernel_initializer": init}
    o = tf.layers.Conv2D(filters=6, kernel_size=5, padding='same', **act)(x)
    o = tf.layers.MaxPooling2D(pool_size=2, strides=2)(o)
    o = tf.layers.Conv2D(filters=16, kernel_size=5, padding='valid', **act)(o)
    o = tf.layers.MaxPooling2D(pool_size=2, strides=2)(o)
    o = tf.layers.Flatten()(o)
    o = tf.layers.Dense(120, **act)(o)
    o = tf.layers.Dense(84, **act)(o)
    o = tf.layers.Dense(10, kernel_initializer=init)(o)
    softmax = tf.nn.softmax(o)
    loss = tf.losses.softmax_cross_entropy(y, logits=o)
    op = tf.train.AdamOptimizer().minimize(loss)
    y_pred = tf.argmax(o, axis=1)
    y_out = tf.argmax(y, axis=1)
    equality = tf.equal(y_pred, y_out)
    accuracy = tf.reduce_mean(tf.cast(equality, 'float'))
    return softmax, op, loss, accuracy
