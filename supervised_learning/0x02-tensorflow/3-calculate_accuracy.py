#!/usr/bin/env python3
"""
    module
"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """ function """
    max_pred = tf.argmax(y_pred, 1)
    equal = tf.equal(tf.argmax(y, 1), max_pred)
    return tf.reduce_mean(tf.cast(equal, tf.float32))
