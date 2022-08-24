#!/usr/bin/env python3
"""
    module
"""
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """ function """
    x = tf.placeholder("float32", name="x", shape=(None, nx))
    y = tf.placeholder("float32", name="y", shape=(None, classes))
    return x, y
