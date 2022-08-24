#!/usr/bin/env python3
"""
    module
"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """ create train op """
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
