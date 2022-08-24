#!/usr/bin/env python3
"""
    module
"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """ function """
    return tf.losses.softmax_cross_entropy(y, logits=y_pred)
