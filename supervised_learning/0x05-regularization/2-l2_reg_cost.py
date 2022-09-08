#!/usr/bin/env python3
"""
    module
"""
import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """ cost """
    return cost + tf.losses.get_regularization_losses()
