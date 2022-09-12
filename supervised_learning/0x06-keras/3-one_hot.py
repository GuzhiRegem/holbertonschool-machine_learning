#!/usr/bin/env python3
"""
    module
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """ function """
    return K.utils.to_categorical(labels, num_classes=classes)
