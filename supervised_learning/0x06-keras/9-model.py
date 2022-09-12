#!/usr/bin/env python3
"""
    module
"""
import tensorflow.keras as K


def save_model(network, filename):
    """ save model """
    network.save(filename)


def load_model(filename):
    """ load model """
    return K.models.load_model(filename)
