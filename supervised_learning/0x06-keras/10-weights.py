#!/usr/bin/env python3
"""
    module
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """ save weights """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """ load weights """
    network.load_weights(filename)
