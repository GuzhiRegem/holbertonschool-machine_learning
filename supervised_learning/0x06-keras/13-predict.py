#!/usr/bin/env python3
"""
    module
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """ function """
    return network.predict(data, verbose=verbose)