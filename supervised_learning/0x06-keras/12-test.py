#!/usr/bin/env python3
"""
    module
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """ function """
    return network.evaluate(data, labels, verbose=verbose)