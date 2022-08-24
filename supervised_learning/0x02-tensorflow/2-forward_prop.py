#!/usr/bin/env python3
"""
    module
"""
import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """ function """
    prev = x
    for lay in range(len(layer_sizes)):
        new = create_layer(prev, layer_sizes[lay], activations[lay])
        prev = new
    return new
