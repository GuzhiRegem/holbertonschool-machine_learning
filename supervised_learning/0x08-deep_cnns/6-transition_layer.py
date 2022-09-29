#!/usr/bin/env python3
"""
    module
"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """ transition_layer """
    init = K.initializers.he_normal()
    act = {"kernel_initializer": init, "padding": "same"}
    nb_filters = int(nb_filters * compression)
    L = K.layers
    out = L.BatchNormalization(axis=3)(X)
    out = L.Activation(K.activations.relu)(out)
    out = L.Conv2D(nb_filters, 1, **act)(out)
    out = L.AveragePooling2D(2, 2, padding='valid')(out)
    return out, nb_filters
