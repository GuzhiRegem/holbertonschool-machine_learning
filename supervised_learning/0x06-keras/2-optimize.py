#!/usr/bin/env python3
"""
    module
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """ function """
    network.compile(
        loss='categorical_crossentropy',
        optimizer=K.optimizers.Adam(
            alpha,
            beta1,
            beta2
        ),
        metrics=['accuracy']
    )
