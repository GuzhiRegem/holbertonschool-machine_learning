#!/usr/bin/env python3
"""
    module
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """ function """
    calls = []
    if (validation_data is not None) and early_stopping:
        calls.append(K.callbacks.EarlyStopping(patience=patience))
    return network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=calls
    )
