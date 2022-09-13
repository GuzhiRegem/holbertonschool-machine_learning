#!/usr/bin/env python3
"""
    module
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """ function """
    calls = []
    if (validation_data is not None):
        if early_stopping:
            calls.append(K.callbacks.EarlyStopping(patience=patience))
        if learning_rate_decay:
            def f(step):
                return alpha / (1 + decay_rate * step)
            calls.append(K.callbacks.LearningRateScheduler(f, verbose=1))
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
