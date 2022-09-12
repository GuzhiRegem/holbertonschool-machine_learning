#!/usr/bin/env python3
"""
    module
"""
import tensorflow.keras as K


def save_config(network, filename):
    """ save config """
    with open(filename, "w") as f:
        f.write(network.to_json())


def load_config(filename):
    """ load config """
    with open(filename, "r") as f:
        return K.models.model_from_json(f.read())
