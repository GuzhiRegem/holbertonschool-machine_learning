#!/usr/bin/env python3
"""
    module
"""
import numpy as np


def gensim_to_keras(model):
    """ bag_of_words """
    return model.wv.get_keras_embedding(train_embeddings=True)
