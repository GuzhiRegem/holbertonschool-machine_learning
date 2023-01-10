#!/usr/bin/env python3
"""
    module
"""
from tensorflow.keras.layers import Embedding


def gensim_to_keras(model):
    """ bag_of_words """
    keyed_vectors = model.wv
    weights = keyed_vectors.vectors
    index_to_key = keyed_vectors.index_to_key 
    return Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True,
    )
