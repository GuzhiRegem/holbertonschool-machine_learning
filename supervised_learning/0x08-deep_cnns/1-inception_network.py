#!/usr/bin/env python3
"""
    module
"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """ inception_network """
    init = K.initializers.he_normal()
    pad = {"padding": "same"}
    act = {
            "activation": K.activations.relu,
            "kernel_initializer": init,
            "padding": "same"
    }

    L = K.layers
    X = K.Input(shape=(224, 224, 3))
    out = L.Conv2D(64, 7, strides=2, **act)(X)
    out = L.MaxPooling2D(pool_size=3, strides=2, **pad)(out)
    out = L.Conv2D(192, 3, strides=1, **act)(out)
    out = L.MaxPooling2D(pool_size=3, strides=2, **pad)(out)
    out = inception_block(out, [64, 96, 128, 16, 32, 32])
    out = inception_block(out, [128, 128, 192, 32, 96, 64])
    out = L.MaxPooling2D(pool_size=3, strides=2, **pad)(out)
    out = inception_block(out, [192, 96, 208, 16, 48, 64])
    out = inception_block(out, [160, 112, 224, 24, 64, 64])
    out = inception_block(out, [128, 128, 256, 24, 64, 64])
    out = inception_block(out, [112, 144, 288, 32, 64, 64])
    out = inception_block(out, [256, 160, 320, 32, 128, 128])
    out = L.MaxPooling2D(pool_size=3, strides=2, **pad)(out)
    out = inception_block(out, [256, 160, 320, 32, 128, 128])
    out = inception_block(out, [384, 192, 384, 48, 128, 128])
    out = L.AveragePooling2D(pool_size=7, strides=1, padding='valid')(out)
    out = L.Dropout(rate=0.4)(out)
    out = L.Dense(1000, activation='softmax', kernel_initializer=init)(out)

    model = K.models.Model(inputs=X, outputs=out)
    return model