#!/usr/bin/env python3
"""
    module
"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """ inception_network """
    init = K.initializers.he_normal()
    act = {"kernel_initializer": init, "padding": "same"}
    L = K.layers

    def block(lis, prev, amount, s=2):
        o = projection_block(prev, lis, s)
        for i in range(amount - 1):
            o = identity_block(o, lis)
        return o
    X = K.Input(shape=(224, 224, 3))
    out = L.Conv2D(64, 7, strides=2, **act)(X)
    out = L.BatchNormalization(axis=3)(out)
    out = L.Activation(K.activations.relu)(out)
    out = L.MaxPooling2D(pool_size=3, strides=2, padding="same")(out)
    out = block([64, 64, 256], out, 3, 1)
    out = block([128, 128, 512], out, 4)
    out = block([256, 256, 1024], out, 6)
    out = block([512, 512, 2048], out, 3)
    out = L.AveragePooling2D(7, 1, padding='valid')(out)
    out = L.Dense(1000, activation='softmax', kernel_initializer=init)(out)

    return K.models.Model(inputs=X, outputs=out)
