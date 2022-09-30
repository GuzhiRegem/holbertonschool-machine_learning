#!/usr/bin/env python3
"""
    module
"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """ inception_network """
    init, L = K.initializers.he_normal(), K.layers
    act = {"kernel_initializer": init, "padding": "same"}
    X, nb_filters = K.Input(shape=(224, 224, 3)), growth_rate * 2
    out = L.Activation('relu')(L.BatchNormalization()(X))
    out = L.Conv2D(64, 7, strides=2, **act)(out)
    out = L.MaxPooling2D(pool_size=3, strides=2, padding="same")(out)
    for i in [6, 12, 24]:
        out, nb_filters = dense_block(out, nb_filters, growth_rate, i)
        out, nb_filters = transition_layer(out, nb_filters, compression)
    out, nb_filters = dense_block(out, nb_filters, growth_rate, 16)
    out = L.AveragePooling2D(7, 1, padding='valid')(out)
    out = L.Dense(1000, activation='softmax', kernel_initializer=init)(out)
    return K.models.Model(inputs=X, outputs=out)
