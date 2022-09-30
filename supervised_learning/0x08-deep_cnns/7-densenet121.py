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
    X, nb_filters_ = K.Input(shape=(224, 224, 3)), 64
    out = L.Activation('relu')(L.BatchNormalization()(X))
    out = L.Conv2D(64, 7, strides=2, **act)(out)
    out_ = L.MaxPooling2D(pool_size=3, strides=2, padding="same")(out)
    for i in [6, 12, 24, 16]:
        out, nb_filters = dense_block(out_, nb_filters_, growth_rate, i)
        out_, nb_filters_ = transition_layer(out, nb_filters, compression)
    out = L.AveragePooling2D(7, 1, padding='valid')(out)
    out = L.Dense(1000, activation='softmax', kernel_initializer=init)(out)
    return K.models.Model(inputs=X, outputs=out)
