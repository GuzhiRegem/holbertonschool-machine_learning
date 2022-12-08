#!/usr/bin/env python3
"""
    module
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """ autoencoder """
    K, L = keras, keras.layers
    inputs = {
        "encod": K.Input(shape=(input_dims)),
        "decod": K.Input(shape=(latent_dims))
    }

    out = inputs["encod"]
    for val in filters:
        out = L.Conv2D(
            val,
            activation='relu',
            kernel_size=(3, 3),
            padding='same'
        )(out)
        out = L.MaxPooling2D((2, 2), padding='same')(out)
    encoder_outputs = out
    encoder = K.Model(inputs=inputs["encod"], outputs=encoder_outputs)

    out = inputs["decod"]
    for val in reversed(filters[1:]):
        out = L.Conv2D(
            val,
            activation='relu',
            kernel_size=(3, 3),
            padding='same'
        )(out)
        out = L.UpSampling2D((2, 2))(out)
    out = L.Conv2D(
        filters[0],
        kernel_size=(3, 3),
        padding='valid',
        activation='relu'
    )(out)
    out = L.UpSampling2D((2, 2))(out)
    out = L.Conv2D(
        input_dims[2],
        activation='sigmoid',
        kernel_size=(3, 3),
        padding='same'
    )(out)
    decoder = K.Model(inputs=inputs["decod"], outputs=out)

    auto = K.Model(
        inputs=inputs["encod"],
        outputs=decoder(encoder(inputs["encod"]))
    )
    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
