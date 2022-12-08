#!/usr/bin/env python3
"""
    module
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """ autoencoder """
    K, L = keras, keras.layers
    inputs = {
        "encod": K.Input(shape=(input_dims,)),
        "decod": K.Input(shape=(latent_dims,))
    }
    conv_conf = {
        "activation": "relu",
        "kernel_size": (3, 3),
        "padding": "same"
    }

    out = inputs["encod"]
    for val in filters:
        out = L.Conv2D(val, **conv_conf)(out)
        out = L.MaxPooling2D((2, 2), padding="same")(out)
    encoder = K.Model(inputs=inputs["encod"], outputs=out)

    out = inputs["decod"]
    for val in reversed(filters):
        out = L.Conv2D(val, **conv_conf)(out)
        out = L.UpSampling2D((2, 2))(out)
    out = L.Conv2D(filters[0], **conv_conf, padding="same")(out)
    out = L.UpSampling2D((2, 2))(out)
    out = L.Conv2D(input_dims[2], **conv_conf, activation="sigmoid")(out)
    decoder = K.Model(inputs=inputs["decod"], outputs=out)

    auto = K.Model(
        inputs=inputs["encod"],
        outputs=decoder(encoder(inputs["encod"]))
    )
    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
