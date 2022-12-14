#!/usr/bin/env python3
"""
    module
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """ autoencoder """
    K, L = keras, keras.layers
    inputs = {
        "encod": K.Input(shape=(input_dims,)),
        "decod": K.Input(shape=(latent_dims,))
    }

    out = inputs["encod"]
    for val in hidden_layers:
        out = L.Dense(val, activation="relu")(out)
    L1 = K.regularizers.l1(lambtha)
    out = L.Dense(latent_dims, activation="relu", activity_regularizer=L1)(out)
    encoder = K.Model(inputs=inputs["encod"], outputs=out)

    out = inputs["decod"]
    for val in reversed(hidden_layers):
        out = L.Dense(val, activation="relu")(out)
    out = L.Dense(input_dims, activation="sigmoid")(out)
    decoder = K.Model(inputs=inputs["decod"], outputs=out)

    auto = K.Model(
        inputs=inputs["encod"],
        outputs=decoder(encoder(inputs["encod"]))
    )
    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
