#!/usr/bin/env python3
"""
    module
"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    X_p = K.applications.densenet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return (X_p, Y_p)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(x_train, y_train)
    X_test, Y_test = preprocess_data(x_test, y_test)

    source_model = K.applications.DenseNet121(
        include_top=True,
        weights="imagenet"
    )
    source_model.trainable = False
    act = K.activations.relu

    inp = K.Input(shape=(32, 32, 3))
    out = K.layers.Resizing(224, 224)(inp)
    out = source_model(out, training=False)
    out = K.layers.Flatten()(out)
    out = K.layers.Dense(500, activation=act)(out)
    out = K.layers.Dropout(0.2)(out)
    out = K.layers.Dense(10, activation='softmax')(out)

    model = K.Model(inputs=inp, outputs=out)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=K.optimizers.Adam(),
        metrics=["accuracy"]
    )
    model.fit(
        x=X_train,
        y=Y_train,
        validation_data=(X_test, Y_test),
        batch_size=300,
        epochs=5,
        verbose=True
    )
    model.save('cifar10-2.h5')
