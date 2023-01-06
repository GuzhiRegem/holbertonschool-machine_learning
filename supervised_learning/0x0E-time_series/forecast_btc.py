#!/usr/bin/env python3
"""
    module
"""
import pandas as pd
import tensorflow.keras as K
from preprocess_data import preprocess_data
import matplotlib.pyplot as plt


def create_model():
    L = K.layers
    inp = K.Input((1440, 5))
    out = L.LSTM(120, input_shape=(1, 5), activation='relu', dropout=0.4)(inp)
    out = L.BatchNormalization()(out)
    out = L.Dense(60, activation='relu')(out)
    out = L.Dropout(0.5)(out)
    out = L.BatchNormalization()(out)
    out = L.Dense(30, activation='relu')(out)
    out = L.Dropout(0.5)(out)
    out = L.BatchNormalization()(out)
    out = L.Dense(5, activation='softmax')(out)
    out = L.BatchNormalization()(out)
    model = K.models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == '__main__':
    coinbase_df = pd.read_csv('coinbase.csv')
    bitstamp_df = pd.read_csv('bitstamp.csv')

    # Preprocess the data
    train_x, train_y = preprocess_data(coinbase_df)
    validate_x, validate_y = preprocess_data(bitstamp_df)


    # Create the model
    model = create_model()
    model.summary()
    print("examples TRAIN:", train_x.shape[0])

    # Train the model
    hist = model.fit(
        train_x,
        train_y,
        batch_size=64,
        epochs=20,
        verbose=True,
        validation_data=(train_x, train_y),
        workers=2,
        callbacks=[K.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=50)]
    )

    plt.plot(hist)
    plt.show()
    val_loss = model.evaluate(train_x, train_y)
    print('Validation loss: {:.4f}'.format(val_loss))
