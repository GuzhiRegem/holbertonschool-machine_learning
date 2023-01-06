#!/usr/bin/env python3
"""
    module
"""
import numpy as np
min_in_d = 24 * 60

def split_array(df):
    df = df[~np.isnan(df).any(axis=1), :]
    df = df[:(df.shape[0] // min_in_d) * min_in_d]
    x = np.stack(np.array_split(df, df.shape[0] / min_in_d))[:-1]
    y = df[((np.arange(x.shape[0]) * min_in_d) + (min_in_d + 60))]
    return x, y

def preprocess_data(df):
    df = df[['Open', 'High', 'Low', 'Close', 'Volume_(BTC)']]

    df = (df-df.mean())/df.std()
    df = df / df.max()

    df = df.to_numpy()
    x, y = split_array(df)
    """
    for i in range(1, min_in_d, 120):
        x_, y_ = split_array(np.roll(df, (-i, 0))[:-min_in_d])
        x = np.concatenate((x, x_), axis=0)
        y = np.concatenate((y, y_), axis=0)
        print(i)
    """

    return x, y

