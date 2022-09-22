#!/usr/bin/env python3
"""
    module
"""
import tensorflow.compat.v1 as tf

def lenet5(x, y):
    """ conv forward """
    init = tf.keras.initializers.VarianceScaling(scale=2.0)
    layers = [
        tf.layers.Conv2D(filters=6, kernel_size=5, padding='same'),
        tf.layers.MaxPooling2D(pool_size=2, strides=2),
        tf.layers.Conv2D(filters=16, kernel_size=5, padding='valid'),
        tf.layers.MaxPooling2D(pool_size=2, strides=2),
        tf.layers.Flatten(),
        tf.layers.Dense(120, activation=tf.nn.relu, kernel_initializer=init),
        tf.layers.Dense(84, activation=tf.nn.relu, kernel_initializer=init),
        tf.layers.Dense(10, kernel_initializer=init)
    ]
    out = x
    for lay in layers:
        out = lay(out)
    softmax = tf.nn.softmax(out)
    loss = tf.losses.softmax_cross_entropy(y, logits=out)
    op = tf.train.AdamOptimizer().minimize(loss)
    y_pred = tf.math.argmax(out, axis=1)
    y_out = tf.math.argmax(y, axis=1)
    equality = tf.math.equal(y_pred, y_out)
    accuracy = tf.reduce_mean(tf.cast(equality, 'float'))
    return softmax, op, loss, accuracy