#!/usr/bin/env python3
"""
    module
"""
import tensorflow.compat.v1 as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """ train """
    m = X_train.shape[0]
    saver = tf.train.import_meta_graph(load_path + '.meta')
    with tf.Session() as sess:
        saver.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]
        for ep in range(epochs):
            x_shuff, y_shuff = shuffle_data(x, y)
            for idx in range(m / batch_size):
                b_size = batch_size
                if (idx + 1) * batch_size > m:
                    b_size = m - (idx * batch_size)
                print(idx * batch_size)
                
