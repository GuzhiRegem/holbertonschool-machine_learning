#!/usr/bin/env python3
"""
module
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
    decoder block
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """ init """
        if type(dm) is not int:
            raise TypeError(
                "dm must be int representing dimensionality of model")
        if type(h) is not int:
            raise TypeError(
                "h must be int representing number of heads")
        if type(hidden) is not int:
            raise TypeError(
                "hidden must be int representing number of hidden units")
        if type(drop_rate) is not float:
            raise TypeError(
                "drop_rate must be float representing dropout rate")
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """ call """
        attention_output1, _ = self.mha1(x, x, x, look_ahead_mask)
        attention_output1 = self.dropout1(attention_output1, training=training)
        output1 = self.layernorm1(x + attention_output1)

        attention_output2, _ = self.mha2(output1, encoder_output,
                                         encoder_output, padding_mask)
        attention_output2 = self.dropout2(attention_output2, training=training)
        output2 = self.layernorm2(output1 + attention_output2)

        dense_output = self.dense_hidden(output2)
        ffn_output = self.dense_output(dense_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        output3 = self.layernorm3(output2 + ffn_output)

        return output3
