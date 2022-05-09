import tensorflow as tf


def get_transformer_encoder_data(x, pad):
    x_mask = tf.cast(x == pad, tf.int32)[:, tf.newaxis, :]
    memory_mask = x_mask
    return x_mask, memory_mask


def get_transformer_decoder_data(y, pad, first_token):
    y_mask = tf.cast(y == pad, tf.int32)
    y_input = tf.pad(y, [[0, 0], [1, 0]], constant_values=first_token)[:, :-1]
    y_input_mask = tf.cast(y_input == pad, tf.int32)[:, tf.newaxis, :]
    return y_input, y_input_mask, y_mask
