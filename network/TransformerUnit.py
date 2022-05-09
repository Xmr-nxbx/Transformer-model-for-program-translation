import numpy as np
import tensorflow as tf
from tensorflow.keras import *


float_precision = tf.float16
# float_precision = tf.float32
if float_precision is tf.float16:
    policy = mixed_precision.experimental.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)


def positional_encoding(position, d_model):
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # sin: even   cos: odd
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=float_precision)


class TokenEmbeddingLayer(layers.Layer):
    def __init__(self, vocab_size, d_model, seq_len_1, seq_len_2):
        super(TokenEmbeddingLayer, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len_1 = seq_len_1
        self.seq_len_2 = seq_len_2
        self.max_len = max(seq_len_1, seq_len_2)

        self.embedding = layers.Embedding(vocab_size, d_model)
        self.positional_encoding = positional_encoding(self.max_len, d_model)

    def call(self, inputs, start=0):
        seq_len = tf.shape(inputs)[1]

        pos_encoding = self.positional_encoding[:, start: start + seq_len, :]

        x = self.embedding(inputs)
        x *= tf.math.sqrt(tf.cast(self.d_model, float_precision))
        x += pos_encoding
        return x


class TransformerEncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dense_dim, dropout=.1):
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.dense_dim = dense_dim

        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=self.head_dim)
        self.dense_proj = Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(d_model), ]
        )

        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

        self.dropout_1 = layers.Dropout(dropout)
        self.dropout_2 = layers.Dropout(dropout)

    def call(self, inputs, mask=None, training=None):
        """
        Transformer EncoderLayer Unit

        :param inputs: [b, seq, hid]
        :param mask: [b, axis, seq] mask: 1 other: 0
        :param training: bool
        :return: [b, seq, hid]
        """
        if mask is not None:
            mask = 1 - mask
        attention_output, _ = self.attention(query=inputs, value=inputs, key=inputs, attention_mask=mask,
                                             return_attention_scores=True)
        attention_output = self.dropout_1(attention_output, training=training)

        add_norm1 = self.layernorm_1(inputs + attention_output)

        dense_output = self.dense_proj(add_norm1)
        dense_output = self.dropout_2(dense_output, training=training)

        add_norm2 = self.layernorm_2(add_norm1 + dense_output)
        return add_norm2


class TransformerEncoder(Model):
    def __init__(self, d_model, num_heads, dense_dim, layer_num, dropout=.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.layer_num = layer_num

        self.encoder_layers = [TransformerEncoderLayer(d_model, num_heads, dense_dim, dropout) for _ in
                               range(layer_num)]
        # self.dropout = layers.Dropout(dropout)

    def call(self, inputs, mask=None, training=None):
        """
        Transformer Encoder Unit

        :param inputs: [b, seq, hid]
        :param mask: [b, axis, seq]
        :param training: bool
        :return: [layer, b, seq, hid]
        """
        # x = self.dropout(inputs, training=training)  # [b, seq, hid]
        x = inputs
        encoder_outputs = tf.TensorArray(dtype=float_precision, size=self.layer_num)
        for i in range(self.layer_num):
            x = self.encoder_layers[i](x, mask=mask, training=training)
            encoder_outputs = encoder_outputs.write(i, x)

        encoder_outputs = encoder_outputs.stack()

        return encoder_outputs  # [layer_num, b, seq, hid]


class TransformerDecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dense_dim, dropout=.1):
        super(TransformerDecoderLayer, self).__init__()
        self.d_model = d_model
        self.dense_dim = dense_dim

        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=self.head_dim)
        self.attention_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=self.head_dim)

        self.dense_proj = Sequential([layers.Dense(dense_dim, activation='relu'), layers.Dense(d_model)])

        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

        self.dropout_1 = layers.Dropout(dropout)
        self.dropout_2 = layers.Dropout(dropout)
        self.dropout_3 = layers.Dropout(dropout)

    def get_decoder_sequence_mask(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        # [[1, 0, ..., 0],
        #  [1, 1, ..., 0],
        #  ...
        #  [1, 1, ..., 1]]
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len), tf.int32), -1, 0)  # pad: 0  other: 1
        mask = mask[tf.newaxis, ...]
        return mask

    def call(self, inputs, encoder_outputs, encoder_sequence_mask=None, decoder_sequence_mask=None, training=None):
        """
        Transformer Decoderlayer Unit

        :param inputs: [b, seq2, hid]
        :param encoder_outputs: [b, seq1, hid]
        :param encoder_sequence_mask: [b, axis, seq1] mask: 1  other: 0
        :param decoder_sequence_mask: [b, axis, seq2] mask: 1  other: 0
        :param training: bool
        :return: [b, seq2, hid]
        """
        decoder_mask = self.get_decoder_sequence_mask(inputs)
        if encoder_sequence_mask is not None:
            encoder_sequence_mask = 1 - encoder_sequence_mask
        if decoder_sequence_mask is not None:
            decoder_sequence_mask = 1 - decoder_sequence_mask
            # combination of sequential mask and origin mask
            decoder_mask = tf.minimum(decoder_mask, decoder_sequence_mask)

        attention_output_1, _ = self.attention_1(query=inputs, value=inputs, key=inputs, attention_mask=decoder_mask,
                                                 return_attention_scores=True)
        attention_output_1 = self.dropout_1(attention_output_1, training)
        out_1 = self.layernorm_1(attention_output_1 + inputs)

        attention_output_2, alignment_weight = self.attention_2(query=out_1, value=encoder_outputs, key=encoder_outputs,
                                                                attention_mask=encoder_sequence_mask,
                                                                return_attention_scores=True)
        attention_output_2 = self.dropout_2(attention_output_2, training)
        out_2 = self.layernorm_2(attention_output_2 + out_1)

        dense_output = self.dense_proj(out_2)
        dense_output = self.dropout_3(dense_output, training)
        out_3 = self.layernorm_3(dense_output + out_2)

        # out: [b, seq2, d_model]
        # alignment_weight: [b, num_heads, seq2, seq1]
        return out_3, alignment_weight


class TransformerDecoder(Model):
    def __init__(self, d_model, num_heads, dense_dim, layer_num, dropout=.1):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.layer_num = layer_num

        self.decoder_layers = [TransformerDecoderLayer(d_model, num_heads, dense_dim, dropout) for _ in
                               range(layer_num)]
        # self.dropout = layers.Dropout(dropout)

    def call(self, inputs, encoder_outputs, encoder_sequence_mask=None, decoder_sequence_mask=None, training=None):
        """
        Transformer Decoder Unit

        :param inputs: [b, seq2, hid]
        :param encoder_outputs: [layer, b, seq1, hid]
        :param encoder_sequence_mask: [b, axis, seq1] mask: 1 other: 0
        :param decoder_sequence_mask: [b, axis, seq2] mask: 1 other: 0
        :param training: bool
        :return: [layer, b, seq2, hid], weight [layer, b, num_heads, seq2, seq1]
        """
        # x = self.dropout(inputs, training=training)
        x = inputs
        decoder_outputs = tf.TensorArray(dtype=float_precision, size=self.layer_num)
        alignment_weights = tf.TensorArray(dtype=float_precision, size=self.layer_num)
        for i in range(self.layer_num):
            x, weight = self.decoder_layers[i](x, encoder_outputs[i], encoder_sequence_mask, decoder_sequence_mask,
                                               training=training)
            decoder_outputs = decoder_outputs.write(i, x)
            alignment_weights = alignment_weights.write(i, weight)

        return decoder_outputs.stack(), alignment_weights.stack()


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class Transformer(Model):
    def __init__(self, vocab_size, input_len, output_len, d_model, num_heads, dense_dim,
                 layer_num, dropout=.1):
        super(Transformer, self).__init__()
        self.vocab_size = vocab_size

        self.input_len_limit = input_len
        self.output_len_limit = output_len

        self.embedding = TokenEmbeddingLayer(vocab_size, d_model, seq_len_1=input_len, seq_len_2=output_len)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

        self.encoder = TransformerEncoder(d_model, num_heads, dense_dim, layer_num, dropout)
        self.decoder = TransformerDecoder(d_model, num_heads, dense_dim, layer_num, dropout)

        self.fc = layers.Dense(vocab_size, use_bias=False, dtype=tf.float32)

    def call(self, inputs, outputs, inputs_mask, outputs_mask=None, inputs_memory_mask=None, training=None):
        """

        :param inputs: tensor [b, inputs_seq]
        :param outputs: tensor [b, outputs_seq]
        :param inputs_mask: tensor [b, axis, inputs_seq] mask: 1 other: 0
        :param outputs_mask: tensor [b, axis, outputs_seq] mask: 1 other: 0
        :param inputs_memory_mask: [b, axis, inputs_seq] mask: 1 other: 0
        :param training: bool
        :return:
        """
        x = self.embedding(inputs)
        x = self.dropout1(x, training=training)

        encoder_outputs = self.encoder(x, inputs_mask, training)

        x = self.embedding(outputs)
        x = self.dropout2(x, training=training)

        decoder_outputs, alignment_weights = self.decoder(x, encoder_outputs, inputs_memory_mask, outputs_mask, training)

        logits = self.fc(decoder_outputs[-1])

        return logits, alignment_weights