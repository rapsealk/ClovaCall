import tensorflow as tf
import tensorflow_addons as tf_addons
import numpy as np


class Decoder(tf.keras.Model):
    pass


class Speller(Decoder):

    def __init__(self):
        super(Speller, self).__init__()
        self.lstm = tf.keras.layers.LSTM(32, return_sequences=True, return_state=True)

    def call(self, x):
        pass


class AttentionLSTM(tf.keras.layers.Layer):

    def __init__(self):
        super(AttentionLSTM, self).__init__()

    def call(self, x):
        pass


if __name__ == "__main__":
    units = 2

    np.random.seed(42)
    x = np.random.uniform(-1.0, 1.0, (1, 4, 2))
    x = tf.convert_to_tensor(x)

    """
    query: [batch_size, Tq, dim]
    value: [batch_size, Tv, dim]
    key: [batch_size, Tk, dim]
    """
    # attn = tf.keras.layers.Attention()
    # print(attn([x, x]))

    lstm = tf.keras.layers.LSTM(units, return_sequences=True)
    h = lstm(x)
    # y
    s = tf.convert_to_tensor(
        np.random.uniform(-1, 1, (1, 1, 2)).astype(np.float32))
    print(f'h: {h} ({h.shape}, dtype={h.dtype})')
    print(f's: {s} ({s.shape}, dtype={s.dtype})')
    # h_ = s + h  # tf.concat([s, h], axis=-1)
    # print(f'h_: {h_} ({h_.shape}, dtype={h_.dtype})')

    # attention = tf.keras.layers.Attention()
    # c = attention([s, h])
    # print(f'attn_ctx: {c} ({c.shape})')

    decoder_units = units

    attn_lstm = tf_addons.seq2seq.AttentionWrapper(
        tf.keras.layers.LSTMCell(units),
        # attention_mechanism=tf_addons.seq2seq.BahdanauAttention(units),
        attention_mechanism=tf_addons.seq2seq.LuongAttention(units)
    )

    output = attn_lstm(s, h)
