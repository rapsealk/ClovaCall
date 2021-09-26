import tensorflow as tf
import numpy as np

from listener import Listener
from speller import AttentionLSTM, Speller


class LAS(tf.keras.Model):

    def __init__(self):
        super(LAS, self).__init__()

    def call(self, x):
        pass

    def attend_and_spell(self, x):
        pass


def main():
    units = 2

    np.random.seed(42)
    x = np.random.uniform(-1.0, 1.0, (1, 4, 2))
    x = tf.convert_to_tensor(x, dtype=tf.float32)

    encoder = Listener(x.shape)
    h = encoder(x)
    print(f'h: {h} ({h.shape})')

    y = np.random.uniform(-1.0, 1.0, (1, 4, 8))
    y = tf.convert_to_tensor(y, dtype=tf.float32)

    # decoder = AttentionLSTM(units=units*4)
    decoder = Speller(units=units*4)
    y = decoder(h, y)
    print(f'y: {y} ({y.shape})')
    return

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
    s = np.random.uniform(-1.0, 1.0, (1, 1, 2))
    s = tf.convert_to_tensor(s, dtype=tf.float32)
    print(f'h: {h} ({h.shape}, dtype={h.dtype})')   # ((1, 4, 2))
    print(f's: {s} ({s.shape}, dtype={s.dtype})')   # ((1, 1, 2))

    attention = tf.keras.layers.Attention()
    c = attention([s, h])
    print(f'c: {c} ({c.shape})')    # ((1, 1, 2))


if __name__ == "__main__":
    main()
