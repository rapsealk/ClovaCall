import string

import tensorflow as tf
import numpy as np

from las.tensorflow_impl.models import Listener, Speller


def main():
    inputs = np.random.uniform(-1.0, 1.0, (1, 4, 8))    # (batch, timestep, size)
    inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
    vocab_size = len(string.ascii_lowercase) + 1 + 2

    # 1. Encoder (Listener)
    encoder = Listener(input_shape=inputs.shape)
    h = encoder(inputs)

    """
    # 2. Decoder (Speller)
    units = 512
    cells = [tf.keras.layers.LSTMCell(units),
             tf.keras.layers.LSTMCell(units)]
    attention = tf.keras.layers.Attention()

    context = tf.zeros((1, units))
    s_h = tf.concat([inputs[:, 0], context], axis=-1)
    hiddens = [cell.get_initial_state(s_h) for cell in cells]

    contexts = []
    for i in range(inputs.shape[1]):
        # s_i = RNN(s_i-1, y_i-1, c_i-1)
        # s == h[0]
        s_y_c = tf.concat([hiddens[0][0], inputs[:, i], context], axis=-1)
        s, hiddens[0] = cells[0](s_y_c, hiddens[0])
        context, attn_w = attention([s, h], return_attention_scores=True)
        context = tf.squeeze(context, axis=1)
        s, hiddens[1] = cells[1](context, hiddens[1])
        contexts.append(tf.concat([s, context], axis=-1))

    # 3. Character Distribution
    char_dist = tf.keras.layers.Dense(vocab_size)

    distributions = []
    for context in contexts:
        logit = char_dist(context)
        dist = tf.nn.softmax(logit, axis=-1)
        distributions.append(dist)

    print(distributions)
    """

    # 2. Decoder (Speller)
    decoder = Speller()
    prob = decoder(h, inputs)
    print(prob)


if __name__ == "__main__":
    main()
