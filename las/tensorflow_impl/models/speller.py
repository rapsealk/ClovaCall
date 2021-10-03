import tensorflow as tf
import numpy as np

from las.tensorflow_impl.models.listener import Listener


class Decoder(tf.keras.Model):
    pass


class Speller(Decoder):

    def __init__(self, units=512, output_shape=46):
        super(Speller, self).__init__()
        self.cells = [tf.keras.layers.LSTMCell(units),
                      tf.keras.layers.LSTMCell(units)]
        self.attention = tf.keras.layers.Attention()
        self.character_distribution = tf.keras.layers.Dense(output_shape)

        self.units = units

    def call(self, h, y):
        context = tf.zeros((1, self.units))
        hiddens = [cell.get_initial_state(tf.concat([y[:, 0], context], axis=-1))
                   for cell in self.cells]

        dist = []
        for i in range(y.shape[1]):
            x = tf.concat([hiddens[0][0], y[:, i], context], axis=-1)
            s, hiddens[0] = self.cells[0](x, hiddens[0])
            context, attn_w = self.attention([s, h], return_attention_scores=True)
            context = tf.squeeze(context, axis=1)
            s, hiddens[1] = self.cells[1](context, hiddens[1])
            dist.append(self.character_distribution(tf.concat([s, context], axis=-1)))

        return tf.nn.softmax(dist, axis=-1)
