import tensorflow as tf
import numpy as np

from las.tensorflow_impl.models.listener import Listener
from las.tensorflow_impl.models.speller import Speller


class Sequence2Sequence(tf.keras.Model):

    def __init__(self, encoder, decoder):
        super(Sequence2Sequence, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x, tokens):
        # Listen
        h = self.encoder(x)
        # Attend and Spell
        y = self.decoder(h, tokens)
        return y


def main():
    letters = 28
    encoder = Listener()
    decoder = Speller(output_shape=letters)
    seq2seq = Sequence2Sequence(encoder, decoder)

    inputs = np.random.uniform(-1.0, 1.0, size=(32, 64, 48))
    inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
    tokens = np.random.randint(0, 26, size=(32, 12, letters)).astype(np.float32)

    output = seq2seq(inputs, tokens)
    print(f'output: {output.shape}')


if __name__ == "__main__":
    main()
