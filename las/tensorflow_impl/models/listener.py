import tensorflow as tf
import numpy as np


class Listener(tf.keras.Model):

    def __init__(self, input_shape):
        super(Listener, self).__init__()
        self.pyramidal_rnn = PyramidalBiLSTM(input_shape=input_shape)

    def call(self, x):
        x = self.pyramidal_rnn(x)
        return x


class PyramidalBiLSTM(tf.keras.layers.Layer):

    def __init__(self, input_shape):
        super(PyramidalBiLSTM, self).__init__()
        self.bottom = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(4, return_sequences=True, input_shape=input_shape)
        )
        self.stack = tf.keras.Sequential([
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(4, return_sequences=True)
            ),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(4, return_sequences=True)
            )
        ])

    def call(self, x):
        h = self.bottom(x)
        for layer in self.stack.layers:
            h = tf.reshape(h, (h.shape[0], -1, h.shape[-1] * 2))
            h = layer(h)
        return h


class PyramidalBiGRU(tf.keras.layers.Layer):
    pass


if __name__ == "__main__":
    inputs = np.random.uniform(-1.0, 1.0, (1, 4, 8))    # (batch, timesteps, feature)
    inputs = tf.convert_to_tensor(inputs)
    model = Listener(input_shape=inputs.shape)
    output = model(inputs)
    print(f'{inputs.shape} -> {output.shape}')
