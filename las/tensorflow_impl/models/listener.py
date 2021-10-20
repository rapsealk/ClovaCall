import tensorflow as tf
import numpy as np


class Encoder(tf.keras.Model):
    pass


class Listener(Encoder):

    def __init__(self, input_shape, units=256):
        super(Listener, self).__init__()
        self.pyramidal_rnn = PyramidalBiLSTM(input_shape=input_shape, units=units)

    def call(self, x):
        # FIXME: assert x.shape[0] == 32 (batch)
        x = self.pyramidal_rnn(x)
        return x


class PyramidalBiLSTM(tf.keras.layers.Layer):

    def __init__(self, input_shape, units=256):
        super(PyramidalBiLSTM, self).__init__()
        self.bottom = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units, input_shape=input_shape, return_sequences=True, kernel_initializer=tf.keras.initializers.RandomUniform(-0.1, 0.1))
        )
        self.pyramid = [
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units, return_sequences=True, kernel_initializer=tf.keras.initializers.RandomUniform(-0.1, 0.1))
            ),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units, return_sequences=True, kernel_initializer=tf.keras.initializers.RandomUniform(-0.1, 0.1))
            )
        ]

    def call(self, x):
        h = self.bottom(x)
        for layer in self.pyramid:
            if h.shape[1] % 2 == 1:
                h = tf.keras.layers.ZeroPadding1D(padding=(0, 1))(h)
            h = tf.reshape(h, (h.shape[0], -1, h.shape[-1] * 2))
            h = layer(h)
        return h


class PyramidalBiGRU(tf.keras.layers.Layer):
    pass


if __name__ == "__main__":
    inputs = np.random.uniform(-1.0, 1.0, (1, 4, 8))    # (batch, timesteps, feature)
    inputs = tf.convert_to_tensor(inputs)
    model = Listener()
    output = model(inputs)
    print(f'{inputs.shape} -> {output.shape}')
