import tensorflow as tf


class DeepSpeech2(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super(DeepSpeech2, self).__init__(*args, **kwargs)
        self.invariant_convs = tf.keras.Sequential([
            tf.keras.layers.Conv2D(),
            tf.keras.layers.Conv2D(),
            tf.keras.layers.Conv2D()
        ])
        self.deep_recurrent = tf.keras.Sequential([
            tf.keras.layers.GRU(return_sequences=True),
            tf.keras.layers.GRU(return_sequences=True),
            tf.keras.layers.GRU(return_sequences=True),
            tf.keras.layers.GRU(return_sequences=True),
            tf.keras.layers.GRU(return_sequences=True),
            tf.keras.layers.GRU(return_sequences=True),
            tf.keras.layers.GRU()
        ])
        self.fully_connected = tf.keras.layers.Dense()

        # loss = tf.nn.ctc_loss()

    def call(self, x):
        pass


if __name__ == "__main__":
    pass
