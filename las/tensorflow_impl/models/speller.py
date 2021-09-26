import tensorflow as tf


class Decoder(tf.keras.Model):
    pass


class Speller(Decoder):

    def __init__(self, units, output_shape=46):
        super(Speller, self).__init__()
        self.attention_context = AttentionLSTM(units)
        self.rnn = tf.keras.Sequential([
            tf.keras.layers.LSTM(units, return_sequences=True),
            tf.keras.layers.LSTM(units)
        ])
        self.character_distribution = tf.keras.layers.Dense(output_shape)

    def call(self, x, y):
        c = self.attention_context(x, y)
        s = self.rnn(c)
        y = self.character_distribution(s)
        return tf.nn.softmax(y)


class AttentionLSTM(tf.keras.layers.Layer):

    def __init__(self, units, return_state=False):
        super(AttentionLSTM, self).__init__()
        self.cell = tf.keras.layers.LSTMCell(units)
        self.attention = tf.keras.layers.Attention()

    def call(self, x, y):
        h = self.cell.get_initial_state(y)
        outputs = []
        for i in range(y.shape[1]):
            s, *h = self.cell(y[:, i], h)
            c = self.attention([tf.expand_dims(s, axis=1), x])
            outputs.append(c)
            h = tf.squeeze(h, axis=0) + c

        # return tf.nn.softmax(tf.squeeze(tf.stack(outputs)))
        return tf.squeeze(tf.stack(outputs), axis=1)


if __name__ == "__main__":
    pass
