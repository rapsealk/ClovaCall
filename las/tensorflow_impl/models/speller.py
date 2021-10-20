import tensorflow as tf


class Decoder(tf.keras.Model):
    pass


class Speller(Decoder):

    def __init__(self, units=512, output_shape=46):
        super(Speller, self).__init__()
        self.cells = [tf.keras.layers.LSTMCell(units, kernel_initializer=tf.keras.initializers.RandomUniform(-0.1, 0.1)),
                      tf.keras.layers.LSTMCell(units, kernel_initializer=tf.keras.initializers.RandomUniform(-0.1, 0.1))]
        self.attention = tf.keras.layers.Attention()
        self.character_distribution = tf.keras.layers.Dense(output_shape, kernel_initializer=tf.keras.initializers.RandomUniform(-0.1, 0.1))

        self.units = units
        self.num_words = output_shape

    def call(self, h, y):
        context = tf.zeros((y.shape[0], self.units))
        hiddens = [cell.get_initial_state(tf.concat([tf.expand_dims(y[:, 0], axis=1), context], axis=-1))
                   for cell in self.cells]
        dist = tf.zeros((y.shape[0], self.num_words))   # y_hat

        distributions = []
        for i in range(y.shape[1]):
            """
            param:quries: Decoder hidden states, Shape=(B, 1, dec_D)
            param:values: Encoder outputs, Shape=(B, enc_T, enc_D)
            param:last_attn: Attention weight of previous step, Shape=(batch, enc_T)
            """
            x = tf.concat([hiddens[0][0], tf.expand_dims(y[:, i], axis=1), context, dist], axis=-1)
            s, hiddens[0] = self.cells[0](x, hiddens[0])
            s = tf.expand_dims(s, axis=1)
            context, attn_w = self.attention([s, h], return_attention_scores=True)
            context = tf.squeeze(context, axis=1)
            s, hiddens[1] = self.cells[1](context, hiddens[1])
            dist = self.character_distribution(tf.concat([s, context], axis=-1))
            distributions.append(dist)

        distributions = tf.transpose(distributions, perm=[1, 0, 2])

        # prob = tf.nn.softmax(distributions, axis=-1)
        prob = tf.nn.log_softmax(distributions, axis=-1)
        return prob
