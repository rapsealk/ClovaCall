import tensorflow as tf


class Decoder(tf.keras.Model):
    pass


class Speller(Decoder):

    def __init__(self, vocab_size=46, units=512):
        super(Speller, self).__init__()
        # (batch, sequence) -> (batch, sequence, embedded)
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=units)
        self.cells = [tf.keras.layers.LSTMCell(units, kernel_initializer=tf.keras.initializers.RandomUniform(-0.1, 0.1)),
                      tf.keras.layers.LSTMCell(units, kernel_initializer=tf.keras.initializers.RandomUniform(-0.1, 0.1))]
        self.attention = tf.keras.layers.Attention()
        self.character_distribution = tf.keras.layers.Dense(vocab_size, kernel_initializer=tf.keras.initializers.RandomUniform(-0.1, 0.1))

        self.units = units
        self.vocab_size = vocab_size

    def call(self, h, y):
        context = tf.zeros((y.shape[0], self.units))
        hiddens = [cell.get_initial_state(tf.concat([tf.expand_dims(y[:, 0], axis=1), context], axis=-1))
                   for cell in self.cells]
        y_hat = tf.zeros((y.shape[0], self.vocab_size))
        embedded = self.embedding(y)

        distributions = []
        for i in range(y.shape[1]):
            """
            param:quries: Decoder hidden states, Shape=(B, 1, dec_D)
            param:values: Encoder outputs, Shape=(B, enc_T, enc_D)
            param:last_attn: Attention weight of previous step, Shape=(batch, enc_T)
            """
            x = tf.concat([embedded[:, i], context, y_hat], axis=-1)
            s, hiddens[0] = self.cells[0](x, hiddens[0])
            s = tf.expand_dims(s, axis=1)
            context, attn_w = self.attention([s, h], return_attention_scores=True)
            context = tf.squeeze(context, axis=1)
            s, hiddens[1] = self.cells[1](context, hiddens[1])
            dist = self.character_distribution(tf.concat([s, context], axis=-1))
            y_hat = tf.nn.log_softmax(dist, axis=-1)    # argmax(log(P(y|x)))   [batch, num_chars]
            # y_hat = tf.nn.softmax(dist, axis=-1)
            distributions.append(y_hat)

        distributions = tf.transpose(distributions, perm=[1, 0, 2])

        return distributions
