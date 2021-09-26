import numpy as np
import tensorflow as tf

from las.tensorflow_impl.models.speller import AttentionLSTM


class LSTMTestCase(tf.test.TestCase):

    def test_outputs_equality(self):
        inputs = np.random.uniform(-1.0, 1.0, (1, 8, 2))
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)

        units = 4

        # (1) LSTM
        tf.random.set_seed(0)
        lstm = tf.keras.layers.LSTM(units, return_sequences=True)
        lstm_output = lstm(inputs)

        # (2) LSTMCell
        tf.random.set_seed(0)
        lstmcell = tf.keras.layers.LSTMCell(units)
        hidden = lstmcell.get_initial_state(inputs)

        outputs = []
        for i in range(inputs.shape[1]):
            output, *hidden = lstmcell(tf.expand_dims(inputs[0, i], axis=0), hidden)
            hidden = hidden[0]
            outputs.append(output)
        cell_outputs = tf.stack(outputs).numpy().reshape(lstm_output.shape)

        output_diff = np.abs(lstm_output.numpy() - cell_outputs)
        self.assertAllLessEqual(output_diff, 1e-6)

        # (3) AttentionLSTM
        tf.random.set_seed(0)
        model = AttentionLSTM(units)
        outputs = model(inputs).numpy().reshape(cell_outputs.shape)
        self.assertAllEqual(cell_outputs, outputs)


if __name__ == "__main__":
    tf.test.main()
