import tensorflow as tf
import numpy as np


class LSTMTestCase(tf.test.TestCase):

    def test_outputs_equality(self):
        inputs = np.random.uniform(-1.0, 1.0, (1, 8, 2))
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)

        units = 4

        tf.random.set_seed(0)
        lstm = tf.keras.layers.LSTM(units, name='lstm')
        output = lstm(inputs)
        print(f'lstm: {output} ({output.shape}, dtype={output.dtype})')

        tf.random.set_seed(0)
        lstm = tf.keras.layers.LSTM(units, return_sequences=True, name='lstm_sequences')
        output = lstm(inputs)
        print(f'lstm(sequences): {output} ({output.shape}, dtype={output.dtype})')

        lstm = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
        # lstm.reset_states()
        output, *state = lstm(inputs)
        # print(f'output: {output} ({output.shape}, dtype={output.dtype})')
        # print(f'state: {state} ({state[0].shape}, {state[1].shape})')

        tf.random.set_seed(0)
        lstmcell = tf.keras.layers.LSTMCell(units, name='lstm_cell')
        # hidden = lstmcell.get_initial_state(inputs)
        hidden = [tf.zeros((1, 4)), tf.zeros((1, 4))]
        # print(f'hidden: {hidden} ({hidden[0].shape})')

        # output, *state = lstmcell(inputs[0], hidden)
        # print(f'cell: {output} ({output.shape}, dtype={output.dtype})')
        outputs = []
        print(f'inputs.shape[1]: {inputs.shape}')
        for i in range(inputs.shape[1]):
            # print(f'inputs_shape: {inputs[0, i].shape}')
            output, *hidden = lstmcell(tf.expand_dims(inputs[0, i], axis=0), hidden)
            outputs.append(output.numpy())
            hidden = hidden[0]
            # print(f'i({i}) output: {output.shape}, hidden: {len(hidden)}')
            # print(f'- hidden[0]: {len(hidden[0])}')
            # print(f'- hidden: {hidden}')
        print(f'outputs: {np.asarray(outputs)}')

        # lstmcell = tf.keras.layers.LSTMCell(units)
        # hidden_state = tf.zeros_like(memory_state)
        # output = lstmcell(inputs, hidden_state)
        # print(f'output: {output}')


if __name__ == "__main__":
    tf.test.main()
