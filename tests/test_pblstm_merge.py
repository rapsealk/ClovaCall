import tensorflow as tf
import numpy as np

np.random.seed(42)


class PyramidalBiLSTMTestCase(tf.test.TestCase):

    def test_reshape(self):
        inputs = tf.convert_to_tensor(
            np.random.uniform(-1.0, 1.0, (1, 8, 4)))
        r1 = tf.reshape(inputs, (inputs.shape[0], -1, inputs.shape[-1] * 2))
        r2 = tf.reshape(r1, (r1.shape[0], -1, r1.shape[-1] * 2))
        self.assertEqual(r1.shape, (1, 4, 8))
        self.assertEqual(r2.shape, (1, 2, 16))
        for i in range(inputs.shape[1]//2):
            self.assertAllEqual(
                np.concatenate(inputs[0, i*2:(i+1)*2].numpy(), axis=-1),
                r1[0, i].numpy())
        for i in range(r1.shape[1]//2):
            self.assertAllEqual(
                np.concatenate(r1[0, i*2:(i+1)*2].numpy(), axis=-1),
                r2[0, i].numpy())


if __name__ == "__main__":
    tf.test.main()
