import os

import librosa
import scipy.io.wavfile
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio


class TensorFlowIOTestCase(tf.test.TestCase):

    def test_tfio_librosa(self):
        path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'aihub_dataset', 'dummy_data', 'hobby_00000982.wav')

        # scipy
        sample_rate, signal = scipy.io.wavfile.read(path)
        signal = np.asarray(signal, dtype=np.float32) / 32768

        # librosa
        y, sr = librosa.load(path, sr=16000)

        # tensorflow-io
        tensor = tfio.audio.AudioIOTensor(path, dtype=tf.int64)
        tensor = tf.cast(tensor.to_tensor(), dtype=tf.float32) / 32768
        tensor = tf.squeeze(tensor, axis=-1).numpy()
        print(tensor.shape)

        self.assertAllEqual(tensor, y)
        self.assertAllEqual(tensor, signal)


if __name__ == "__main__":
    tf.test.main()
