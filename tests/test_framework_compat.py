import os
import sys
import random
import unittest

import numpy as np
import tensorflow as tf
import torch

from las.tensorflow_impl.models import Listener as TFListener
from las.pytorch_impl.models import Listener as TorchListener


class FrameworkCompatTestCase(unittest.TestCase):

    def setUp(self):
        self.seed = 42

    def test_initializer_compatibility(self):
        np.random.seed(self.seed)
        np_values = np.random.uniform(-0.1, 0.1, (4,))

        tf.random.set_seed(self.seed)
        tf_values_ = tf.random.uniform((4,), -0.1, 0.1)

        tf_initializer = tf.keras.initializers.RandomUniform(-0.1, 0.1)
        tf.random.set_seed(self.seed)
        tf_values = tf_initializer(shape=(4,))

        torch.manual_seed(self.seed)
        torch_values_ = torch.zeros((4,)).uniform_(-0.1, 0.1)

        torch_values = torch.zeros((4,))
        torch.manual_seed(self.seed)
        torch_values = torch.nn.init.uniform_(torch_values, -0.1, 0.1)

        print(f'np_values: {np_values}')
        print(f'tf_values_: {tf_values_}')
        print(f'tf_values: {tf_values}')
        print(f'torch_values_: {torch_values_}')
        print(f'torch_values: {torch_values}')

    def test_listener_compatability(self):
        inputs = np.random.uniform(-1.0, 1.0, (1, 4, 8))

        tf_listener = TFListener(input_shape=inputs.shape[-1:], units=4)
        tf_output = tf_listener(inputs)

        torch_listener = TorchListener(input_shape=inputs.shape[-1], units=4)
        # for (name, param), w in zip(torch_listener.named_parameters(), tf_listener.get_weights()):
        #     print(f'name: {name}, w.shape: {w.shape}')
        #     param.data = torch.from_numpy(w)
        with torch.no_grad():
            torch_output, (h, c) = torch_listener(inputs)

        print(f'tf_output: {tf_output[:, :, :4]} ({tf_output.shape})')
        print(f'torch_output: {torch_output[:, :, :4]} ({torch_output.shape})')

        # print(f'TensorFlow.get_weights(): {tf_listener.get_weights()}')
        for i, w in enumerate(tf_listener.get_weights()):
            print(f'w[{i}].shape: {w.shape}')
        for name, param in torch_listener.named_parameters():
            print(f'{name}: {param.shape}')
            if 'bias' in name:
                print(param)
        print(len(tuple(tf_listener.get_weights())), len(tuple(torch_listener.named_parameters())))
        # print(len(tf_listener.get_weights()))
        # print(f'PyTorch.named_parameters(): {list(torch_listener.named_parameters())}')


if __name__ == "__main__":
    unittest.main()
