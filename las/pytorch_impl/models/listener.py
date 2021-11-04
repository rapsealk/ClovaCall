import torch
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    pass


class Listener(Encoder):

    def __init__(self, input_shape, units=256):
        super(Listener, self).__init__()
        self.pyramidal_rnn = PyramidalBiLSTM(input_shape=input_shape, units=units)

    def forward(self, x):
        x = self.pyramidal_rnn(x)
        return x


class PyramidalBiLSTM(nn.Module):

    def __init__(self, input_shape, units=256, num_layers=3):
        super(PyramidalBiLSTM, self).__init__()
        self.bottom = nn.LSTM(input_shape, units, num_layers=1, bidirectional=True, batch_first=True)
        self.pyramid = nn.ModuleList([
            nn.LSTM(input_shape, units, bidirectional=True, batch_first=True),
            nn.LSTM(input_shape, units, bidirectional=True, batch_first=True)
        ])

        self._units = units
        self._num_layers = num_layers

    def forward(self, x):
        x = torch.from_numpy(x).float()
        h, c = self.reset_hidden_state(x.size(0), num_layers=2)
        y, (h, c) = self.bottom(x, (h, c))
        return y, (h, c)

    def reset_hidden_state(self, batch_size=32, num_layers=0, bidirectional=False):
        num_layers = num_layers or self.num_layers
        num_layers *= 2 if bidirectional else 1
        return (torch.zeros(num_layers, batch_size, self.units),
                torch.zeros(num_layers, batch_size, self.units))

    @property
    def units(self):
        return self._units

    @property
    def num_layers(self):
        return self._num_layers


if __name__ == "__main__":
    inputs = np.random.uniform(-1.0, 1.0, (1, 4, 8))    # (batch, timesteps, feature)
    model = Listener(input_shape=inputs.shape[-1])
    output, (h, c) = model(inputs)
    print(f'{inputs.shape} -> {output.shape}')
