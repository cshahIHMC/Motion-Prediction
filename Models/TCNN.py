import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class Chomp1d(nn.Module):
    """Trims the extra padding added for causal convolutions."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Single dilated causal convolution block with residual connection.
    Input/output shape: (batch, channels, time)
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x: (batch, n_inputs, time)
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Stack of TemporalBlocks with exponentially increasing dilation.
    Receptive field grows as 2^num_levels * kernel_size.
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, stride=1,
                              dilation=dilation,
                              padding=(kernel_size - 1) * dilation,
                              dropout=dropout)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, num_inputs, time) -> (batch, num_channels[-1], time)
        return self.network(x)


class TCNModel(nn.Module):
    """
    TCN for single-step prediction.
    Uses the last time step's hidden state to produce output.

    Args:
        input_size:   number of input features (channels)
        output_size:  number of output features
        num_channels: list of hidden channel sizes per TCN level
        kernel_size:  convolution kernel size
        dropout:      dropout rate
    """
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x: (batch, input_size, seq_len)
        y = self.tcn(x)               # (batch, num_channels[-1], seq_len)
        return self.linear(y[:, :, -1])  # (batch, output_size)


class TCNModel_Forecast(nn.Module):
    """
    TCN for multi-step forecasting.
    Projects the last hidden state to (output_size * horizon) and reshapes.

    Args:
        input_size:   number of input features (channels)
        output_size:  number of output features per time step
        horizon:      number of future steps to predict
        num_channels: list of hidden channel sizes per TCN level
        kernel_size:  convolution kernel size
        dropout:      dropout rate

    Returns:
        (batch, output_size, horizon)
    """
    def __init__(self, input_size, output_size, horizon, num_channels, kernel_size=2, dropout=0.2):
        super(TCNModel_Forecast, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.horizon = horizon
        self.output_size = output_size
        self.linear = nn.Linear(num_channels[-1], output_size * horizon)

    def forward(self, x):
        # x: (batch, input_size, seq_len)
        y = self.tcn(x)                          # (batch, num_channels[-1], seq_len)
        out = self.linear(y[:, :, -1])            # (batch, output_size * horizon)
        # reshape to (batch, output_size, horizon)
        return out.view(x.size(0), self.output_size, self.horizon)
