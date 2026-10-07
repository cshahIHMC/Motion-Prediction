import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F

class LN_v2(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y

class Model(nn.Module):
    def __init__(self, input_channels, embedding_channels, intermediate_channels, time_range, window):
        super(Model, self).__init__()
        self.input_channels = input_channels
        self.embedding_channels = embedding_channels
        self.time_range = time_range
        self.window = window

        # intermediate_channels = int(input_channels/3)
        #
        # Accepts either a single int (ORIGINAL behaviour -- one hidden conv
        # stage: input_channels -> intermediate_channels -> embedding_channels)
        # or a list of ints (multiple CASCADED hidden stages, e.g. [10, 3]
        # tapering input_channels=30 -> 10 -> 3 before the final conv up to
        # embedding_channels) -- mirrors Models/FLD.py's own
        # encoder_shape/decoder_shape list convention. A bare int is
        # normalised to a length-1 list right away so everything below only
        # ever deals with the list form; existing callers passing a single
        # int are completely unaffected (same 2-conv-layer network as before).
        if isinstance(intermediate_channels, int):
            intermediate_channels = [intermediate_channels]
        self.intermediate_channels = list(intermediate_channels)

        self.tpi = Parameter(torch.from_numpy(np.array([2.0*np.pi], dtype=np.float32)), requires_grad=False)
        self.args = Parameter(torch.from_numpy(np.linspace(-self.window/2, self.window/2, self.time_range, dtype=np.float32)), requires_grad=False)
        self.freqs = Parameter(torch.fft.rfftfreq(time_range)[1:] * time_range / self.window, requires_grad=False) #Remove DC frequency

        conv_padding = int((time_range - 1) / 2)

        # Encoder: input_channels -> intermediate_channels[0] -> ... ->
        # intermediate_channels[-1] -> embedding_channels. Each hidden stage
        # is Conv1d + LN_v2(time_range) + ELU (same trio the original single-
        # stage version used); the FINAL stage is a plain Conv1d with no
        # norm/activation after it -- `latent` (in forward()) is that raw
        # conv output, exactly as before.
        encoder_layers = []
        curr_in = input_channels
        for hidden in self.intermediate_channels:
            encoder_layers.append(
                nn.Conv1d(curr_in, hidden, time_range, stride=1, padding=conv_padding,
                          dilation=1, groups=1, bias=True, padding_mode='zeros')
            )
            encoder_layers.append(LN_v2(time_range))
            encoder_layers.append(nn.ELU())
            curr_in = hidden
        encoder_layers.append(
            nn.Conv1d(curr_in, embedding_channels, time_range, stride=1, padding=conv_padding,
                      dilation=1, groups=1, bias=True, padding_mode='zeros')
        )
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc = torch.nn.ModuleList()
        for i in range(embedding_channels):
            self.fc.append(nn.Linear(time_range, 2))

        # Decoder mirrors the encoder in reverse: embedding_channels ->
        # intermediate_channels[-1] -> ... -> intermediate_channels[0] ->
        # input_channels. Same per-stage trio as the encoder; final stage is
        # a plain Conv1d (no norm/activation), matching the original.
        decoder_layers = []
        curr_in = embedding_channels
        for hidden in reversed(self.intermediate_channels):
            decoder_layers.append(
                nn.Conv1d(curr_in, hidden, time_range, stride=1, padding=conv_padding,
                          dilation=1, groups=1, bias=True, padding_mode='zeros')
            )
            decoder_layers.append(LN_v2(time_range))
            decoder_layers.append(nn.ELU())
            curr_in = hidden
        decoder_layers.append(
            nn.Conv1d(curr_in, input_channels, time_range, stride=1, padding=conv_padding,
                      dilation=1, groups=1, bias=True, padding_mode='zeros')
        )
        self.decoder = nn.Sequential(*decoder_layers)

    #Returns the frequency for a function over a time window in s
    def FFT(self, function, dim):
        rfft = torch.fft.rfft(function, dim=dim)
        magnitudes = rfft.abs()
        spectrum = magnitudes[:,:,1:] #Spectrum without DC component
        power = spectrum**2

        #Frequency
        freq = torch.sum(self.freqs * power, dim=dim) / torch.sum(power, dim=dim)

        #Amplitude
        amp = 2 * torch.sqrt(torch.sum(power, dim=dim)) / self.time_range

        #Offset
        offset = rfft.real[:,:,0] / self.time_range #DC component

        return freq, amp, offset

    def forward(self, x):
        y = x

        # Signal Embedding

        # y = y.reshape(y.shape[0], self.input_channels, self.time_range)

        y = self.encoder(y)

        latent = y #Save latent for returning

        #Frequency, Amplitude, Offset
        f, a, b = self.FFT(y, dim=2)

        #Phase
        p = torch.empty((y.shape[0], self.embedding_channels), dtype=torch.float32, device=y.device)
        for i in range(self.embedding_channels):
            v = self.fc[i](y[:,i,:])
            p[:,i] = torch.atan2(v[:,1], v[:,0]) / self.tpi

        #Parameters
        p = p.unsqueeze(2)
        f = f.unsqueeze(2)
        a = a.unsqueeze(2)
        b = b.unsqueeze(2)
        params = [p, f, a, b] #Save parameters for returning

        #Latent Reconstruction
        y = a * torch.sin(self.tpi * (f * self.args + p)) + b

        signal = y #Save signal for returning

        #Signal Reconstruction
        y = self.decoder(y)

        # y = y.reshape(y.shape[0], self.input_channels*self.time_range)

        # return y, latent, signal, params
        return params
