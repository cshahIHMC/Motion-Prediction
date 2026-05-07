import torch
import torch.nn as nn


class FLD(nn.Module):
    def __init__(self,
                 observation_dim,
                 history_horizon,
                 latent_channel,
                 device,
                 output_dim=None,
                 dt=0.02,
                 encoder_shape=None,
                 decoder_shape=None,
                 **kwargs,
                 ):
        if kwargs:
            print("FLD.__init__ got unexpected arguments, which will be ignored: "
                  + str([key for key in kwargs.keys()]))
        super(FLD, self).__init__()
        self.input_channel = observation_dim
        self.output_channel = output_dim if output_dim is not None else observation_dim
        self.history_horizon = history_horizon
        self.latent_channel = latent_channel
        self.device = device
        self.dt = dt

        # history_horizon must be odd for symmetric same-size Conv1d padding
        assert history_horizon % 2 == 1, (
            f"history_horizon must be odd (got {history_horizon}). "
            "Use 151 instead of 150, for example."
        )

        self.args = torch.linspace(
            -(history_horizon - 1) * self.dt / 2,
            (history_horizon - 1) * self.dt / 2,
            self.history_horizon,
            dtype=torch.float,
            device=self.device,
        )
        self.freqs = torch.fft.rfftfreq(history_horizon, device=self.device)[1:] * history_horizon

        self.encoder_shape = encoder_shape if encoder_shape is not None else [int(self.input_channel / 3)]
        self.decoder_shape = decoder_shape if decoder_shape is not None else [int(self.input_channel / 3)]

        conv_padding = (history_horizon - 1) // 2  # same-size output for odd kernel

        encoder_layers = []
        curr_in_channel = self.input_channel
        for hidden_channel in self.encoder_shape:
            encoder_layers.append(
                nn.Conv1d(curr_in_channel, hidden_channel, history_horizon,
                          stride=1, padding=conv_padding, dilation=1,
                          groups=1, bias=True, padding_mode='zeros')
            )
            encoder_layers.append(nn.BatchNorm1d(num_features=hidden_channel))
            encoder_layers.append(nn.ELU())
            curr_in_channel = hidden_channel
        encoder_layers.append(
            nn.Conv1d(self.encoder_shape[-1], latent_channel, history_horizon,
                      stride=1, padding=conv_padding, dilation=1,
                      groups=1, bias=True, padding_mode='zeros')
        )
        encoder_layers.append(nn.BatchNorm1d(num_features=latent_channel))
        encoder_layers.append(nn.ELU())
        self.encoder = nn.Sequential(*encoder_layers).to(self.device)

        self.phase_encoder = nn.ModuleList()
        for _ in range(latent_channel):
            phase_encoder_layers = [
                nn.Linear(history_horizon, 2),
                nn.BatchNorm1d(num_features=2),
            ]
            self.phase_encoder.append(nn.Sequential(*phase_encoder_layers).to(self.device))

        decoder_layers = []
        curr_in_channel = latent_channel
        for hidden_channel in self.decoder_shape:
            decoder_layers.append(
                nn.Conv1d(curr_in_channel, hidden_channel, history_horizon,
                          stride=1, padding=conv_padding, dilation=1,
                          groups=1, bias=True, padding_mode='zeros')
            )
            decoder_layers.append(nn.BatchNorm1d(num_features=hidden_channel))
            decoder_layers.append(nn.ELU())
            curr_in_channel = hidden_channel
        decoder_layers.append(
            nn.Conv1d(self.decoder_shape[-1], self.output_channel, history_horizon,
                      stride=1, padding=conv_padding, dilation=1,
                      groups=1, bias=True, padding_mode='zeros')
        )
        self.decoder = nn.Sequential(*decoder_layers).to(self.device)

    def forward(self, x, k=1):
        # x: (batch_size, obs_dim, history_horizon)
        x = self.encoder(x)
        latent = x  # (batch_size, latent_channel, history_horizon)

        frequency, amplitude, offset = self.fft(x)  # each (batch_size, latent_channel)

        phase = torch.zeros((x.size(0), self.latent_channel), device=self.device, dtype=torch.float)
        for i in range(self.latent_channel):
            phase_shift = self.phase_encoder[i](x[:, i, :])
            phase[:, i] = torch.atan2(phase_shift[:, 1], phase_shift[:, 0]) / (2 * torch.pi)

        params = [phase, frequency, amplitude, offset]  # each (batch_size, latent_channel)

        # phase_dynamics: (k, batch_size, latent_channel)
        phase_dynamics = (
            phase.unsqueeze(0)
            + frequency.unsqueeze(0)
            * self.dt
            * torch.arange(0, k, device=self.device, dtype=torch.float, requires_grad=False).view(-1, 1, 1)
        )

        # z: (k, batch_size, latent_channel, history_horizon)
        z = (
            amplitude.unsqueeze(-1).unsqueeze(0)
            * torch.sin(
                2 * torch.pi
                * (
                    (frequency.unsqueeze(-1) * self.args).unsqueeze(0)
                    + phase_dynamics.unsqueeze(-1)
                )
            )
            + offset.unsqueeze(-1).unsqueeze(0)
        )
        signal = z[0]  # (batch_size, latent_channel, history_horizon)

        pred_dynamics = self.decoder(z.flatten(0, 1)).view(
            k, -1, self.output_channel, self.history_horizon
        )  # (k, batch_size, obs_dim, history_horizon)

        return pred_dynamics, latent, signal, params

    def fft(self, x):
        # x: (batch_size, latent_channel, history_horizon)
        rfft = torch.fft.rfft(x, dim=2)
        magnitude = rfft.abs()
        spectrum = magnitude[:, :, 1:]  # remove DC bin
        power = torch.square(spectrum)
        frequency = torch.sum(self.freqs * power, dim=2) / (torch.sum(power, dim=2) + 1e-8)
        amplitude = 2 * torch.sqrt(torch.sum(power, dim=2)) / self.history_horizon
        offset = rfft.real[:, :, 0] / self.history_horizon
        return frequency, amplitude, offset

    def get_dynamics_error(self, state_transitions, k):
        """Evaluate reconstruction error over a continuous trajectory.

        state_transitions: (num_clips, num_steps, obs_dim)
        """
        self.eval()
        num_windows = state_transitions.size(1) - self.history_horizon + 1
        seqs = torch.zeros(
            state_transitions.size(0), num_windows,
            self.history_horizon, state_transitions.size(2),
            dtype=torch.float, device=self.device, requires_grad=False,
        )
        for step in range(num_windows):
            seqs[:, step] = state_transitions[:, step:step + self.history_horizon, :]

        with torch.no_grad():
            pred_dynamics, _, _, _ = self.forward(
                seqs.flatten(0, 1).swapaxes(1, 2), k
            )
        pred_dynamics = pred_dynamics.swapaxes(2, 3).view(
            k, -1, num_windows, self.history_horizon, state_transitions.size(2)
        )
        error = torch.zeros(state_transitions.size(0), device=self.device, dtype=torch.float, requires_grad=False)
        for i in range(k):
            error += torch.square(
                pred_dynamics[i, :, :num_windows - i] - seqs[:, i:]
            ).mean(dim=(1, 2, 3))
        return error
