"""
FLD_teacher_student.py

Teacher-student variant of Models/FLD.py for cross-modal gait estimation.
The original FLD is a single-modality autoencoder (one encoder, one decoder,
same feature space in and out, or a fixed sensor->joint cross-map). This
module gives TWO encoders -- one for privileged biomechanics windows
(teacher, training-only), one for wearable-sensor windows (student,
deployed) -- writing into the SAME Fourier latent parameterization
(phase, frequency, amplitude, offset), decoded by ONE SHARED decoder.

Does NOT modify Models/FLD.py or DataLoader/data_loader.py -- runs side by
side with them. Reuses the original's math by COMPOSITION, not
reimplementation: this module constructs real `FLD` instances internally and
calls their own `.encoder`, `.phase_encoder`, `.fft()`, `.decoder`, `.args`,
`.dt` -- the only code written fresh here is the ~10-line phase-propagation
formula from FLD.forward(), which is inlined there and not reachable as a
standalone callable (forward() always re-derives params from its own
encoder, it can't accept externally-supplied params) -- see decode()'s
docstring for exactly which lines it mirrors.

Tensor shape convention (matches Models/FLD.py / DataLoader/data_loader.py
exactly -- see their docstrings):
  context window   : (B, obs_dim, H)         -- channels-first
  future target     : (K, out_dim)            -- time-first
  latent params      : phase/frequency/amplitude/offset, each (B, latent_dim)
                        -- phase is a FRACTION OF A CYCLE (atan2(...)/(2*pi),
                        range (-0.5, 0.5]), NOT radians. Anything comparing
                        two phases (e.g. an alignment loss) must scale by
                        2*pi before using cos()/sin() -- see train_fld_teacher_student.py's
                        alignment_loss().
  z (pre-decode)     : (k, B, latent_dim, H)  -- k phase-shifted copies of the window
  decoded output      : (k, B, output_dim, H)
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from Models.FLD import FLD

# Type alias for the 4-tuple every encoder branch produces.
LatentParams = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]  # phase, frequency, amplitude, offset


class FLDTeacherStudent(nn.Module):
    """Two independent-weight encoders (teacher: biomech->latent, student:
    sensor->latent) writing into one shared Fourier latent space, decoded by
    one shared decoder (biomech_dim output). See module docstring for the
    composition-over-reimplementation design and tensor shape convention.

    At inference only student_encoder + decoder run -- forward_teacher()
    should simply not be called (nothing here forces that; it's a calling
    convention the training script follows, same as the original FLD's own
    forward() being called differently at train vs. eval time elsewhere in
    this repo).
    """

    def __init__(
        self,
        sensor_dim: int,
        biomech_dim: int,
        history_horizon: int,
        latent_dim: int,
        device: torch.device,
        dt: float,
        teacher_encoder_shape: Optional[List[int]] = None,
        student_encoder_shape: Optional[List[int]] = None,
        decoder_shape: Optional[List[int]] = None,
        teacher_input_dim: Optional[int] = None,
    ) -> None:
        """
        Args:
            sensor_dim: number of wearable-sensor channels (student encoder's input_channel).
            biomech_dim: number of biomechanics channels the SHARED DECODER outputs (and
                what forward_teacher()/forward_student() are both scored against). NOT
                necessarily the teacher encoder's own input width any more -- see
                teacher_input_dim.
            history_horizon: context window length in frames. Must be odd (enforced by
                the underlying FLD instances' own constructor assert -- reused, not
                re-asserted here).
            latent_dim: number of Fourier latent channels, shared by both branches (a
                hard requirement, not a convenience: forward_teacher()/forward_student()
                params must live in the SAME latent space for the alignment loss and the
                shared decoder to mean anything).
            device: torch device for internal buffers (self._core.args, arange calls in
                decode()), matching the original FLD's explicit-device-arg convention.
            dt: seconds per frame (1/sample_rate). Must match the sample rate BOTH
                branches' windows were captured at -- phase is referenced to this dt.
            teacher_encoder_shape: hidden Conv1d channel sizes for the teacher's encoder
                (default: FLD's own default, input_channel/3).
            student_encoder_shape: hidden Conv1d channel sizes for the student's encoder.
            decoder_shape: hidden Conv1d channel sizes for the shared decoder.
            teacher_input_dim: number of channels the TEACHER ENCODER reads, if different
                from biomech_dim -- e.g. biomech_dim (angle+velocity+moment) PLUS extra
                privileged channels (EMG) the teacher gets to condition on but the shared
                decoder never reconstructs. Defaults to None, which falls back to
                biomech_dim -- i.e. teacher input width == decoder output width, EXACTLY
                the original behaviour, so every existing caller that doesn't pass this is
                completely unaffected (same pattern as FLD.py's own
                use_positional_encoding=False default -- an opt-in generalisation, not a
                behaviour change).
        """
        super().__init__()
        self.sensor_dim = sensor_dim
        self.biomech_dim = biomech_dim
        self.history_horizon = history_horizon
        self.latent_dim = latent_dim
        self.device = device
        self.dt = dt
        self.teacher_input_dim = teacher_input_dim if teacher_input_dim is not None else biomech_dim

        # "_core" carries the teacher branch's encoder/phase-head AND the ONE
        # shared decoder. observation_dim=self.teacher_input_dim (defaults to
        # biomech_dim, see teacher_input_dim above) is what the TEACHER
        # ENCODER reads; output_dim=biomech_dim (always) is what the shared
        # decoder produces, regardless of how wide the teacher's own input is
        # -- FLD already supports observation_dim != output_dim natively (see
        # Models/FLD.py's own constructor), this just exposes that here.
        self._core = FLD(
            observation_dim=self.teacher_input_dim,
            output_dim=biomech_dim,
            history_horizon=history_horizon,
            latent_channel=latent_dim,
            device=device,
            dt=dt,
            encoder_shape=teacher_encoder_shape,
            decoder_shape=decoder_shape,
        )
        # "_student_core" is only harvested for its (differently-shaped, since
        # sensor_dim != biomech_dim in general) encoder + phase-head + fft.
        # Its own decoder is never used -- the shared one above is used
        # instead -- so it's replaced with a no-op right after construction
        # to avoid silently carrying an unused, never-trained decoder's
        # worth of dead parameters in this module's state_dict.
        self._student_core = FLD(
            observation_dim=sensor_dim,
            output_dim=biomech_dim,
            history_horizon=history_horizon,
            latent_channel=latent_dim,
            device=device,
            dt=dt,
            encoder_shape=student_encoder_shape,
            decoder_shape=decoder_shape,
        )
        self._student_core.decoder = nn.Identity()

    # -- read-only aliases (properties, not re-registered submodules -- see
    # module docstring: assigning an already-registered nn.Module to a new
    # attribute would double-register it under two state_dict keys) --------
    @property
    def teacher_encoder(self) -> nn.Module:
        return self._core.encoder

    @property
    def student_encoder(self) -> nn.Module:
        return self._student_core.encoder

    @property
    def decoder(self) -> nn.Module:
        """The one decoder shared by both branches."""
        return self._core.decoder

    def _encode(self, core: FLD, x: torch.Tensor) -> LatentParams:
        """Runs core.encoder -> core.fft -> core.phase_encoder -- i.e. exactly
        the pre-decode portion of FLD.forward() (Models/FLD.py lines ~120-130),
        reused verbatim via `core`'s own submodules/method. Only the small
        "assemble z + call decoder" tail of forward() is intentionally not
        invoked here -- see decode() below, which runs that tail against the
        SHARED decoder instead of whichever core.decoder belongs to `core`.

        Args:
            core: the FLD instance whose encoder/phase_encoder/fft to use
                (self._core for the teacher branch, self._student_core for
                the student branch).
            x: (B, core.input_channel, H) -- context window.

        Returns:
            (phase, frequency, amplitude, offset), each (B, latent_dim).
        """
        h = core.encoder(x)                          # (B, latent_dim, H) -- FLD.forward's `x = self.encoder(x)`
        frequency, amplitude, offset = core.fft(h)    # reuses FLD.fft verbatim
        phase = torch.zeros((h.size(0), core.latent_channel), device=h.device, dtype=torch.float)
        for i in range(core.latent_channel):
            phase_shift = core.phase_encoder[i](h[:, i, :])
            phase[:, i] = torch.atan2(phase_shift[:, 1], phase_shift[:, 0]) / (2 * math.pi)
        return phase, frequency, amplitude, offset

    def forward_teacher(self, biomech_win: torch.Tensor) -> LatentParams:
        """Encode a biomechanics context window into latent params. Training
        only -- never called at deployment (biomechanics ground truth isn't
        available outside a lab).

        Args:
            biomech_win: (B, teacher_input_dim, H) -- channels-first, matching
                DataLoader/teacher_student_data_loader.py's `biomech_window`.
                teacher_input_dim == biomech_dim unless the constructor's
                teacher_input_dim was set wider (e.g. biomech + EMG).

        Returns:
            (phase, frequency, amplitude, offset), each (B, latent_dim).
        """
        return self._encode(self._core, biomech_win)

    def forward_student(self, sensor_win: torch.Tensor) -> LatentParams:
        """Encode a wearable-sensor context window into latent params. This
        is what runs at deployment (with decode(), below).

        Args:
            sensor_win: (B, sensor_dim, H) -- channels-first, matching
                DataLoader/teacher_student_data_loader.py's `sensor_window`.

        Returns:
            (phase, frequency, amplitude, offset), each (B, latent_dim).
        """
        return self._encode(self._student_core, sensor_win)

    def decode(self, params: LatentParams, horizon_steps: int) -> torch.Tensor:
        """Propagate phase forward and decode, against the SHARED decoder.

        Reproduces FLD.forward()'s phase_dynamics/z formula (Models/FLD.py
        lines ~132-156) verbatim -- the one piece of FLD's math not reachable
        as a standalone callable (it's inlined in forward(), which always
        re-derives params from its own encoder rather than accepting
        externally-supplied ones). Uses self._core's own `args`/`dt`/
        `decoder`/`history_horizon`/`output_channel` (not re-derived) to stay
        byte-for-byte consistent with the original -- only the arithmetic
        expression itself is duplicated, every tensor/module it operates on
        is the original's own.

        Args:
            params: (phase, frequency, amplitude, offset), each (B, latent_dim)
                -- from forward_teacher() or forward_student().
            horizon_steps: k. k=1 decodes just the window the params were fit
                on (step 0 = the context's own reconstruction). k=forecast_horizon+1
                additionally rolls forward forecast_horizon genuinely-future
                steps -- see forecast() below, which applies the exact
                "drop step 0, take each shifted window's last frame"
                convention train_fld.py's _forecast() uses.

        Returns:
            (horizon_steps, B, biomech_dim, H) -- decoded, phase-shifted windows.
        """
        phase, frequency, amplitude, offset = params
        core = self._core
        B = phase.size(0)

        phase_dynamics = (
            phase.unsqueeze(0)
            + frequency.unsqueeze(0)
            * core.dt
            * torch.arange(0, horizon_steps, device=core.device, dtype=torch.float, requires_grad=False).view(-1, 1, 1)
        )
        z = (
            amplitude.unsqueeze(-1).unsqueeze(0)
            * torch.sin(
                2 * math.pi
                * (
                    (frequency.unsqueeze(-1) * core.args).unsqueeze(0)
                    + phase_dynamics.unsqueeze(-1)
                )
            )
            + offset.unsqueeze(-1).unsqueeze(0)
        )
        pred = core.decoder(z.flatten(0, 1)).view(
            horizon_steps, B, core.output_channel, core.history_horizon
        )
        return pred

    def forecast(self, params: LatentParams, forecast_horizon: int) -> torch.Tensor:
        """Convenience wrapper around decode(): genuinely-future single points
        only (drops step 0, takes the last frame of each of steps 1..K) --
        the exact convention train_fld.py's _forecast() uses, so this
        directly comparable against a `future_biomech` target of shape
        (K, B, biomech_dim) after a .permute(1, 0, 2).

        Args:
            params: (phase, frequency, amplitude, offset), each (B, latent_dim).
            forecast_horizon: K, number of genuinely-future steps.

        Returns:
            (forecast_horizon, B, biomech_dim) -- genuinely-future predictions.
        """
        pred_dynamics = self.decode(params, horizon_steps=forecast_horizon + 1)
        return pred_dynamics[1:, :, :, -1]
