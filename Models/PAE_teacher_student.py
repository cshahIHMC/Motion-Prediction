"""
PAE_teacher_student.py

Teacher-student variant of Models/PAE.py (Periodic Autoencoder), analogous to
Models/FLD_teacher_student.py but built on PAE's per-window autoencoding math
instead of FLD's propagate-and-forecast math.

TWO encoders -- teacher (privileged biomechanics: joint angle + velocity +
moment, 30ch, training-only) and student (wearable sensors only: body IMU +
insole IMU + insole force/COP, 48ch, what's deployed) -- write into the SAME
per-channel sinusoidal latent parameterization (phase, frequency, amplitude,
offset -- one sinusoid per latent channel), decoded by ONE SHARED decoder
back into the 30ch biomechanics window.

Unlike FLD_teacher_student.py, there is no forecast/propagation step here:
Models/PAE.py's own Model reconstructs the SAME window it encoded (an
ordinary autoencoder) -- it never rolls phase forward to predict genuinely-
future frames the way FLD.forward()'s phase_dynamics does. So both branches'
target is simply the time-aligned biomechanics window itself: "teacher
reconstructs its own input", "student reconstructs the teacher's input via
the shared decoder" -- matching the explicit "for the teacher ... reconstruct
the same" request this was built for.

Reuses Models/PAE.py's `Model` class by COMPOSITION, the same precedent
Models/FLD_teacher_student.py set for Models/FLD.py: this module constructs
TWO real PAE `Model` instances internally and calls their own `encoder`
(Conv1d(+LN_v2+ELU)* stack -- see Models/PAE.py, now possibly MULTIPLE
cascaded hidden stages, not just one -- ending in a plain Conv1d up to
embedding_channels), `fc` (per-channel phase head), and `FFT()` method
directly -- see _encode() below for exactly which lines of
PAE.Model.forward() this mirrors. Only the student's own `decoder` is
unused -- nulled with nn.Identity right after construction, same precedent
FLD_teacher_student.py sets for its own `_student_core.decoder =
nn.Identity()` -- because the SHARED decoder used by both branches is the
teacher core's own `decoder` instead (see decode() below).

Hidden-stage widths (teacher_intermediate_channels / student_intermediate_
channels, each a list of ints): taking inspiration from Models/FLD.py's own
encoder_shape/decoder_shape convention -- an explicit list of hidden-layer
widths supplied by the caller, NOT derived from the branch's own input width
(replaces an earlier floor-divide-by-3-from-input-width cascade -- see git
history). Default (when not passed) is [64, 32] for BOTH branches, same
values train_fld_teacher_student_100hz.py's own teacher_encoder_shape/
student_encoder_shape default to -- e.g. biomech_dim=30 -> 30 -> 64 -> 32 ->
embedding_channels(8), sensor_dim=48 -> 48 -> 64 -> 32 -> 8. Pass a different
list (from train_pae_teacher_student.py's CONFIG) to override.

Does NOT modify Models/PAE.py or DataLoader/pae_teacher_student_data_loader.py
-- runs side by side with them.

Tensor shape convention (channels-first, matches Models/PAE.py exactly):
  context window (either branch's input) : (B, obs_dim, H)
  latent params   : phase/frequency/amplitude/offset, each (B, latent_dim) --
                     phase is a FRACTION OF A CYCLE (atan2(...)/(2*pi), same
                     convention as Models/FLD_teacher_student.py -- NOT
                     radians. An alignment loss comparing two phases must
                     scale by 2*pi before cos()/sin() -- see
                     train_fld_teacher_student.py's alignment_loss(), reused
                     as-is by train_pae_teacher_student.py).
  reconstructed window                     : (B, biomech_dim, H)
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from Models.PAE import Model as PAE

# Type alias for the 4-tuple either encoder branch produces.
LatentParams = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]  # phase, frequency, amplitude, offset


# Default hidden-stage widths for BOTH branches when the caller doesn't pass
# its own -- an explicit constant, not derived from either branch's input
# width (see module docstring: this replaces an earlier floor-divide-by-3
# cascade). Same values train_fld_teacher_student_100hz.py's own
# teacher_encoder_shape/student_encoder_shape default to.
DEFAULT_INTERMEDIATE_CHANNELS = [64, 32]


class PAETeacherStudent(nn.Module):
    """Two independent-weight PAE encoders (teacher: biomech->latent,
    student: sensor->latent) writing into one shared sinusoidal latent space,
    reconstructed by one shared PAE decoder (biomech_dim output). See module
    docstring for the composition-over-reimplementation design and tensor
    shape convention.

    At inference only student_encoder (forward_student) + decode() run --
    forward_teacher() should simply not be called (nothing here forces that;
    it's a calling convention the training script follows, same as
    FLDTeacherStudent's own forward_teacher()/forward_student() split).
    """

    def __init__(
        self,
        sensor_dim: int,
        biomech_dim: int,
        history_horizon: int,
        latent_dim: int,
        device: torch.device,
        dt: float,
        teacher_intermediate_channels: Optional[List[int]] = None,
        student_intermediate_channels: Optional[List[int]] = None,
    ) -> None:
        """
        Args:
            sensor_dim: number of wearable-sensor channels (student encoder's input_channels).
            biomech_dim: number of biomechanics channels the teacher encoder reads AND the
                SHARED DECODER outputs (both branches are scored against this).
            history_horizon: window length in frames == PAE's `time_range`. Must be odd --
                PAE's Conv1d layers use kernel_size == time_range with padding
                int((time_range-1)/2) for same-size output, which only holds for odd
                time_range (mirrors FLD's own history_horizon%2==1 requirement).
            latent_dim: number of latent channels == PAE's `embedding_channels`, shared by
                both branches (both must write into the same-width latent space for the
                alignment loss and shared decoder to mean anything).
            device: torch device.
            dt: seconds per frame (1/sample_rate). Used to build PAE's `window` arg (total
                seconds spanned by the history_horizon-frame window), so PAE's internal
                `args`/`freqs` are scaled consistently with the real sample rate.
            teacher_intermediate_channels: PAE's `intermediate_channels` for the teacher's
                conv encoder/decoder -- an explicit list of hidden-stage widths (FLD-style,
                see module docstring), e.g. [64, 32] tapering biomech_dim=30 -> 64 -> 32 ->
                latent_dim. Defaults (None) to DEFAULT_INTERMEDIATE_CHANNELS ([64, 32]).
            student_intermediate_channels: same, for the student's conv encoder (its
                decoder half is never used -- see module docstring). Defaults to
                DEFAULT_INTERMEDIATE_CHANNELS ([64, 32]).
        """
        super().__init__()
        assert history_horizon % 2 == 1, (
            f"history_horizon must be odd (got {history_horizon}) -- PAE's Conv1d layers "
            f"need symmetric same-size padding (kernel_size == time_range)."
        )
        self.sensor_dim = sensor_dim
        self.biomech_dim = biomech_dim
        self.history_horizon = history_horizon
        self.latent_dim = latent_dim
        self.device = device
        self.dt = dt

        # Total seconds spanned by the window -- matches Models/FLD.py's own
        # `self.args = linspace(-(H-1)*dt/2, (H-1)*dt/2, H)` span exactly, so
        # PAE's identically-shaped `self.args = linspace(-window/2, window/2,
        # time_range)` steps by the same dt between consecutive samples.
        window = (history_horizon - 1) * dt

        t_channels = (teacher_intermediate_channels if teacher_intermediate_channels is not None
                      else DEFAULT_INTERMEDIATE_CHANNELS)
        s_channels = (student_intermediate_channels if student_intermediate_channels is not None
                      else DEFAULT_INTERMEDIATE_CHANNELS)
        self.teacher_intermediate_channels = list(t_channels)
        self.student_intermediate_channels = list(s_channels)

        # "_core" carries the teacher branch's encoder AND the one shared decoder.
        self._core = PAE(
            input_channels=biomech_dim, embedding_channels=latent_dim,
            intermediate_channels=t_channels, time_range=history_horizon, window=window,
        ).to(device)
        # "_student_core" is only harvested for its (differently-shaped,
        # since sensor_dim != biomech_dim) encoder + phase head + FFT. Its
        # own decoder is never used -- the shared one above is used instead
        # -- so it's replaced with a no-op right after construction, same
        # precedent Models/FLD_teacher_student.py sets for its own
        # `_student_core.decoder`.
        self._student_core = PAE(
            input_channels=sensor_dim, embedding_channels=latent_dim,
            intermediate_channels=s_channels, time_range=history_horizon, window=window,
        ).to(device)
        self._student_core.decoder = nn.Identity()

    def _encode(self, core: PAE, x: torch.Tensor) -> LatentParams:
        """Runs core.encoder -> core.FFT -> core.fc[i] -- i.e. exactly the
        pre-reconstruction portion of PAE.Model.forward(), reused verbatim
        via `core`'s own submodules/method. Only the small "assemble signal +
        decode" tail of forward() is intentionally not invoked here -- see
        decode() below, which runs that tail against the SHARED decoder
        instead of whichever core's own decoder `core` has.

        Args:
            core: the PAE instance whose encoder/fc/FFT to use (self._core
                for the teacher branch, self._student_core for the student
                branch).
            x: (B, core.input_channels, H) -- context window.

        Returns:
            (phase, frequency, amplitude, offset), each (B, latent_dim).
        """
        h = core.encoder(x)   # (B, latent_dim, H) -- PAE.Model.forward's `latent = y`; may be >1 cascaded conv stage

        # torch.fft.rfft (inside core.FFT) has no bf16/fp16 kernel, so this
        # must run in fp32 regardless of an enclosing autocast context --
        # same caveat Models/FLD.py's own fft() documents and handles the
        # same way. core.FFT is PAE.Model's own method, reused verbatim;
        # only the fp32 cast is added here.
        with torch.autocast(device_type=h.device.type, enabled=False):
            frequency, amplitude, offset = core.FFT(h.float(), dim=2)   # reuses PAE.Model.FFT verbatim

        phase = torch.empty((h.size(0), core.embedding_channels), dtype=torch.float32, device=h.device)
        for i in range(core.embedding_channels):
            v = core.fc[i](h[:, i, :])
            phase[:, i] = torch.atan2(v[:, 1], v[:, 0]) / core.tpi
        return phase, frequency, amplitude, offset

    def forward_teacher(self, biomech_win: torch.Tensor) -> LatentParams:
        """Encode a biomechanics context window into latent params. Training
        only -- never called at deployment (biomechanics ground truth isn't
        available outside a lab).

        Args:
            biomech_win: (B, biomech_dim, H) -- channels-first, matching
                DataLoader/pae_teacher_student_data_loader.py's `biomech_window`.

        Returns:
            (phase, frequency, amplitude, offset), each (B, latent_dim).
        """
        return self._encode(self._core, biomech_win)

    def forward_student(self, sensor_win: torch.Tensor) -> LatentParams:
        """Encode a wearable-sensor context window into latent params. This
        is what runs at deployment (with decode(), below).

        Args:
            sensor_win: (B, sensor_dim, H) -- channels-first, matching
                DataLoader/pae_teacher_student_data_loader.py's `sensor_window`.

        Returns:
            (phase, frequency, amplitude, offset), each (B, latent_dim).
        """
        return self._encode(self._student_core, sensor_win)

    def decode(self, params: LatentParams) -> torch.Tensor:
        """Reconstruct a (B, biomech_dim, H) window from latent params,
        against the SHARED decoder. Reproduces PAE.Model.forward()'s
        `y = a*sin(tpi*(f*args+p)) + b` reconstruction line and the decode
        tail verbatim -- uses self._core's own `args`/`tpi`/`decoder` (not
        re-derived), so this stays byte-for-byte consistent with the
        original PAE math. Unlike Models/FLD_teacher_student.py's decode(),
        there is no phase-propagation/`k` argument -- PAE reconstructs the
        SAME window it was given, it does not roll phase forward to future
        steps.

        Args:
            params: (phase, frequency, amplitude, offset), each (B, latent_dim)
                -- from forward_teacher() or forward_student().

        Returns:
            (B, biomech_dim, H) -- reconstructed window.
        """
        phase, frequency, amplitude, offset = params
        core = self._core

        p = phase.unsqueeze(2)        # (B, latent_dim, 1)
        f = frequency.unsqueeze(2)
        a = amplitude.unsqueeze(2)
        b = offset.unsqueeze(2)

        signal = a * torch.sin(core.tpi * (f * core.args + p)) + b   # (B, latent_dim, H)
        return core.decoder(signal)   # (B, biomech_dim, H)
