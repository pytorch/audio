import math
from typing import Optional, Tuple, Union

import torch
import torchaudio
from torch import Tensor
from torchaudio._internal import module_utils as _mod_utils


def _compute_image_sources(
    room: torch.Tensor,
    source: torch.Tensor,
    max_order: int,
    absorption: torch.Tensor,
    scatter: Optional[torch.Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Compute image sources in a shoebox-like room.

    Args:
        room (torch.Tensor): The 1D Tensor to determine the room size. The shape is
            `(D,)`, where ``D`` is 2 if room is a 2D room, or 3 if room is a 3D room.
        source (torch.Tensor): The coordinate of the sound source. Tensor with dimensions
            `(D)`.
        max_order (int): The maximum number of reflections of the source.
        absorption (torch.Tensor): The absorption coefficients of wall materials.
            ``absorption`` is a Tensor with dimensions `(num_band, num_wall)`.
            The shape options are ``[(1, 4), (1, 6), (7, 4), (7, 6)]``.
            ``num_band`` is `1` if the coefficients is the same for all frequencies, or is `7`
            if the coefficients are different to different frequencies. `7` refers to the default number
            of octave bands. (See note in `simulate_rir_ism` method).
            ``num_wall`` is `4` if the room is a 2D room, representing absorption coefficients
            of ``"west"``, ``"east"``, ``"south"``, and ``"north"`` walls, respectively.
            Or it is `6` if the room is a 3D room, representing absorption coefficients
            of ``"west"``, ``"east"``, ``"south"``, ``"north"``, ``"floor"``, and ``"ceiling"``, respectively.
        scatter (torch.Tensor): The scattering coefficients of wall materials.
            The shape of ``scatter`` must match that of ``absorption``. If ``None``, it is not
            used in image source computation. (Default: ``None``)

    Returns:
        (torch.Tensor): The coordinates of all image sources within ``max_order`` number of reflections.
            Tensor with dimensions `(num_image_source, D)`.
        (torch.Tensor): The attenuation of corresponding image sources. Tensor with dimensions
            `(num_band, num_image_source)`.
    """
    if scatter is None:
        tr = torch.sqrt(1 - absorption)
    else:
        tr = torch.sqrt(1 - absorption) * torch.sqrt(1 - scatter)

    ind = torch.arange(-max_order, max_order + 1, device=source.device)
    if room.shape[0] == 2:
        XYZ = torch.meshgrid(ind, ind, indexing="ij")
    else:
        XYZ = torch.meshgrid(ind, ind, ind, indexing="ij")
    XYZ = torch.stack([c.reshape((-1,)) for c in XYZ], dim=-1)
    XYZ = XYZ[XYZ.abs().sum(dim=-1) <= max_order]

    # compute locations of image sources
    d = room[None, :]
    s = source[None, :]
    img_loc = torch.where(XYZ % 2 == 1, d * (XYZ + 1) - s, d * XYZ + s)

    # attenuation
    exp_lo = abs(torch.floor((XYZ / 2)))
    exp_hi = abs(torch.floor((XYZ + 1) / 2))
    t_lo = tr[:, ::2].unsqueeze(1).repeat(1, XYZ.shape[0], 1)  # (num_band, left walls)
    t_hi = tr[:, 1::2].unsqueeze(1).repeat(1, XYZ.shape[0], 1)  # (num_band, right walls)
    att = torch.prod((t_lo**exp_lo) * (t_hi**exp_hi), dim=-1)  # (num_band, num_image_source)
    return img_loc, att


def _hann(x: torch.Tensor, T: int):
    """Compute the Hann window where the values are truncated based on window length.
    torch.hann_window can only sample window function at integer points, the method is to sample
    continuous window function at non-integer points.

    Args:
        x (torch.Tensor): The fractional component of time delay Tensor.
        T (torch.Tensor): The window length of sinc function.

    Returns:
        (torch.Tensor): The hann window Tensor where values outside
            the sinc window (`T`) is set to zero.
    """
    y = torch.where(
        torch.abs(x) <= T / 2,
        0.5 * (1 + torch.cos(2 * math.pi * x / T)),
        x.new_zeros(1),
    )
    return y


def _frac_delay(delay: torch.Tensor, delay_i: torch.Tensor, delay_filter_length: int):
    """Compute fractional delay of impulse response signal.

    Args:
        delay (torch.Tensor): The time delay Tensor in samples.
        delay_i (torch.Tensor): The integer part of delay.
        delay_filter_length (int): The window length for sinc function.

    Returns:
        (torch.Tensor): The impulse response Tensor for all image sources.
    """
    if delay_filter_length % 2 != 1:
        raise ValueError("The filter length must be odd")

    pad = delay_filter_length // 2
    n = torch.arange(-pad, pad + 1, device=delay.device) + delay_i[..., None]
    delay = delay[..., None]

    return torch.special.sinc(n - delay) * _hann(n - delay, 2 * pad)


def _validate_inputs(
    room: torch.Tensor, source: torch.Tensor, mic_array: torch.Tensor, absorption: Union[float, torch.Tensor]
) -> torch.Tensor:
    """Validate dimensions of input arguments, and normalize different kinds of absorption into the same dimension.

    Args:
        room (torch.Tensor): Room coordinates. The shape of `room` must be `(3,)` which represents
            three dimensions of the room.
        source (torch.Tensor): Sound source coordinates. Tensor with dimensions `(3,)`.
        mic_array (torch.Tensor): Microphone coordinates. Tensor with dimensions `(channel, 3)`.
        absorption (float or torch.Tensor): The absorption coefficients of wall materials.
            If the dtype is ``float``, the absorption coefficient is identical for all walls and
            all frequencies.
            If ``absorption`` is a 1D Tensor, the shape must be `(6,)`, where the values represent
            absorption coefficients of ``"west"``, ``"east"``, ``"south"``, ``"north"``, ``"floor"``,
            and ``"ceiling"``, respectively.
            If ``absorption`` is a 2D Tensor, the shape must be `(7, 6)`, where 7 represents the number of octave bands.

    Returns:
        (torch.Tensor): The absorption Tensor. The shape is `(1, 6)` for single octave band case,
            or `(7, 6)` for multi octave band case.
    """
    if room.ndim != 1:
        raise ValueError(f"room must be a 1D Tensor. Found {room.shape}.")
    D = room.shape[0]
    if D != 3:
        raise ValueError(f"room must be a 3D room. Found {room.shape}.")
    num_wall = 6
    if source.shape[0] != D:
        raise ValueError(f"The shape of source must be `(3,)`. Found {source.shape}")
    if mic_array.ndim != 2:
        raise ValueError(f"mic_array must be a 2D Tensor. Found {mic_array.shape}.")
    if mic_array.shape[1] != D:
        raise ValueError(f"The second dimension of mic_array must be 3. Found {mic_array.shape}.")
    if isinstance(absorption, float):
        absorption = torch.ones(1, num_wall) * absorption
    elif isinstance(absorption, Tensor) and absorption.ndim == 1:
        if absorption.shape[0] != num_wall:
            raise ValueError(
                "The shape of absorption must be `(6,)` if it is a 1D Tensor." f"Found the shape {absorption.shape}."
            )
        absorption = absorption.unsqueeze(0)
    elif isinstance(absorption, Tensor) and absorption.ndim == 2:
        if absorption.shape != (7, num_wall):
            raise ValueError(
                "The shape of absorption must be `(7, 6)` if it is a 2D Tensor."
                f"Found the shape of room is {D} and shape of absorption is {absorption.shape}."
            )
        absorption = absorption
    else:
        absorption = absorption
    return absorption


@_mod_utils.requires_rir()
def simulate_rir_ism(
    room: torch.Tensor,
    source: torch.Tensor,
    mic_array: torch.Tensor,
    max_order: int,
    absorption: Union[float, torch.Tensor],
    output_length: Optional[int] = None,
    delay_filter_length: int = 81,
    center_frequency: Optional[torch.Tensor] = None,
    sound_speed: float = 343.0,
    sample_rate: float = 16000.0,
) -> Tensor:
    r"""Compute Room Impulse Response (RIR) based on the image source method.
    The implementation is based on *pyroomacoustics* :cite:`scheibler2018pyroomacoustics`.

    .. devices:: CPU

    .. properties:: Autograd TorchScript

    Args:
        room (torch.Tensor): Room coordinates. The shape of `room` must be `(3,)` which represents
            three dimensions of the room.
        source (torch.Tensor): Sound source coordinates. Tensor with dimensions `(3,)`.
        mic_array (torch.Tensor): Microphone coordinates. Tensor with dimensions `(channel, 3)`.
        max_order (int): The maximum number of reflections of the source.
        absorption (float or torch.Tensor): The absorption coefficients of wall materials.
            If the dtype is ``float``, the absorption coefficient is identical for all walls and
            all frequencies.
            If ``absorption`` is a 1D Tensor, the shape must be `(6,)`, where the values represent
            absorption coefficients of ``"west"``, ``"east"``, ``"south"``, ``"north"``, ``"floor"``,
            and ``"ceiling"``, respectively.
            If ``absorption`` is a 2D Tensor, the shape must be `(7, 6)`, where 7 represents the number of octave bands.
        output_length (int or None, optional): The output length of simulated RIR signal. If ``None``,
            the length is defined as

            .. math::
                \frac{\text{max\_d} \cdot \text{sample\_rate}}{\text{sound\_speed}} + \text{delay\_filter\_length}

            where ``max_d`` is the maximum distance between image sources and microphones.
        delay_filter_length (int, optional): The filter length for computing sinc function. (Default: ``81``)
        center_frequency (torch.Tensor, optional): The center frequencies of octave bands for multi-band walls.
            Only used when ``absorption`` is a 2D Tensor.
        sound_speed (float, optional): The speed of sound. (Default: ``343.0``)
        sample_rate (float, optional): The sample rate of the generated room impulse response signal.
            (Default: ``16000.0``)

    Returns:
        (torch.Tensor): The simulated room impulse response waveform. Tensor with dimensions
        `(channel, rir_length)`.

    Note:
        If ``absorption`` is a 2D Tensor and ``center_frequency`` is set to ``None``, the center frequencies
        of octave bands are fixed to ``[125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]``.
        Users need to tune the values of ``absorption`` to the corresponding frequencies.
    """
    absorption = _validate_inputs(room, source, mic_array, absorption)
    img_location, att = _compute_image_sources(room, source, max_order, absorption)

    # compute distances between image sources and microphones
    vec = img_location[:, None, :] - mic_array[None, :, :]
    dist = torch.linalg.norm(vec, dim=-1)  # (image_source, channel)

    img_src_att = att[..., None] / dist[None, ...]  # (band, image_source, channel)

    # separate delays in integer / frac part
    delay = dist * sample_rate / sound_speed  # distance to delay in samples
    delay_i = torch.ceil(delay)  # integer part

    # compute the shorts IRs corresponding to each image source
    irs = img_src_att[..., None] * _frac_delay(delay, delay_i, delay_filter_length)[None, ...]

    rir_length = int(delay_i.max() + irs.shape[-1])
    rir = torch.ops.torchaudio._simulate_rir(irs, delay_i.type(torch.int32), rir_length)

    # multi-band processing
    if absorption.shape[0] > 1:
        if center_frequency is None:
            center = torch.tensor(
                [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0], dtype=room.dtype, device=room.device
            )
        else:
            center = center_frequency
        # n_fft is set to 512 by default.
        filters = torch.ops.torchaudio._make_rir_filter(center, sample_rate, n_fft=512)
        l = rir.shape[-1]
        rir = torchaudio.prototype.functional.fftconvolve(rir, filters.unsqueeze(1).repeat(1, rir.shape[1], 1))
        rir = rir[..., (filters.shape[-1] - 1) // 2 : (filters.shape[-1]) // 2 + l]

    # sum up rir signals of all image sources into one waveform.
    rir = rir.sum(0)

    if output_length is not None:
        if output_length > rir.shape[-1]:
            rir = torch.nn.functional.pad(rir, (0, output_length - rir.shape[-1]), "constant", 0.0)
        else:
            rir = rir[..., :output_length]

    return rir
