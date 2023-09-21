import math
from typing import Optional, Tuple, Union

import torch
import torchaudio
from torch import Tensor


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


def _adjust_coeff(coeffs: Union[float, torch.Tensor], name: str) -> torch.Tensor:
    """Validates and converts absorption or scattering parameters to a tensor with appropriate shape

    Args:
        coeff (float or torch.Tensor): The absorption coefficients of wall materials.

            If the dtype is ``float``, the absorption coefficient is identical for all walls and
            all frequencies.

            If ``absorption`` is a 1D Tensor, the shape must be `(2*dim,)`,
            where the values represent absorption coefficients of ``"west"``, ``"east"``,
            ``"south"``, ``"north"``, ``"floor"``, and ``"ceiling"``, respectively.

            If ``absorption`` is a 2D Tensor, the shape must be `(7, 2*dim)`,
            where 7 represents the number of octave bands.

    Returns:
        (torch.Tensor): The expanded coefficient.
            The shape is `(1, 6)` for single octave band case, and
            `(7, 6)` for multi octave band case.
    """
    num_walls = 6
    if isinstance(coeffs, float):
        if coeffs < 0:
            raise ValueError(f"`{name}` must be non-negative. Found: {coeffs}")
        return torch.full((1, num_walls), coeffs)
    if isinstance(coeffs, Tensor):
        if torch.any(coeffs < 0):
            raise ValueError(f"`{name}` must be non-negative. Found: {coeffs}")
        if coeffs.ndim == 1:
            if coeffs.numel() != num_walls:
                raise ValueError(
                    f"The shape of `{name}` must be ({num_walls},) when it is a 1D Tensor. "
                    f"Found the shape {coeffs.shape}."
                )
            return coeffs.unsqueeze(0)
        if coeffs.ndim == 2:
            if coeffs.shape[1] != num_walls:
                raise ValueError(
                    f"The shape of `{name}` must be (NUM_BANDS, {num_walls}) when it "
                    f"is a 2D Tensor. Found: {coeffs.shape}."
                )
            return coeffs
    raise TypeError(f"`{name}` must be float or Tensor.")


def _validate_inputs(
    room: torch.Tensor,
    source: torch.Tensor,
    mic_array: torch.Tensor,
):
    """Validate dimensions of input arguments, and normalize different kinds of absorption into the same dimension.

    Args:
        room (torch.Tensor): The size of the room. width, length (and height)
        source (torch.Tensor): Sound source coordinates. Tensor with dimensions `(dim,)`.
        mic_array (torch.Tensor): Microphone coordinates. Tensor with dimensions `(channel, dim)`.
    """
    if not (room.ndim == 1 and room.numel() == 3):
        raise ValueError(f"`room` must be a 1D Tensor with 3 elements. Found {room.shape}.")
    if not (source.ndim == 1 and source.numel() == 3):
        raise ValueError(f"`source` must be 1D Tensor with 3 elements. Found {source.shape}.")
    if not (mic_array.ndim == 2 and mic_array.shape[1] == 3):
        raise ValueError(f"`mic_array` must be a 2D Tensor with shape (num_channels, 3). Found {mic_array.shape}.")


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
    r"""Compute Room Impulse Response (RIR) based on the *image source method* :cite:`allen1979image`.
    The implementation is based on *pyroomacoustics* :cite:`scheibler2018pyroomacoustics`.

    .. devices:: CPU

    .. properties:: TorchScript

    Args:
        room (torch.Tensor): Room coordinates. The shape of `room` must be `(3,)` which represents
            three dimensions of the room.
        source (torch.Tensor): Sound source coordinates. Tensor with dimensions `(3,)`.
        mic_array (torch.Tensor): Microphone coordinates. Tensor with dimensions `(channel, 3)`.
        max_order (int): The maximum number of reflections of the source.
        absorption (float or torch.Tensor): The *absorption* :cite:`wiki:Absorption_(acoustics)`
            coefficients of wall materials for sound energy.
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
    _validate_inputs(room, source, mic_array)
    absorption = _adjust_coeff(absorption, "absorption")
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
        rir = torchaudio.functional.fftconvolve(rir, filters.unsqueeze(1).repeat(1, rir.shape[1], 1), mode="same")

    # sum up rir signals of all image sources into one waveform.
    rir = rir.sum(0)

    if output_length is not None:
        if output_length > rir.shape[-1]:
            rir = torch.nn.functional.pad(rir, (0, output_length - rir.shape[-1]), "constant", 0.0)
        else:
            rir = rir[..., :output_length]

    return rir


def ray_tracing(
    room: torch.Tensor,
    source: torch.Tensor,
    mic_array: torch.Tensor,
    num_rays: int,
    absorption: Union[float, torch.Tensor] = 0.0,
    scattering: Union[float, torch.Tensor] = 0.0,
    mic_radius: float = 0.5,
    sound_speed: float = 343.0,
    energy_thres: float = 1e-7,
    time_thres: float = 10.0,
    hist_bin_size: float = 0.004,
) -> torch.Tensor:
    r"""Compute energy histogram via ray tracing.

    The implementation is based on *pyroomacoustics* :cite:`scheibler2018pyroomacoustics`.

    ``num_rays`` rays are casted uniformly in all directions from the source;
    when a ray intersects a wall, it is reflected and part of its energy is absorbed.
    It is also scattered (sent directly to the microphone(s)) according to the ``scattering``
    coefficient.
    When a ray is close to the microphone, its current energy is recorded in the output
    histogram for that given time slot.

    .. devices:: CPU

    .. properties:: TorchScript

    Args:
        room (torch.Tensor): Room coordinates. The shape of `room` must be `(3,)` which represents
            three dimensions of the room.
        source (torch.Tensor): Sound source coordinates. Tensor with dimensions `(3,)`.
        mic_array (torch.Tensor): Microphone coordinates. Tensor with dimensions `(channel, 3)`.
        absorption (float or torch.Tensor, optional): The absorption coefficients of wall materials.
            (Default: ``0.0``).
            If the type is ``float``, the absorption coefficient is identical to all walls and
            all frequencies.
            If ``absorption`` is a 1D Tensor, the shape must be `(6,)`, representing absorption
            coefficients of ``"west"``, ``"east"``, ``"south"``, ``"north"``, ``"floor"``, and
            ``"ceiling"``, respectively.
            If ``absorption`` is a 2D Tensor, the shape must be  `(num_bands, 6)`.
            ``num_bands`` is the number of frequency bands (usually 7).
        scattering(float or torch.Tensor, optional): The scattering coefficients of wall materials. (Default: ``0.0``)
            The shape and type of this parameter is the same as for ``absorption``.
        mic_radius(float, optional): The radius of the microphone in meters. (Default: 0.5)
        sound_speed (float, optional): The speed of sound in meters per second. (Default: ``343.0``)
        energy_thres (float, optional): The energy level below which we stop tracing a ray. (Default: ``1e-7``)
            The initial energy of each ray is ``2 / num_rays``.
        time_thres (float, optional): The maximal duration for which rays are traced. (Unit: seconds) (Default: 10.0)
        hist_bin_size (float, optional): The size of each bin in the output histogram. (Unit: seconds) (Default: 0.004)

    Returns:
        (torch.Tensor): The 3D histogram(s) where the energy of the traced ray is recorded.
            Each bin corresponds to a given time slot.
            The shape is `(channel, num_bands, num_bins)`, where
            ``num_bins = ceil(time_thres / hist_bin_size)``.
            If both ``absorption`` and ``scattering`` are floats, then ``num_bands == 1``.
    """
    if time_thres < hist_bin_size:
        raise ValueError(
            "`time_thres` must be greater than `hist_bin_size`. "
            f"Found: hist_bin_size={hist_bin_size}, time_thres={time_thres}."
        )

    if room.dtype != source.dtype or source.dtype != mic_array.dtype:
        raise ValueError(
            "dtype of `room`, `source` and `mic_array` must match. "
            f"Found: `room` ({room.dtype}), `source` ({source.dtype}) and "
            f"`mic_array` ({mic_array.dtype})"
        )

    _validate_inputs(room, source, mic_array)
    absorption = _adjust_coeff(absorption, "absorption").to(room.dtype)
    scattering = _adjust_coeff(scattering, "scattering").to(room.dtype)

    # Bring absorption and scattering to the same shape
    if absorption.shape[0] == 1 and scattering.shape[0] > 1:
        absorption = absorption.expand(scattering.shape)
    if scattering.shape[0] == 1 and absorption.shape[0] > 1:
        scattering = scattering.expand(absorption.shape)
    if absorption.shape != scattering.shape:
        raise ValueError(
            "`absorption` and `scattering` must be broadcastable to the same number of bands and walls. "
            f"Inferred shapes absorption={absorption.shape} and scattering={scattering.shape}"
        )

    histograms = torch.ops.torchaudio.ray_tracing(
        room,
        source,
        mic_array,
        num_rays,
        absorption,
        scattering,
        mic_radius,
        sound_speed,
        energy_thres,
        time_thres,
        hist_bin_size,
    )

    return histograms
