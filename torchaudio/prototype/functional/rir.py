import math
from random import sample
from typing import Union

import torch
import torchaudio
from torch import Tensor

_CENTER_FREQUENCY = torch.tensor([125, 250, 500, 1000, 2000, 4000, 8000], dtype=torch.float)


def _compute_image_sources(room, source, max_order, e_abs, e_scatter=None):
    if e_scatter is None:
        e_scatter = torch.zeros_like(e_abs)
    # reflection coefficients
    tr = torch.sqrt(1 - e_abs) * torch.sqrt(1 - e_scatter)

    ind = torch.arange(-max_order, max_order + 1, device=source.device)
    XYZ = torch.meshgrid(ind, ind, ind, indexing="ij")
    XYZ = torch.stack([c.reshape((-1,)) for c in XYZ], dim=-1)
    XYZ = XYZ[XYZ.abs().sum(dim=-1) <= max_order]

    # location of image sources
    d = room[None, :]
    s = source[None, :]
    img_loc = torch.where(XYZ % 2 == 1, d * (XYZ + 1) - s, d * XYZ + s)

    # attenuation
    exp_lo = abs(torch.floor(XYZ / 2))
    exp_hi = abs(torch.floor((XYZ + 1) / 2))
    t_lo = tr[:, ::2].unsqueeze(1).repeat(1, XYZ.shape[0], 1)  # num_band, left walls
    t_hi = tr[:, 1::2].unsqueeze(1).repeat(1, XYZ.shape[0], 1)  # num_band, right walls
    att = torch.prod((t_lo**exp_lo) * (t_hi**exp_hi), dim=-1)  # num_band, num_image_source
    return img_loc, att


def _hann(x, T):
    """Compute he Hann window."""
    y = torch.where(
        torch.abs(x) <= T / 2,
        0.5 * (1 + torch.cos(2 * math.pi * x / T)),
        x.new_zeros(1),
    )
    return y


def _frac_delay(tau, filter_len=41):
    if filter_len % 2 != 1:
        raise ValueError("The filter length must be odd")

    pad = filter_len // 2
    n = torch.arange(-pad, pad + 1, device=tau.device)
    tau = tau[..., None]

    return torch.special.sinc(n - tau) * _hann(n - tau, 2 * pad)


def simulate_rir_ism(
    room: Tensor,
    source: Tensor,
    mic_array: Tensor,
    max_order: int,
    e_absorption: Union[float, Tensor],
    sound_speed: float = 343.0,
    sample_rate: float = 16000.0,
) -> Tensor:
    """Compute Room Impulse Response (RIR) based on image source method.

    Args:
        room (torch.Tensor): The 1D Tensor to determine the room size. The shape is
            `(D,)`, where D is 2 if room is a 2D room, or 3 if room is a 3D room.
        source (torch.Tensor): The coordinate of the sound source. Tensor with dimensions
            `(D)`.
        mic_array (torch.Tensor): The coordinate of microphone array. Tensor with dimensions
            `(channel, D)`.
        max_order (int): The maximum order of relfections of image sources.
        e_absorption (float or torch.Tensor): The absorption coefficients of wall materials.
            If the dtype is ``float``, the absorption coefficient is identical to all walls and
            all frequencies.
            If ``e_absorption`` is a 1D Tensor, the shape must be `(4)` if the room is a 2D room,
            or `(6)` if the room is a 3D room, where 4 represents 4 walls, 6 represents 4 walls,
            ceiling, and floor.
            If ``e_absorption`` is a 2D Tensor, the shape must be `(4, 7)` if the room is a 2D room,
            or `(6, 7)` if the room is a 3D room, where 7 represents the number of frequency bands.
        sound_speed (float): The speed of sound. (Default: ``343.0``)
        sample_rate (float): The sample rate of the generated room impulse response signal.
            (Default: ``16000.0``)

    Returns:
        (torch.Tensor): The simulated room impulse response waveform. Tensor with dimensions
            `(channel, rir_length)`.
    """
    if isinstance(e_absorption, float):
        e_absorption = torch.ones(1, 6) * e_absorption

    img_location, att = _compute_image_sources(room, source, max_order, e_absorption)
    vec = img_location[:, None, :] - mic_array[None, :, :]

    dist = torch.linalg.norm(vec, dim=-1)  # (num_band, n_img, n_mics)

    img_src_att = att[..., None] / dist[None, ...]  # (n_band, n_img_src, n_mics)

    # separate delays in integer / frac part
    delay = dist / sound_speed * sample_rate  # distance to delay in samples
    delay_i = torch.round(delay)  # integer part
    delay_f = delay - delay_i  # frac part, in [-0.5, 0.5)

    # compute the shorts IRs corresponding to each image source
    irs = img_src_att[..., None] * _frac_delay(delay_f, filter_len=81)[None, ...]

    rir_length = int(delay_i.max() + irs.shape[-1])
    rir = torch.ops.rir.build_rir(irs, delay_i.type(torch.int32), rir_length)
    if rir.shape[0] > 1:
        filters = torch.ops.rir.make_filter(_CENTER_FREQUENCY.to(room.device), sample_rate, 512)
        rir = torchaudio.prototype.functional.fftconvolve(rir, filters.unsqueeze(1).repeat(1, rir.shape[1], 1))
        rir = rir[..., (filters.shape[-1]-1) // 2 : -(filters.shape[-1]-1) // 2]
    return rir.sum(0)
