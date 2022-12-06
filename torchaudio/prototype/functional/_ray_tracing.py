from typing import Union

import torch


def _validate_absorption_scattering(
    v: Union[float, torch.Tensor], name: str, D: int, dtype: torch.dtype
) -> torch.Tensor:
    """Validates and converts absorption or scattering parameters to a tensor with appropriate shape"""
    num_walls = 4 if D == 2 else 6
    if isinstance(v, float):
        out = torch.full(
            size=(
                1,
                num_walls,
            ),
            fill_value=v,
        )
    elif isinstance(v, torch.Tensor) and v.ndim == 1:
        if v.shape[0] != num_walls:
            raise ValueError(
                f"The shape of {name} must be (4,) or (6,) if it is a 1D Tensor."
                f"Found the shape of room is {D} and shape of {name} is {v.shape}."
            )
        out = v[None, :]
    elif isinstance(v, torch.Tensor) and v.ndim == 2:
        if v.shape[1] != num_walls:
            raise ValueError(
                f"The shape of {name} must be (num_bands, 4) for a 2D room or (num_bands, 6) "
                "for a 3D room if it is a 2D Tensor. "
                f"Found the shape of room is {D} and shape of {name} is {v.shape}."
            )
        out = v
    else:
        out = v
    assert out.ndim == 2
    out = out.to(dtype)

    return out


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

    ``num_rays`` rays are casted uniformly in all directions from the source; when a ray intersects a wall,
    it is reflected and part of its energy is absorbed. It is also scattered (sent directly to the microphone(s))
    according to the ``scattering`` coefficient. When a ray is close to the microphone, its current energy is
    recoreded in the output histogram for that given time slot.

    .. devices:: CPU

    .. properties:: TorchScript

    Args:
        room (torch.Tensor): The room dimensions. The shape is
            `(D,)`, where ``D`` is 2 if room is a 2D room, or 3 if room is a 3D room. All rooms
            are assumed to be "shoebox" rooms.
        source (torch.Tensor): The coordinate of the sound source. Tensor with dimensions `(D,)`.
        mic_array (torch.Tensor): The coordinate of microphone array. Tensor with dimensions `(channel, D)`.
        absorption (float or torch.Tensor, optional): The absorption coefficients of wall materials.
            (Default: ``0.0``).
            If the dtype is ``float``, the absorption coefficient is identical to all walls and
            all frequencies.
            If ``absorption`` is a 1D Tensor, the shape must be `(4,)` if the room is a 2D room,
            representing absorption coefficients of ``"west"``, ``"east"``, ``"south"``, and
            ``"north"`` walls, respectively.
            Or the shape must be `(6,)` if the room is a 3D room, representing absorption coefficients
            of ``"west"``, ``"east"``, ``"south"``, ``"north"``, ``"floor"``, and ``"ceiling"``, respectively.
            If ``absorption`` is a 2D Tensor, the shape must be `(num_bands, 4)` if the room is a 2D room,
            or `(num_bands, 6)` if the room is a 3D room. ``num_bands`` is the number of frequency bands (usually 7),
            but you can choose other values.
        scattering(float or torch.Tensor, optional): The scattering coefficients of wall materials.
            (Default: ``0.0``).  The shape and type of this parameter is the same as for ``absorption``.
        mic_radius(float, optional): The radius of the microphone in meters. (Default: 0.5)
        sound_speed (float, optional): The speed of sound in meters per second. (Default: ``343.0``)
        energy_thres (float, optional): The energy level below which we stop tracing a ray. (Default: ``1e-7``).
            The initial energy of each ray is ``2 / num_rays``.
        time_thres (float, optional):  The maximal duration (in seconds) for which rays are traced. (Default: 10.0)
        hist_bin_size (float, optional): The size (in seconds) of each bin in the output histogram. (Default: 0.004)
    Returns:
        (torch.Tensor): The 3D histogram(s) where the energy of the traced ray is recorded. Each bin corresponds
            to a given time slot. The shape is `(channel, num_bands, num_bins)`
            where ``num_bins = ceil(time_thres / hist_bin_size)``. If both ``absorption`` and ``scattering``
            are floats, then ``num_bands == 1``.
    """
    if room.ndim != 1 or room.shape[0] not in (2, 3):
        raise ValueError(f"room must be a 1D tensor of shape (2,) or (3,), got shape={room.shape}")
    D = room.shape[0]

    if mic_array.ndim == 1:
        mic_array = mic_array[None, :]
    if mic_array.ndim != 2:
        raise ValueError(
            "mic_array must be 1D tensor of shape (D,), or 2D tensor of shape (num_mics, D) "
            f"where D is 2 or 3. Got shape = {mic_array.shape}."
        )
    if room.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"room must be of float32 or float64 dtype, got {room.dtype} instead.")
    if not (room.dtype == source.dtype == mic_array.dtype):
        raise ValueError(
            "dtype of room, source and mic_array must be the same. "
            f"Got {room.dtype}, {source.dtype}, and {mic_array.dtype}"
        )
    if not (D == source.shape[0] == mic_array.shape[1]):
        raise ValueError(
            "Room dimension D must match with source and mic_array. "
            f"Got {D}, {source.shape[0]}, and {mic_array.shape[1]}"
        )
    if time_thres < hist_bin_size:
        raise ValueError(f"time_thres={time_thres} must be at least greater than hist_bin_size={hist_bin_size}.")

    absorption = _validate_absorption_scattering(absorption, name="absorption", D=D, dtype=room.dtype)
    scattering = _validate_absorption_scattering(scattering, name="scattering", D=D, dtype=room.dtype)

    # Bring absorption and scattering to the same shape
    if absorption.shape[0] == 1 and scattering.shape[0] > 1:
        absorption = absorption.expand(scattering.shape)
    if scattering.shape[0] == 1 and absorption.shape[0] > 1:
        scattering = scattering.expand(absorption.shape)
    if absorption.shape != scattering.shape:
        raise ValueError(
            "absorption and scattering must have the same number of bands and walls. "
            f"Inferred shapes are {absorption.shape}  and {scattering.shape}"
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
