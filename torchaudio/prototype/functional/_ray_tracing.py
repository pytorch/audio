import torch
import torchaudio


def ray_tracing(
    room: torch.Tensor,
    source: torch.Tensor,
    mic_array: torch.Tensor,
    num_rays: int,  # TODO: find good default
    e_absorption: float = 0,  # TODO: accept tensor like in ISM
    scattering: float = 0,  # TODO: accept tensor like in ISM
    mic_radius: float = 0.5,
    sound_speed: float = 343,
    energy_thres: float = 1e-7,
    time_thres: float = 10,  # 10s
    hist_bin_size: float = 0.004,  # 4ms
) -> torch.Tensor:
    r"""Compute energy histogram via ray tracing.

    The implementation is based on *pyroomacoustics* :cite:`scheibler2018pyroomacoustics`.

    .. devices:: CPU

    .. properties:: Autograd TorchScript

    TODO: document args and returns.

    """
    if mic_array.dim() == 1:
        mic_array = mic_array[None, :]
    if mic_array.dim() != 2:
        raise ValueError(
            f"mic_array must be 1D tensor of shape D, or 2D tensor of shape (num_mics, D) where D is 2 or 3. Got shape = {mic_array.shape}."
        )
    if room.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"room must be of float32 or float64 dtype, got {room.dtype} instead.")
    if len(set([room.shape[0], source.shape[0], mic_array.shape[1]])) != 1:
        raise ValueError(
            f"Room dimension D must match with source and mic_array. Got {room.shape[0]}, {source.shape[0]}, and {mic_array.shape[1]}"
        )
    if time_thres < hist_bin_size:
        raise ValueError(f"time_thres={time_thres} must be greater than hist_bin_size={hist_bin_size}.")

    histograms = torch.ops.torchaudio.ray_tracing(
        room,
        source,
        mic_array,
        num_rays,
        e_absorption,
        scattering,
        mic_radius,
        sound_speed,
        energy_thres,
        time_thres,
        hist_bin_size,
    )

    if mic_array.shape[0] == 1:
        histograms = histograms[0]

    return histograms
