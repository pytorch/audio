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
    if mic_array.dim() == 2:
        if mic_array.shape[0] != 1:
            raise ValueError("Only 1 channel (1 microphone) supported.")
        mic_array = mic_array[0]
    if room.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"room must be of float32 or float64 dtype, got {room.dtype} instead.")
    if time_thres < hist_bin_size:
        raise ValueError(f"time_thres={time_thres} must be greater than hist_bin_size={hist_bin_size}.")

    return torch.ops.torchaudio.ray_tracing(
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
