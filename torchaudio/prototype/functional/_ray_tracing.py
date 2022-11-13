import torch
import torchaudio


def ray_tracing(
    room: torch.Tensor,
    source: torch.Tensor,
    mic_array: torch.Tensor,
    num_rays: int,  # TODO: find good default
    e_absorption: float = 0,  # TODO: accept tensor like in ISM
    scatter: float = 0,  # TODO: accept tensor like in ISM
    mic_radius: float = 0.5,
    sound_speed: float = 343,
    energy_thres: float = 1e-7,
    time_thres: float = 10,  # 10s
    hist_bin_size: float = 0.004,  # 4ms
) -> torch.Tensor:
    if mic_array.dim() == 2:
        if mic_array.shape[0] != 1:
            raise ValueError("Only 1 channel (1 microphone) supported.")
        mic_array = mic_array[0]
    if room.dtype != torch.float:
        raise ValueError(f"room must be of float dtype, got {room.dtype} instead.")
    if time_thres < hist_bin_size:
        raise ValueError(f"time_thres={time_thres} must be greater than hist_bin_size={hist_bin_size}.")

    return torch.ops.torchaudio.ray_tracing(
        room,
        source,
        mic_array,
        num_rays,
        e_absorption,
        scatter,
        mic_radius,
        sound_speed,
        energy_thres,
        time_thres,
        hist_bin_size,
    )
