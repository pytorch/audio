import torch
import torchaudio
from torch import Tensor



def ray_tracing(
    room: Tensor,
    source: torch.Tensor,
    mic_array: torch.Tensor,
    num_rays: int, # TODO: find good default
    e_absorption: float = 0,  # TODO: accept tensor like in ISM
    sound_speed: float = 343,
    energy_thres: float = 1e-7,
    time_thres: float = 10,  # 10s
    hist_bin_size: float = 0.004,  # 4ms
    # sample_rate: float = 16000.0,
) -> Tensor:
    if mic_array.dim() == 2:
        if mic_array.shape[0] != 1:
            raise ValueError("Only 1 channel supported")  # TODO: remove?
        mic_array = mic_array[0]
    if room.shape[0] > 2:
        raise ValueError("Only 2D room supported")  # TODO: support 3D !!

    assert time_thres > hist_bin_size  # TODO: raise proper ValueErRor
    return torch.ops.torchaudio.ray_tracing(room, source, mic_array, num_rays, e_absorption, sound_speed, energy_thres, time_thres, hist_bin_size)
