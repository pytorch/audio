from typing import Optional

import torch
import scipy.io.wavfile


def normalize_wav(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.float32:
        pass
    elif tensor.dtype == torch.int32:
        tensor = tensor.to(torch.float32)
        tensor[tensor > 0] /= 2147483647.0
        tensor[tensor < 0] /= 2147483648.0
    elif tensor.dtype == torch.int16:
        tensor = tensor.to(torch.float32)
        tensor[tensor > 0] /= 32767.0
        tensor[tensor < 0] /= 32768.0
    elif tensor.dtype == torch.uint8:
        tensor = tensor.to(torch.float32) - 128
        tensor[tensor > 0] /= 127.0
        tensor[tensor < 0] /= 128.0
    return tensor


def get_wav_data(
    dtype: str,
    num_channels: int,
    *,
    num_frames: Optional[int] = None,
    normalize: bool = True,
    channels_first: bool = True,
):
    """Generate linear signal of the given dtype and num_channels

    Data range is
        [-1.0, 1.0] for float32,
        [-2147483648, 2147483647] for int32
        [-32768, 32767] for int16
        [0, 255] for uint8

    num_frames allow to change the linear interpolation parameter.
    Default values are 256 for uint8, else 1 << 16.
    1 << 16 as default is so that int16 value range is completely covered.
    """
    dtype_ = getattr(torch, dtype)

    if num_frames is None:
        if dtype == "uint8":
            num_frames = 256
        else:
            num_frames = 1 << 16

    if dtype == "uint8":
        base = torch.linspace(0, 255, num_frames, dtype=dtype_)
    if dtype == "float32":
        base = torch.linspace(-1.0, 1.0, num_frames, dtype=dtype_)
    if dtype == "int32":
        base = torch.linspace(-2147483648, 2147483647, num_frames, dtype=dtype_)
    if dtype == "int16":
        base = torch.linspace(-32768, 32767, num_frames, dtype=dtype_)
    data = base.repeat([num_channels, 1])
    if not channels_first:
        data = data.transpose(1, 0)
    if normalize:
        data = normalize_wav(data)
    return data


def load_wav(path: str, normalize=True, channels_first=True) -> torch.Tensor:
    """Load wav file without torchaudio"""
    sample_rate, data = scipy.io.wavfile.read(path)
    data = torch.from_numpy(data.copy())
    if data.ndim == 1:
        data = data.unsqueeze(1)
    if normalize:
        data = normalize_wav(data)
    if channels_first:
        data = data.transpose(1, 0)
    return data, sample_rate


def save_wav(path, data, sample_rate, channels_first=True):
    """Save wav file without torchaudio"""
    if channels_first:
        data = data.transpose(1, 0)
    scipy.io.wavfile.write(path, sample_rate, data.numpy())
