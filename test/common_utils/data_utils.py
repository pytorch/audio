import os.path
from typing import Union

import torch


_TEST_DIR_PATH = os.path.realpath(
    os.path.join(os.path.dirname(__file__), '..'))


def get_asset_path(*paths):
    """Return full path of a test asset"""
    return os.path.join(_TEST_DIR_PATH, 'assets', *paths)


def get_whitenoise(
    *,
    sample_rate: int = 16000,
    duration: float = 1,  # seconds
    n_channels: int = 1,
    seed: int = 0,
    dtype: Union[str, torch.dtype] = "float32",
    device: Union[str, torch.device] = "cpu",
):
    """Generate pseudo audio data with whitenoise

    Args:
        sample_rate: Sampling rate
        duration: Length of the resulting Tensor in seconds.
        n_channels: Number of channels
        seed: Seed value used for random number generation.
            Note that this function does not modify global random generator state.
        dtype: Torch dtype
        device: device
    Returns:
        Tensor: shape of (n_channels, sample_rate * duration)
    """
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    shape = [n_channels, sample_rate * duration]
    # According to the doc, folking rng on all CUDA devices is slow when there are many CUDA devices,
    # so we only folk on CPU, generate values and move the data to the given device
    with torch.random.fork_rng([]):
        torch.random.manual_seed(seed)
        tensor = torch.randn(shape, dtype=dtype, device='cpu')
    tensor /= 2.0
    tensor.clamp_(-1.0, 1.0)
    return tensor.to(device=device)


def get_sinusoid(
    *,
    frequency: float = 300,
    sample_rate: int = 16000,
    duration: float = 1,  # seconds
    n_channels: int = 1,
    dtype: Union[str, torch.dtype] = "float32",
    device: Union[str, torch.device] = "cpu",
):
    """Generate pseudo audio data with sine wave.

    Args:
        frequency: Frequency of sine wave
        sample_rate: Sampling rate
        duration: Length of the resulting Tensor in seconds.
        n_channels: Number of channels
        dtype: Torch dtype
        device: device

    Returns:
        Tensor: shape of (n_channels, sample_rate * duration)
    """
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    pie2 = 2 * 3.141592653589793
    end = pie2 * frequency * duration
    theta = torch.linspace(0, end, sample_rate * duration, dtype=dtype, device=device)
    return torch.sin(theta, out=None).repeat([n_channels, 1])
