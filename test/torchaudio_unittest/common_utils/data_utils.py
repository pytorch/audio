import os.path
from typing import Union

import torch


_TEST_DIR_PATH = os.path.realpath(
    os.path.join(os.path.dirname(__file__), '..'))


def get_asset_path(*paths):
    """Return full path of a test asset"""
    return os.path.join(_TEST_DIR_PATH, 'assets', *paths)


def convert_tensor_encoding(
    tensor: torch.tensor,
    dtype: torch.dtype,
):
    """Convert input tensor with values between -1 and 1 to integer encoding
    Args:
        tensor: input tensor, assumed between -1 and 1
        dtype: desired output tensor dtype
    Returns:
        Tensor: shape of (n_channels, sample_rate * duration)
    """
    if dtype == torch.int32:
        tensor *= (tensor > 0) * 2147483647 + (tensor < 0) * 2147483648
    if dtype == torch.int16:
        tensor *= (tensor > 0) * 32767 + (tensor < 0) * 32768
    if dtype == torch.uint8:
        tensor *= (tensor > 0) * 127 + (tensor < 0) * 128
        tensor += 128
    tensor = tensor.to(dtype)
    return tensor


def get_whitenoise(
    *,
    sample_rate: int = 16000,
    duration: float = 1,  # seconds
    n_channels: int = 1,
    seed: int = 0,
    dtype: Union[str, torch.dtype] = "float32",
    device: Union[str, torch.device] = "cpu",
    channels_first=True,
    scale_factor: float = 1,
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
        channels_first: whether first dimension is n_channels
        scale_factor: scale the Tensor before clamping and quantization
    Returns:
        Tensor: shape of (n_channels, sample_rate * duration)
    """
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    if dtype not in [torch.float32, torch.int32, torch.int16, torch.uint8]:
        raise NotImplementedError(f'dtype {dtype} is not supported.')
    # According to the doc, folking rng on all CUDA devices is slow when there are many CUDA devices,
    # so we only fork on CPU, generate values and move the data to the given device
    with torch.random.fork_rng([]):
        torch.random.manual_seed(seed)
        tensor = torch.randn([n_channels, int(sample_rate * duration)],
                             dtype=torch.float32, device='cpu')
    tensor /= 2.0
    tensor *= scale_factor
    tensor.clamp_(-1.0, 1.0)
    if not channels_first:
        tensor = tensor.t()
    return convert_tensor_encoding(tensor, dtype)


def get_sinusoid(
    *,
    frequency: float = 300,
    sample_rate: int = 16000,
    duration: float = 1,  # seconds
    n_channels: int = 1,
    dtype: Union[str, torch.dtype] = "float32",
    device: Union[str, torch.device] = "cpu",
    channels_first: bool = True,
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
    theta = torch.linspace(0, end, int(sample_rate * duration), dtype=torch.float32, device=device)
    tensor = torch.sin(theta, out=None).repeat([n_channels, 1])
    if not channels_first:
        tensor = tensor.t()
    return convert_tensor_encoding(tensor, dtype)
