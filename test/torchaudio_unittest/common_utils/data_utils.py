import os.path
from typing import Optional, Union

import torch


_TEST_DIR_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))


def get_asset_path(*paths):
    """Return full path of a test asset"""
    return os.path.join(_TEST_DIR_PATH, "assets", *paths)


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
    if dtype not in [torch.float64, torch.float32, torch.int32, torch.int16, torch.uint8]:
        raise NotImplementedError(f"dtype {dtype} is not supported.")
    # According to the doc, folking rng on all CUDA devices is slow when there are many CUDA devices,
    # so we only fork on CPU, generate values and move the data to the given device
    with torch.random.fork_rng([]):
        torch.random.manual_seed(seed)
        tensor = torch.randn([n_channels, int(sample_rate * duration)], dtype=torch.float32, device="cpu")
    tensor /= 2.0
    tensor *= scale_factor
    tensor.clamp_(-1.0, 1.0)
    if not channels_first:
        tensor = tensor.t()

    tensor = tensor.to(device)

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
    num_frames = int(sample_rate * duration)
    # Randomize the initial phase. (except the first channel)
    theta0 = pie2 * torch.randn(n_channels, 1, dtype=torch.float32, device=device)
    theta0[0, :] = 0
    theta = torch.linspace(0, end, num_frames, dtype=torch.float32, device=device)
    theta = theta0 + theta
    tensor = torch.sin(theta, out=None)
    if not channels_first:
        tensor = tensor.t()
    return convert_tensor_encoding(tensor, dtype)


def get_spectrogram(
    waveform,
    *,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Optional[torch.Tensor] = None,
    center: bool = True,
    pad_mode: str = "reflect",
    power: Optional[float] = None,
):
    """Generate a spectrogram of the given Tensor

    Args:
        n_fft: The number of FFT bins.
        hop_length: Stride for sliding window. default: ``n_fft // 4``.
        win_length: The size of window frame and STFT filter. default: ``n_fft``.
        winwdow: Window function. default: Hann window
        center: Pad the input sequence if True. See ``torch.stft`` for the detail.
        pad_mode: Padding method used when center is True. Default: "reflect".
        power: If ``None``, raw spectrogram with complex values are returned,
            otherwise the norm of the spectrogram is returned.
    """
    hop_length = hop_length or n_fft // 4
    win_length = win_length or n_fft
    window = torch.hann_window(win_length, device=waveform.device) if window is None else window
    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=center,
        window=window,
        pad_mode=pad_mode,
        return_complex=True,
    )
    if power is not None:
        spec = spec.abs() ** power
    return spec
