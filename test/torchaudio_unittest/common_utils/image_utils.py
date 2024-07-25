import torch
from torchaudio._internal.module_utils import is_module_available

if is_module_available("PIL"):
    from PIL import Image


def save_image(path, data, mode=None):
    """Save image.

    The input image is expected to be CHW order
    """
    if torch.is_tensor(data):
        data = data.numpy()
    if mode == "L" and data.ndim == 3:
        assert data.shape[0] == 1
        data = data[0]
    if data.ndim == 3:
        data = data.transpose(1, 2, 0)
    Image.fromarray(data, mode=mode).save(path)


def get_image(width, height, grayscale=False):
    """Generate image Tensor, returns CHW"""
    channels = 1 if grayscale else 3
    numel = width * height * channels
    img = torch.arange(numel, dtype=torch.int64) % 256
    img = img.reshape(channels, height, width).to(torch.uint8)
    return img


def rgb_to_yuv_ccir(img):
    """rgb to yuv conversion ported from ffmpeg

    The input image is expected to be (..., channel, height, width).
    """
    assert img.dtype == torch.uint8
    img = img.to(torch.float32)

    r, g, b = torch.split(img, 1, dim=-3)

    # https://github.com/FFmpeg/FFmpeg/blob/870bfe16a12bf09dca3a4ae27ef6f81a2de80c40/libavutil/colorspace.h#L98
    y = 263 * r + 516 * g + 100 * b + 512 + 16384
    y /= 1024

    # https://github.com/FFmpeg/FFmpeg/blob/870bfe16a12bf09dca3a4ae27ef6f81a2de80c40/libavutil/colorspace.h#L102
    # shift == 0
    u = -152 * r - 298 * g + 450 * b + 512 - 1
    u /= 1024
    u += 128

    # https://github.com/FFmpeg/FFmpeg/blob/870bfe16a12bf09dca3a4ae27ef6f81a2de80c40/libavutil/colorspace.h#L106
    # shift == 0
    v = 450 * r - 377 * g - 73 * b + 512 - 1
    v /= 1024
    v += 128

    return torch.cat([y, u, v], -3).to(torch.uint8)


def rgb_to_gray(img):
    """rgb to gray conversion

    The input image is expected to be (..., channel, height, width).
    """
    assert img.dtype == torch.uint8
    img = img.to(torch.float32)

    r, g, b = torch.split(img, 1, dim=-3)

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.to(torch.uint8)
