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
