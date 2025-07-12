from torio.utils import ffmpeg_utils

from . import sox_utils
from .download import download_asset
import os

def load_torchcodec(file, normalize=True, channels_first=True, start_seconds=0.0, stop_seconds=None, **args):
    if not normalize:
        raise Exception("Torchcodec does not support non-normalized file reading")
    try:
        from torchcodec.decoders import AudioDecoder
    except:
         raise Exception("To use this feature, you must install torchcodec. See https://github.com/pytorch/torchcodec for installation instructions")
    decoder = AudioDecoder(file, **args)
    samples = decoder.get_samples_played_in_range(start_seconds, stop_seconds)
    data = samples.data if channels_first else samples.data.T
    return (data, samples.sample_rate)

__all__ = [
    "load_torchcodec",
    "download_asset",
    "sox_utils",
    "ffmpeg_utils",
]
