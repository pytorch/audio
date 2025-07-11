from torio.utils import ffmpeg_utils

from . import sox_utils
from .download import download_asset
import os

def load_torchcodec(file, normalize=True, channels_first=True, **args):
    if not normalize:
        raise Exception("Torchcodec does not support non-normalized file reading")
    from torchcodec.decoders import AudioDecoder
    decoder = AudioDecoder(file)
    if 'start_seconds' in args or 'stop_seconds' in args:
        samples = decoder.get_samples_played_in_range(**args)
    else:
        samples = decoder.get_all_samples()
    data = samples.data if channels_first else samples.data.T
    return (data, samples.sample_rate)

__all__ = [
    "load_torchcodec",
    "download_asset",
    "sox_utils",
    "ffmpeg_utils",
]
