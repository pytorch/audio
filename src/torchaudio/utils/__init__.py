from torio.utils import ffmpeg_utils

from . import sox_utils
from .download import download_asset

from torchcodec.decoders import AudioDecoder

def load_torchcodec(file, **args):
    decoder = AudioDecoder(file)
    if 'start_seconds' in args or 'stop_seconds' in args:
        samples = decoder.get_samples_played_in_range(**args)
    else:
        samples = decoder.get_all_samples()
    return (samples.data, samples.sample_rate)

__all__ = [
    "load_torchcodec",
    "download_asset",
    "sox_utils",
    "ffmpeg_utils",
]
