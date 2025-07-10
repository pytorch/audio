from torio.utils import ffmpeg_utils

from . import sox_utils
from .download import download_asset
import os
from torchcodec.decoders import AudioDecoder

def load_torchcodec(file, **args):
    try:
        decoder = AudioDecoder(file)
        if 'start_seconds' in args or 'stop_seconds' in args:
            samples = decoder.get_samples_played_in_range(**args)
        else:
            samples = decoder.get_all_samples()
        return (samples.data, samples.sample_rate)
    except Exception as e:
        if "buggy FFmpeg version" in str(e) and "PYTEST_CURRENT_TEST" in os.environ:
            import pytest
            pytest.skip()
        else:
            raise e

__all__ = [
    "load_torchcodec",
    "download_asset",
    "sox_utils",
    "ffmpeg_utils",
]
