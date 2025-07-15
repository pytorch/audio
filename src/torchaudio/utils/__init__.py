from torio.utils import ffmpeg_utils

from . import sox_utils
from .download import download_asset

def load_torchcodec(file, channels_first=True, start_seconds=0.0, stop_seconds=None, **args):
    """Load audio data using an `AudioDecoder` from `torchcodec`.

    Args:
        file : (path, url, or file-like object)
            Source of the audio.
        channels_first : bool
            When `True`, the returned tensor has dimension `[channel, time]`. Otherwise, the dimension is `[time, channel]`.
        start_seconds : float
            Time, in seconds, of the start of the range. Default: 0.
        stop_seconds : float or None
            Time, in seconds, of the end of the range. As a half open range, the end is excluded. Default: None, which decodes samples until the end.

    Returns:
        tensor: The list of available backends.
        int: The sample rate.
    """

    if not args.pop('normalize', True):
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
    "download_asset",
    "sox_utils",
    "ffmpeg_utils",
    "load_torchcodec"
]
