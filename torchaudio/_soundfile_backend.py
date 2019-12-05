import os

import torch


def check_input(src):
    if not torch.is_tensor(src):
        raise TypeError("Expected a tensor, got %s" % type(src))
    if src.is_cuda:
        raise TypeError("Expected a CPU based tensor, got %s" % type(src))


def load(
    filepath,
    out=None,
    normalization=True,
    channels_first=True,
    num_frames=0,
    offset=0,
    filetype=None,
    **_,
):
    r"""Loads an audio file from disk into a tensor

    Args:
        filepath (str or pathlib.Path): Path to audio file
        out (torch.Tensor, optional): An output tensor to use instead of creating one. (Default: ``None``)
        normalization (bool, number, or callable, optional): If boolean `True`, then output is divided by `1 << 31`
            (assumes signed 32-bit audio), and normalizes to `[-1, 1]`.
            If `number`, then output is divided by that number
            If `callable`, then the output is passed as a parameter
            to the given function, then the output is divided by
            the result. (Default: ``True``)
        channels_first (bool): Set channels first or length first in result. (Default: ``True``)
        num_frames (int, optional): Number of frames to load.  0 to load everything after the offset.
            (Default: ``0``)
        offset (int, optional): Number of frames from the start of the file to begin data loading.
            (Default: ``0``)
        filetype (str, optional): A filetype or extension to be set if not determined automatically.
            (Default: ``None``)

    Returns:
        Tuple[torch.Tensor, int]: An output tensor of size `[C x L]` or `[L x C]` where L is the number
        of audio frames and C is the number of channels. An integer which is the sample rate of the
        audio (as listed in the metadata of the file)

    Example
        >>> data, sample_rate = torchaudio.load('foo.mp3')
        >>> print(data.size())
        torch.Size([2, 278756])
        >>> print(sample_rate)
        44100
        >>> data_vol_normalized, _ = torchaudio.load('foo.mp3', normalization=lambda x: torch.abs(x).max())
        >>> print(data_vol_normalized.abs().max())
        1.

    """
    # stringify if `pathlib.Path` (noop if already `str`)
    filepath = str(filepath)
    # check if valid file
    if not os.path.isfile(filepath):
        raise OSError("{} not found or is a directory".format(filepath))

    if num_frames < -1:
        raise ValueError("Expected value for num_samples -1 (entire file) or >=0")
    if num_frames == 0:
        num_frames = -1
    if offset < 0:
        raise ValueError("Expected positive offset value")

    import soundfile

    # initialize output tensor
    # TODO remove pysoundfile and call directly soundfile to avoid going through numpy
    if out is not None:
        check_input(out)
        _, sample_rate = soundfile.read(
            filepath, frames=num_frames, start=offset, always_2d=True, out=out
        )
    else:
        out, sample_rate = soundfile.read(
            filepath, frames=num_frames, start=offset, always_2d=True
        )
        out = torch.tensor(out).t()

    # normalize if needed
    # _audio_normalization(out, normalization)

    return out, sample_rate


def save(filepath, src, sample_rate, channels_first=True, **_):
    r"""Saves a tensor of an audio signal to disk as a standard format like mp3, wav, etc.

    Args:
        filepath (str): Path to audio file
        src (torch.Tensor): An input 2D tensor of shape `[C x L]` or `[L x C]` where L is
            the number of audio frames, C is the number of channels
        sample_rate (int): An integer which is the sample rate of the
            audio (as listed in the metadata of the file)
    """
    if channels_first:
        src = src.t()

    import soundfile

    return soundfile.write(filepath, src, sample_rate)


def info(filepath, **_):
    r"""Gets metadata from an audio file without loading the signal.

     Args:
        filepath (str): Path to audio file

     Returns:
        Object with information about a SoundFile

     Example
         >>> si, ei = torchaudio.info('foo.wav')
         >>> rate, channels, encoding = si.rate, si.channels, ei.encoding
     """
    import soundfile

    return soundfile.info(filepath)
