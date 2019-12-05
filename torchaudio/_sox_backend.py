import os.path

import torch

import torchaudio


def load(filepath,
         out=None,
         normalization=True,
         channels_first=True,
         num_frames=0,
         offset=0,
         signalinfo=None,
         encodinginfo=None,
         filetype=None,
         **_):
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
        signalinfo (sox_signalinfo_t, optional): A sox_signalinfo_t type, which could be helpful if the
            audio type cannot be automatically determined. (Default: ``None``)
        encodinginfo (sox_encodinginfo_t, optional): A sox_encodinginfo_t type, which could be set if the
            audio type cannot be automatically determined. (Default: ``None``)
        filetype (str, optional): A filetype or extension to be set if sox cannot determine it
            automatically. (Default: ``None``)

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

    # initialize output tensor
    if out is not None:
        torchaudio.check_input(out)
    else:
        out = torch.FloatTensor()

    if num_frames < -1:
        raise ValueError("Expected value for num_samples -1 (entire file) or >=0")
    if offset < 0:
        raise ValueError("Expected positive offset value")

    import _torch_sox
    sample_rate = _torch_sox.read_audio_file(
        filepath,
        out,
        channels_first,
        num_frames,
        offset,
        signalinfo,
        encodinginfo,
        filetype
    )

    # normalize if needed
    torchaudio._audio_normalization(out, normalization)

    return out, sample_rate


def save(filepath, src, sample_rate, precision=16, channels_first=True, **_):
    r"""Convenience function for `save_encinfo`.

    Args:
        filepath (str): Path to audio file
        src (torch.Tensor): An input 2D tensor of shape `[C x L]` or `[L x C]` where L is
            the number of audio frames, C is the number of channels
        sample_rate (int): An integer which is the sample rate of the
            audio (as listed in the metadata of the file)
        precision (int): Bit precision (Default: ``16``)
        channels_first (bool): Set channels first or length first in result. (
            Default: ``True``)
    """
    si = torchaudio.sox_signalinfo_t()
    ch_idx = 0 if channels_first else 1
    si.rate = sample_rate
    si.channels = 1 if src.dim() == 1 else src.size(ch_idx)
    si.length = src.numel()
    si.precision = precision
    return torchaudio.save_encinfo(filepath, src, channels_first, si)


def info(filepath, **_):
    r"""Gets metadata from an audio file without loading the signal.

     Args:
        filepath (str): Path to audio file

     Returns:
        Tuple[sox_signalinfo_t, sox_encodinginfo_t]: A si (sox_signalinfo_t) signal
        info as a python object. An ei (sox_encodinginfo_t) encoding info

     Example
         >>> si, ei = torchaudio.info('foo.wav')
         >>> rate, channels, encoding = si.rate, si.channels, ei.encoding
     """

    import _torch_sox
    return _torch_sox.get_info(filepath)
