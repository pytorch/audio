from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

from torch import Tensor
from torchaudio import (
    compliance,
    datasets,
    kaldi_io,
    sox_effects,
    transforms
)
from torchaudio.backend import (
    _get_audio_backend_module,
    list_audio_backends,
    get_audio_backend,
    set_audio_backend,
    save_encinfo,
    sox_signalinfo_t,
    sox_encodinginfo_t,
    get_sox_option_t,
    get_sox_encoding_t,
    get_sox_bool,
    SignalInfo,
    EncodingInfo,
)
from torchaudio._internal import (
    module_utils as _mod_utils,
    misc_ops as _misc_ops,
)
from torchaudio.sox_effects import initialize_sox, shutdown_sox

try:
    from .version import __version__, git_version  # noqa: F401
except ImportError:
    pass


if _mod_utils.is_module_available('torchaudio._torchaudio'):
    from . import _torchaudio
    initialize_sox()


def load(filepath: Union[str, Path],
         out: Optional[Tensor] = None,
         normalization: Union[bool, float, Callable] = True,
         channels_first: bool = True,
         num_frames: int = 0,
         offset: int = 0,
         signalinfo: Optional[SignalInfo] = None,
         encodinginfo: Optional[EncodingInfo] = None,
         filetype: Optional[str] = None) -> Tuple[Tensor, int]:
    r"""Loads an audio file from disk into a tensor

    Args:
        filepath (str or Path): Path to audio file
        out (Tensor, optional): An output tensor to use instead of creating one. (Default: ``None``)
        normalization (bool, float, or callable, optional): If boolean `True`, then output is divided by `1 << 31`
            (assumes signed 32-bit audio), and normalizes to `[-1, 1]`.
            If `float`, then output is divided by that number
            If `Callable`, then the output is passed as a parameter
            to the given function, then the output is divided by
            the result. (Default: ``True``)
        channels_first (bool, optional): Set channels first or length first in result. (Default: ``True``)
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
        (Tensor, int): An output tensor of size `[C x L]` or `[L x C]` where L is the number
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
    return _get_audio_backend_module().load(
        filepath,
        out=out,
        normalization=normalization,
        channels_first=channels_first,
        num_frames=num_frames,
        offset=offset,
        signalinfo=signalinfo,
        encodinginfo=encodinginfo,
        filetype=filetype,
    )


def load_wav(filepath: Union[str, Path], **kwargs: Any) -> Tuple[Tensor, int]:
    r""" Loads a wave file. It assumes that the wav file uses 16 bit per sample that needs normalization by shifting
    the input right by 16 bits.

    Args:
        filepath (str or Path): Path to audio file

    Returns:
        (Tensor, int): An output tensor of size `[C x L]` or `[L x C]` where L is the number
        of audio frames and C is the number of channels. An integer which is the sample rate of the
        audio (as listed in the metadata of the file)
    """
    kwargs['normalization'] = 1 << 16
    return load(filepath, **kwargs)


def save(filepath: str, src: Tensor, sample_rate: int, precision: int = 16, channels_first: bool = True) -> None:
    r"""Convenience function for `save_encinfo`.

    Args:
        filepath (str): Path to audio file
        src (Tensor): An input 2D tensor of shape `[C x L]` or `[L x C]` where L is
            the number of audio frames, C is the number of channels
        sample_rate (int): An integer which is the sample rate of the
            audio (as listed in the metadata of the file)
        precision (int, optional): Bit precision (Default: ``16``)
        channels_first (bool, optional): Set channels first or length first in result. (
            Default: ``True``)
    """

    return _get_audio_backend_module().save(
        filepath, src, sample_rate, precision=precision, channels_first=channels_first
    )


def info(filepath: str) -> Tuple[SignalInfo, EncodingInfo]:
    r"""Gets metadata from an audio file without loading the signal.

     Args:
        filepath (str): Path to audio file

     Returns:
        (sox_signalinfo_t, sox_encodinginfo_t): A si (sox_signalinfo_t) signal
        info as a python object. An ei (sox_encodinginfo_t) encoding info

     Example
         >>> si, ei = torchaudio.info('foo.wav')
         >>> rate, channels, encoding = si.rate, si.channels, ei.encoding
    """
    return _get_audio_backend_module().info(filepath)
