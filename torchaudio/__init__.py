import atexit
import os.path
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import torch
from torch import Tensor
from torchaudio import (
    compliance,
    datasets,
    kaldi_io,
    sox_effects,
    transforms
)
from torchaudio._backend import (
    check_input,
    _get_audio_backend_module,
    get_audio_backend,
    set_audio_backend,
)
from torchaudio._soundfile_backend import SignalInfo, EncodingInfo
from torchaudio._internal import module_utils as _mod_utils

try:
    from .version import __version__, git_version  # noqa: F401
except ImportError:
    pass


if _mod_utils.is_module_available('torchaudio._torchaudio'):
    from . import _torchaudio


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


@_mod_utils.requires_module('torchaudio._torchaudio')
def save_encinfo(filepath: str,
                 src: Tensor,
                 channels_first: bool = True,
                 signalinfo: Optional[SignalInfo] = None,
                 encodinginfo: Optional[EncodingInfo] = None,
                 filetype: Optional[str] = None) -> None:
    r"""Saves a tensor of an audio signal to disk as a standard format like mp3, wav, etc.

    Args:
        filepath (str): Path to audio file
        src (Tensor): An input 2D tensor of shape `[C x L]` or `[L x C]` where L is
            the number of audio frames, C is the number of channels
        channels_first (bool, optional): Set channels first or length first in result. (Default: ``True``)
        signalinfo (sox_signalinfo_t, optional): A sox_signalinfo_t type, which could be helpful if the
            audio type cannot be automatically determined (Default: ``None``).
        encodinginfo (sox_encodinginfo_t, optional): A sox_encodinginfo_t type, which could be set if the
            audio type cannot be automatically determined (Default: ``None``).
        filetype (str, optional): A filetype or extension to be set if sox cannot determine it
            automatically. (Default: ``None``)

    Example
        >>> data, sample_rate = torchaudio.load('foo.mp3')
        >>> torchaudio.save('foo.wav', data, sample_rate)

    """
    ch_idx, len_idx = (0, 1) if channels_first else (1, 0)

    # check if save directory exists
    abs_dirpath = os.path.dirname(os.path.abspath(filepath))
    if not os.path.isdir(abs_dirpath):
        raise OSError("Directory does not exist: {}".format(abs_dirpath))
    # check that src is a CPU tensor
    check_input(src)
    # Check/Fix shape of source data
    if src.dim() == 1:
        # 1d tensors as assumed to be mono signals
        src.unsqueeze_(ch_idx)
    elif src.dim() > 2 or src.size(ch_idx) > 16:
        # assumes num_channels < 16
        raise ValueError(
            "Expected format where C < 16, but found {}".format(src.size()))
    # sox stores the sample rate as a float, though practically sample rates are almost always integers
    # convert integers to floats
    if signalinfo:
        if signalinfo.rate and not isinstance(signalinfo.rate, float):
            if float(signalinfo.rate) == signalinfo.rate:
                signalinfo.rate = float(signalinfo.rate)
            else:
                raise TypeError('Sample rate should be a float or int')
        # check if the bit precision (i.e. bits per sample) is an integer
        if signalinfo.precision and not isinstance(signalinfo.precision, int):
            if int(signalinfo.precision) == signalinfo.precision:
                signalinfo.precision = int(signalinfo.precision)
            else:
                raise TypeError('Bit precision should be an integer')
    # programs such as librosa normalize the signal, unnormalize if detected
    if src.min() >= -1.0 and src.max() <= 1.0:
        src = src * (1 << 31)
        src = src.long()
    # set filetype and allow for files with no extensions
    extension = os.path.splitext(filepath)[1]
    filetype = extension[1:] if len(extension) > 0 else filetype
    # transpose from C x L -> L x C
    if channels_first:
        src = src.transpose(1, 0)
    # save data to file
    src = src.contiguous()
    _torchaudio.write_audio_file(filepath, src, signalinfo, encodinginfo, filetype)


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


@_mod_utils.requires_module('torchaudio._torchaudio')
def sox_signalinfo_t() -> SignalInfo:
    r"""Create a sox_signalinfo_t object. This object can be used to set the sample
    rate, number of channels, length, bit precision and headroom multiplier
    primarily for effects

    Returns: sox_signalinfo_t(object)
        - rate (float), sample rate as a float, practically will likely be an integer float
        - channel (int), number of audio channels
        - precision (int), bit precision
        - length (int), length of audio in samples * channels, 0 for unspecified and -1 for unknown
        - mult (float, optional), headroom multiplier for effects and ``None`` for no multiplier

    Example
        >>> si = torchaudio.sox_signalinfo_t()
        >>> si.channels = 1
        >>> si.rate = 16000.
        >>> si.precision = 16
        >>> si.length = 0
    """
    return _torchaudio.sox_signalinfo_t()


@_mod_utils.requires_module('torchaudio._torchaudio')
def sox_encodinginfo_t() -> EncodingInfo:
    r"""Create a sox_encodinginfo_t object.  This object can be used to set the encoding
    type, bit precision, compression factor, reverse bytes, reverse nibbles,
    reverse bits and endianness.  This can be used in an effects chain to encode the
    final output or to save a file with a specific encoding.  For example, one could
    use the sox ulaw encoding to do 8-bit ulaw encoding.  Note in a tensor output
    the result will be a 32-bit number, but number of unique values will be determined by
    the bit precision.

    Returns: sox_encodinginfo_t(object)
        - encoding (sox_encoding_t), output encoding
        - bits_per_sample (int), bit precision, same as `precision` in sox_signalinfo_t
        - compression (float), compression for lossy formats, 0.0 for default compression
        - reverse_bytes (sox_option_t), reverse bytes, use sox_option_default
        - reverse_nibbles (sox_option_t), reverse nibbles, use sox_option_default
        - reverse_bits (sox_option_t), reverse bytes, use sox_option_default
        - opposite_endian (sox_bool), change endianness, use sox_false

    Example
        >>> ei = torchaudio.sox_encodinginfo_t()
        >>> ei.encoding = torchaudio.get_sox_encoding_t(1)
        >>> ei.bits_per_sample = 16
        >>> ei.compression = 0
        >>> ei.reverse_bytes = torchaudio.get_sox_option_t(2)
        >>> ei.reverse_nibbles = torchaudio.get_sox_option_t(2)
        >>> ei.reverse_bits = torchaudio.get_sox_option_t(2)
        >>> ei.opposite_endian = torchaudio.get_sox_bool(0)

    """
    ei = _torchaudio.sox_encodinginfo_t()
    sdo = get_sox_option_t(2)  # sox_default_option
    ei.reverse_bytes = sdo
    ei.reverse_nibbles = sdo
    ei.reverse_bits = sdo
    return ei


@_mod_utils.requires_module('torchaudio._torchaudio')
def get_sox_encoding_t(i: int = None) -> EncodingInfo:
    r"""Get enum of sox_encoding_t for sox encodings.

    Args:
        i (int, optional): Choose type or get a dict with all possible options
            use ``__members__`` to see all options when not specified. (Default: ``None``)

    Returns:
        sox_encoding_t: A sox_encoding_t type for output encoding
    """
    if i is None:
        # one can see all possible values using the .__members__ attribute
        return _torchaudio.sox_encoding_t
    else:
        return _torchaudio.sox_encoding_t(i)


@_mod_utils.requires_module('torchaudio._torchaudio')
def get_sox_option_t(i: int = 2) -> Any:
    r"""Get enum of sox_option_t for sox encodinginfo options.

    Args:
        i (int, optional): Choose type or get a dict with all possible options
            use ``__members__`` to see all options when not specified.
            (Default: ``sox_option_default`` or ``2``)
    Returns:
        sox_option_t: A sox_option_t type
    """
    if i is None:
        return _torchaudio.sox_option_t
    else:
        return _torchaudio.sox_option_t(i)


@_mod_utils.requires_module('torchaudio._torchaudio')
def get_sox_bool(i: int = 0) -> Any:
    r"""Get enum of sox_bool for sox encodinginfo options.

    Args:
        i (int, optional): Choose type or get a dict with all possible options
            use ``__members__`` to see all options when not specified. (Default:
            ``sox_false`` or ``0``)

    Returns:
        sox_bool: A sox_bool type
    """
    if i is None:
        return _torchaudio.sox_bool
    else:
        return _torchaudio.sox_bool(i)


_SOX_INITIALIZED: Optional[bool] = False
# This variable has a micro lifecycle. (False -> True -> None)
# False: Not initialized
# True: Initialized
# None: Already shut down (should not be initialized again.)

_SOX_SUCCESS_CODE = 0
# defined at
# https://fossies.org/dox/sox-14.4.2/sox_8h.html#a8e07e80cebeff3339265d89c387cea93a9ef2b87ec303edfe40751d9a85fadeeb


@_mod_utils.requires_module("torchaudio._torchaudio")
def initialize_sox() -> int:
    """Initialize sox for use with effects chains.

    You only need to call this function once to use SoX effects chains multiple times.
    It is safe to call this function multiple times as long as ``shutdown_sox`` is not yet called.
    Once ``shutdown_sox`` is called, you can no longer use SoX effects and calling this function
    results in `RuntimeError`.

    Note:
        This function is not required for simple loading.

    Returns:
        int: Code corresponding to sox_error_t enum. See
        https://fossies.org/dox/sox-14.4.2/sox_8h.html#a8e07e80cebeff3339265d89c387cea93
    """
    global _SOX_INITIALIZED
    if _SOX_INITIALIZED is None:
        raise RuntimeError('SoX effects chain has been already shut down. Can not initialize again.')
    if not _SOX_INITIALIZED:
        code = _torchaudio.initialize_sox()
        if code == _SOX_SUCCESS_CODE:
            _SOX_INITIALIZED = True
            atexit.register(shutdown_sox)
        return code
    return _SOX_SUCCESS_CODE


@_mod_utils.requires_module("torchaudio._torchaudio")
def shutdown_sox() -> int:
    """Showdown sox for effects chain.

    You do not need to call this function as it will be called automatically
    at the end of program execution, if ``initialize_sox`` was called.

    It is safe to call this function multiple times.

    Returns:
        int: Code corresponding to sox_error_t enum. See
        https://fossies.org/dox/sox-14.4.2/sox_8h.html#a8e07e80cebeff3339265d89c387cea93
    """
    global _SOX_INITIALIZED
    if _SOX_INITIALIZED:
        code = _torchaudio.shutdown_sox()
        if code == _SOX_INITIALIZED:
            _SOX_INITIALIZED = None
        return code
    return _SOX_SUCCESS_CODE


def _audio_normalization(signal: Tensor, normalization: Union[bool, float, Callable]) -> None:
    """Audio normalization of a tensor in-place.  The normalization can be a bool,
    a number, or a callable that takes the audio tensor as an input. SoX uses
    32-bit signed integers internally, thus bool normalizes based on that assumption.
    """

    if not normalization:
        return

    if isinstance(normalization, bool):
        normalization = 1 << 31

    if isinstance(normalization, (float, int)):
        # normalize with custom value
        a = normalization
        signal /= a
    elif callable(normalization):
        a = normalization(signal)
        signal /= a
