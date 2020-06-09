import os.path
from typing import Any, Optional, Tuple

import torch
from torch import Tensor

from torchaudio._internal import (
    module_utils as _mod_utils,
    misc_ops as _misc_ops,
)
from .common import SignalInfo, EncodingInfo

if _mod_utils.is_module_available('torchaudio._torchaudio'):
    from torchaudio import _torchaudio


@_mod_utils.requires_module('torchaudio._torchaudio')
def load(filepath: str,
         out: Optional[Tensor] = None,
         normalization: bool = True,
         channels_first: bool = True,
         num_frames: int = 0,
         offset: int = 0,
         signalinfo: SignalInfo = None,
         encodinginfo: EncodingInfo = None,
         filetype: Optional[str] = None) -> Tuple[Tensor, int]:
    r"""See torchaudio.load"""

    # stringify if `pathlib.Path` (noop if already `str`)
    filepath = str(filepath)
    # check if valid file
    if not os.path.isfile(filepath):
        raise OSError("{} not found or is a directory".format(filepath))

    # initialize output tensor
    if out is not None:
        _misc_ops.check_input(out)
    else:
        out = torch.FloatTensor()

    if num_frames < -1:
        raise ValueError("Expected value for num_samples -1 (entire file) or >=0")
    if offset < 0:
        raise ValueError("Expected positive offset value")

    sample_rate = _torchaudio.read_audio_file(
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
    _misc_ops.normalize_audio(out, normalization)

    return out, sample_rate


@_mod_utils.requires_module('torchaudio._torchaudio')
def save(filepath: str, src: Tensor, sample_rate: int, precision: int = 16, channels_first: bool = True) -> None:
    r"""See torchaudio.save"""

    si = sox_signalinfo_t()
    ch_idx = 0 if channels_first else 1
    si.rate = sample_rate
    si.channels = 1 if src.dim() == 1 else src.size(ch_idx)
    si.length = src.numel()
    si.precision = precision
    return save_encinfo(filepath, src, channels_first, si)


@_mod_utils.requires_module('torchaudio._torchaudio')
def info(filepath: str) -> Tuple[SignalInfo, EncodingInfo]:
    r"""See torchaudio.info"""
    return _torchaudio.get_info(filepath)


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
    _misc_ops.check_input(src)
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
