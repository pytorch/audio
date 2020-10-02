from typing import Any, Optional

from torchaudio._internal import module_utils as _mod_utils


class AudioMetaData:
    """Data class to be returned by ``torchaudio.info`` function.

    This class is used by :ref:`"sox_io" backend<sox_io_backend>` and
    :ref:`"soundfile" backend with the new interface<soundfile_backend>`.

    :ivar int sample_rate: Sample rate
    :ivar int num_frames: The number of frames
    :ivar int num_channels: The number of channels
    """
    def __init__(self, sample_rate: int, num_frames: int, num_channels: int):
        self.sample_rate = sample_rate
        self.num_frames = num_frames
        self.num_channels = num_channels


@_mod_utils.deprecated('Please migrate to `AudioMetaData`.', '0.9.0')
class SignalInfo:
    """Data class returned ``info`` functions.

    This class is used by :ref:`"sox" backend (deprecated)<sox_backend>` and
    :ref:`"soundfile" backend with the legacy interface<soundfile_legacy_backend>`.

    Warnings:
        This class is used by deprecated backends, and will be removed in ``0.9.0``.
        Please opt-in to the new backend, which uses :py:class:`AudioMetaData`

    See https://fossies.org/dox/sox-14.4.2/structsox__signalinfo__t.html

    :ivar Optional[int] channels: The number of channels
    :ivar Optional[float] rate: Sampleing rate
    :ivar Optional[int] precision: Bit depth
    :ivar Optional[int] length: For :ref:`sox backend<sox_backend>`, the number of samples.
        (frames * channels). For :ref:`soundfile backend<soundfile_backend>`, the number of frames.
    """
    def __init__(self,
                 channels: Optional[int] = None,
                 rate: Optional[float] = None,
                 precision: Optional[int] = None,
                 length: Optional[int] = None) -> None:
        self.channels = channels
        self.rate = rate
        self.precision = precision
        self.length = length


@_mod_utils.deprecated('Please migrate to `AudioMetaData`.', '0.9.0')
class EncodingInfo:
    """Data class returned ``info`` functions.

    This class is used by :ref:`"sox" backend (deprecated)<sox_backend>` and
    :ref:`"soundfile" backend with the legacy interface<soundfile_legacy_backend>`.

    Warnings:
        This class is used by deprecated backends, and will be removed in ``0.9.0``.
        Please opt-in to the new backend, which uses :py:class:`AudioMetaData`

    See https://fossies.org/dox/sox-14.4.2/structsox__encodinginfo__t.html

    :ivar Optional[int] encoding: sox_encoding_t
    :ivar Optional[int] bits_per_sample: bit depth
    :ivar Optional[float] compression: Compression option
    :ivar Any reverse_bytes:
    :ivar Any reverse_nibbles:
    :ivar Any reverse_bits:
    :ivar Optional[bool] opposite_endian:
    """
    def __init__(self,
                 encoding: Any = None,
                 bits_per_sample: Optional[int] = None,
                 compression: Optional[float] = None,
                 reverse_bytes: Any = None,
                 reverse_nibbles: Any = None,
                 reverse_bits: Any = None,
                 opposite_endian: Optional[bool] = None) -> None:
        self.encoding = encoding
        self.bits_per_sample = bits_per_sample
        self.compression = compression
        self.reverse_bytes = reverse_bytes
        self.reverse_nibbles = reverse_nibbles
        self.reverse_bits = reverse_bits
        self.opposite_endian = opposite_endian


_LOAD_DOCSTRING = r"""Loads an audio file from disk into a tensor

Warnings:
    This function is deprecated. Please migrate to the new backend.

Args:
    filepath: Path to audio file

    out: An optional output tensor to use instead of creating one. (Default: ``None``)

    normalization: Optional normalization.
        If boolean `True`, then output is divided by `1 << 31`.
        Assuming the input is signed 32-bit audio, this normalizes to `[-1, 1]`.
        If `float`, then output is divided by that number.
        If `Callable`, then the output is passed as a paramete to the given function,
        then the output is divided by the result. (Default: ``True``)

    channels_first: Set channels first or length first in result. (Default: ``True``)

    num_frames: Number of frames to load.  0 to load everything after the offset.
        (Default: ``0``)

    offset: Number of frames from the start of the file to begin data loading.
        (Default: ``0``)

    signalinfo: A sox_signalinfo_t type, which could be helpful if the
        audio type cannot be automatically determined. (Default: ``None``)

    encodinginfo: A sox_encodinginfo_t type, which could be set if the
        audio type cannot be automatically determined. (Default: ``None``)

    filetype: A filetype or extension to be set if sox cannot determine it
        automatically. (Default: ``None``)

Returns:
    (Tensor, int): An output tensor of size `[C x L]` or `[L x C]` where
        L is the number of audio frames and
        C is the number of channels.
        An integer which is the sample rate of the audio (as listed in the metadata of the file)

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


_LOAD_WAV_DOCSTRING = r""" Loads a wave file.

It assumes that the wav file uses 16 bit per sample that needs normalization by
shifting the input right by 16 bits.

Warnings:
    This function is deprecated. Please migrate to the new backend and use ``load(.., normalize=False)``

Args:
    filepath: Path to audio file

Returns:
    (Tensor, int): An output tensor of size `[C x L]` or `[L x C]` where L is the number
        of audio frames and C is the number of channels. An integer which is the sample rate of the
        audio (as listed in the metadata of the file)
"""

_SAVE_DOCSTRING = r"""Saves a Tensor on file as an audio file

Warnings:
    This function is deprecated. Please migrate to the new backend.

Args:
    filepath: Path to audio file
    src: An input 2D tensor of shape `[C x L]` or `[L x C]` where L is
        the number of audio frames, C is the number of channels
    sample_rate: An integer which is the sample rate of the
        audio (as listed in the metadata of the file)
    precision Bit precision (Default: ``16``)
    channels_first (bool, optional): Set channels first or length first in result. (
        Default: ``True``)
"""


_INFO_DOCSTRING = r"""Gets metadata from an audio file without loading the signal.

Warnings:
    This function is deprecated. Please migrate to the new backend.

Args:
    filepath: Path to audio file

Returns:
    (sox_signalinfo_t, sox_encodinginfo_t): A si (sox_signalinfo_t) signal
        info as a python object. An ei (sox_encodinginfo_t) encoding info

Example
    >>> si, ei = torchaudio.info('foo.wav')
    >>> rate, channels, encoding = si.rate, si.channels, ei.encoding
"""


def _impl_load(func):
    setattr(func, '__doc__', _LOAD_DOCSTRING)
    return func


def _impl_load_wav(func):
    setattr(func, '__doc__', _LOAD_WAV_DOCSTRING)
    return func


def _impl_save(func):
    setattr(func, '__doc__', _SAVE_DOCSTRING)
    return func


def _impl_info(func):
    setattr(func, '__doc__', _INFO_DOCSTRING)
    return func
