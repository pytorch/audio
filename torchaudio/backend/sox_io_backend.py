import os
import warnings
from typing import Tuple, Optional

import torch
from torchaudio._internal import (
    module_utils as _mod_utils,
)

import torchaudio
from .common import AudioMetaData


@torch.jit.unused
def _info(
        filepath: str,
        format: Optional[str] = None,
) -> AudioMetaData:
    if hasattr(filepath, 'read'):
        sinfo = torchaudio._torchaudio.get_info_fileobj(filepath, format)
        return AudioMetaData(*sinfo)
    sinfo = torch.ops.torchaudio.sox_io_get_info(os.fspath(filepath), format)
    return AudioMetaData(
        sinfo.get_sample_rate(),
        sinfo.get_num_frames(),
        sinfo.get_num_channels(),
        sinfo.get_bits_per_sample(),
        sinfo.get_encoding(),
    )


@_mod_utils.requires_module('torchaudio._torchaudio')
def info(
        filepath: str,
        format: Optional[str] = None,
) -> AudioMetaData:
    """Get signal information of an audio file.

    Args:
        filepath (path-like object or file-like object):
            Source of audio data. When the function is not compiled by TorchScript,
            (e.g. ``torch.jit.script``), the following types are accepted;
                  * ``path-like``: file path
                  * ``file-like``: Object with ``read(size: int) -> bytes`` method,
                    which returns byte string of at most ``size`` length.
            When the function is compiled by TorchScript, only ``str`` type is allowed.

            Note:
                  * When the input type is file-like object, this function cannot
                    get the correct length (``num_samples``) for certain formats,
                    such as ``mp3`` and ``vorbis``.
                    In this case, the value of ``num_samples`` is ``0``.
                  * This argument is intentionally annotated as ``str`` only due to
                    TorchScript compiler compatibility.
        format (str, optional):
            Override the format detection with the given format.
            Providing the argument might help when libsox can not infer the format
            from header or extension,

    Returns:
        AudioMetaData: Metadata of the given audio.
    """
    if not torch.jit.is_scripting():
        return _info(filepath, format)
    sinfo = torch.ops.torchaudio.sox_io_get_info(filepath, format)
    return AudioMetaData(
        sinfo.get_sample_rate(),
        sinfo.get_num_frames(),
        sinfo.get_num_channels(),
        sinfo.get_bits_per_sample(),
        sinfo.get_encoding())


@_mod_utils.requires_module('torchaudio._torchaudio')
def load(
        filepath: str,
        frame_offset: int = 0,
        num_frames: int = -1,
        normalize: bool = True,
        channels_first: bool = True,
        format: Optional[str] = None,
) -> Tuple[torch.Tensor, int]:
    """Load audio data from file.

    Note:
        This function can handle all the codecs that underlying libsox can handle,
        however it is tested on the following formats;

        * WAV, AMB

            * 32-bit floating-point
            * 32-bit signed integer
            * 16-bit signed integer
            * 8-bit unsigned integer (WAV only)

        * MP3
        * FLAC
        * OGG/VORBIS
        * OPUS
        * SPHERE
        * AMR-NB

        To load ``MP3``, ``FLAC``, ``OGG/VORBIS``, ``OPUS`` and other codecs ``libsox`` does not
        handle natively, your installation of ``torchaudio`` has to be linked to ``libsox``
        and corresponding codec libraries such as ``libmad`` or ``libmp3lame`` etc.

    By default (``normalize=True``, ``channels_first=True``), this function returns Tensor with
    ``float32`` dtype and the shape of ``[channel, time]``.
    The samples are normalized to fit in the range of ``[-1.0, 1.0]``.

    When the input format is WAV with integer type, such as 32-bit signed integer, 16-bit
    signed integer and 8-bit unsigned integer (24-bit signed integer is not supported),
    by providing ``normalize=False``, this function can return integer Tensor, where the samples
    are expressed within the whole range of the corresponding dtype, that is, ``int32`` tensor
    for 32-bit signed PCM, ``int16`` for 16-bit signed PCM and ``uint8`` for 8-bit unsigned PCM.

    ``normalize`` parameter has no effect on 32-bit floating-point WAV and other formats, such as
    ``flac`` and ``mp3``.
    For these formats, this function always returns ``float32`` Tensor with values normalized to
    ``[-1.0, 1.0]``.

    Args:
        filepath (path-like object or file-like object):
            Source of audio data. When the function is not compiled by TorchScript,
            (e.g. ``torch.jit.script``), the following types are accepted;
                  * ``path-like``: file path
                  * ``file-like``: Object with ``read(size: int) -> bytes`` method,
                    which returns byte string of at most ``size`` length.
            When the function is compiled by TorchScript, only ``str`` type is allowed.

            Note:
                * This argument is intentionally annotated as ``str`` only due to
                  TorchScript compiler compatibility.
        frame_offset (int):
            Number of frames to skip before start reading data.
        num_frames (int):
            Maximum number of frames to read. ``-1`` reads all the remaining samples,
            starting from ``frame_offset``.
            This function may return the less number of frames if there is not enough
            frames in the given file.
        normalize (bool):
            When ``True``, this function always return ``float32``, and sample values are
            normalized to ``[-1.0, 1.0]``.
            If input file is integer WAV, giving ``False`` will change the resulting Tensor type to
            integer type.
            This argument has no effect for formats other than integer WAV type.
        channels_first (bool):
            When True, the returned Tensor has dimension ``[channel, time]``.
            Otherwise, the returned Tensor's dimension is ``[time, channel]``.
        format (str, optional):
            Override the format detection with the given format.
            Providing the argument might help when libsox can not infer the format
            from header or extension,

    Returns:
        Tuple[torch.Tensor, int]: Resulting Tensor and sample rate.
            If the input file has integer wav format and normalization is off, then it has
            integer type, else ``float32`` type. If ``channels_first=True``, it has
            ``[channel, time]`` else ``[time, channel]``.
    """
    if not torch.jit.is_scripting():
        if hasattr(filepath, 'read'):
            return torchaudio._torchaudio.load_audio_fileobj(
                filepath, frame_offset, num_frames, normalize, channels_first, format)
        signal = torch.ops.torchaudio.sox_io_load_audio_file(
            os.fspath(filepath), frame_offset, num_frames, normalize, channels_first, format)
        return signal
    signal = torch.ops.torchaudio.sox_io_load_audio_file(
        filepath, frame_offset, num_frames, normalize, channels_first, format)
    return signal


@torch.jit.unused
def _save(
        filepath: str,
        src: torch.Tensor,
        sample_rate: int,
        channels_first: bool = True,
        compression: Optional[float] = None,
        format: Optional[str] = None,
        dtype: Optional[str] = None,
):
    if hasattr(filepath, 'write'):
        if format is None:
            raise RuntimeError('`format` is required when saving to file object.')
        torchaudio._torchaudio.save_audio_fileobj(
            filepath, src, sample_rate, channels_first, compression, format, dtype)
    else:
        torch.ops.torchaudio.sox_io_save_audio_file(
            os.fspath(filepath), src, sample_rate, channels_first, compression, format, dtype)


@_mod_utils.requires_module('torchaudio._torchaudio')
def save(
        filepath: str,
        src: torch.Tensor,
        sample_rate: int,
        channels_first: bool = True,
        compression: Optional[float] = None,
        format: Optional[str] = None,
        dtype: Optional[str] = None,
):
    """Save audio data to file.

    Note:
        Supported formats are;

        * WAV, AMB

            * 32-bit floating-point
            * 32-bit signed integer
            * 16-bit signed integer
            * 8-bit unsigned integer

        * MP3
        * FLAC
        * OGG/VORBIS
        * SPHERE
        * AMR-NB

        To save ``MP3``, ``FLAC``, ``OGG/VORBIS``, and other codecs ``libsox`` does not
        handle natively, your installation of ``torchaudio`` has to be linked to ``libsox``
        and corresponding codec libraries such as ``libmad`` or ``libmp3lame`` etc.

    Args:
        filepath (str or pathlib.Path):
            Path to save file. This function also handles ``pathlib.Path`` objects, but is annotated
            as ``str`` for TorchScript compiler compatibility.
        tensor (torch.Tensor): Audio data to save. must be 2D tensor.
        sample_rate (int): sampling rate
        channels_first (bool):
            If ``True``, the given tensor is interpreted as ``[channel, time]``,
            otherwise ``[time, channel]``.
        compression (Optional[float]):
            Used for formats other than WAV. This corresponds to ``-C`` option of ``sox`` command.

                * | ``MP3``: Either bitrate (in ``kbps``) with quality factor, such as ``128.2``, or
                  | VBR encoding with quality factor such as ``-4.2``. Default: ``-4.5``.
                * | ``FLAC``: compression level. Whole number from ``0`` to ``8``.
                  | ``8`` is default and highest compression.
                * | ``OGG/VORBIS``: number from ``-1`` to ``10``; ``-1`` is the highest compression
                  | and lowest quality. Default: ``3``.

            See the detail at http://sox.sourceforge.net/soxformat.html.
        format (str, optional):
            Output audio format. This is required when the output audio format cannot be infered from
            ``filepath``, (such as file extension or ``name`` attribute of the given file object).
        dtype (str, optional)
            Output tensor dtype.
            Valid values: ``"uint8", "int16", "int32", "float32", "float64", None``
            ``dtype=None`` means no conversion is performed.
            ``dtype`` parameter is only effective for ``float32`` Tensor.
    """
    if src.dtype == torch.float32 and dtype is None:
        warnings.warn(
            '`dtype` default value will be changed to `int16` in 0.9 release.'
            'Specify `dtype` to suppress this warning.'
        )
    if not torch.jit.is_scripting():
        _save(filepath, src, sample_rate, channels_first, compression, format, dtype)
        return
    torch.ops.torchaudio.sox_io_save_audio_file(
        filepath, src, sample_rate, channels_first, compression, format, dtype)


@_mod_utils.requires_module('torchaudio._torchaudio')
@_mod_utils.deprecated('Please use "torchaudio.load".', '0.9.0')
def load_wav(
        filepath: str,
        frame_offset: int = 0,
        num_frames: int = -1,
        channels_first: bool = True,
) -> Tuple[torch.Tensor, int]:
    """Load wave file.

    This function is defined only for the purpose of compatibility against other backend
    for simple usecases, such as ``torchaudio.load_wav(filepath)``.
    The implementation is same as :py:func:`load`.
    """
    return load(filepath, frame_offset, num_frames, normalize=False, channels_first=channels_first)
