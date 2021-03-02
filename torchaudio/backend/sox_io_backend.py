import os
from typing import Tuple, Optional

import torch
from torchaudio._internal import (
    module_utils as _mod_utils,
)
from torchaudio.utils.sox_utils import requires_sox

import torchaudio
from .common import AudioMetaData


@requires_sox()
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
        if hasattr(filepath, 'read'):
            sinfo = torchaudio._torchaudio.get_info_fileobj(filepath, format)
            return AudioMetaData(*sinfo)
        filepath = os.fspath(filepath)
    sinfo = torch.ops.torchaudio.sox_io_get_info(filepath, format)
    return AudioMetaData(*sinfo)


@requires_sox()
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

            Note: This argument is intentionally annotated as ``str`` only due to
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
        filepath = os.fspath(filepath)
    return torch.ops.torchaudio.sox_io_load_audio_file(
        filepath, frame_offset, num_frames, normalize, channels_first, format)


@requires_sox()
@_mod_utils.requires_module('torchaudio._torchaudio')
def save(
        filepath: str,
        src: torch.Tensor,
        sample_rate: int,
        channels_first: bool = True,
        compression: Optional[float] = None,
        format: Optional[str] = None,
        encoding: Optional[str] = None,
        bits_per_sample: Optional[int] = None,
):
    """Save audio data to file.

    Args:
        filepath (str or pathlib.Path): Path to save file.
            This function also handles ``pathlib.Path`` objects, but is annotated
            as ``str`` for TorchScript compiler compatibility.
        src (torch.Tensor): Audio data to save. must be 2D tensor.
        sample_rate (int): sampling rate
        channels_first (bool): If ``True``, the given tensor is interpreted as ``[channel, time]``,
            otherwise ``[time, channel]``.
        compression (Optional[float]): Used for formats other than WAV.
            This corresponds to ``-C`` option of ``sox`` command.

            ``"mp3"``
                Either bitrate (in ``kbps``) with quality factor, such as ``128.2``, or
                VBR encoding with quality factor such as ``-4.2``. Default: ``-4.5``.

            ``"flac"``
                Whole number from ``0`` to ``8``. ``8`` is default and highest compression.

            ``"ogg"``, ``"vorbis"``
                Number from ``-1`` to ``10``; ``-1`` is the highest compression
                and lowest quality. Default: ``3``.

            See the detail at http://sox.sourceforge.net/soxformat.html.
        format (str, optional): Override the audio format.
            When ``filepath`` argument is path-like object, audio format is infered from
            file extension. If file extension is missing or different, you can specify the
            correct format with this argument.

            When ``filepath`` argument is file-like object, this argument is required.

            Valid values are ``"wav"``, ``"mp3"``, ``"ogg"``, ``"vorbis"``, ``"amr-nb"``,
            ``"amb"``, ``"flac"``, ``"sph"``, ``"gsm"``, and ``"htk"``.

        encoding (str, optional): Changes the encoding for the supported formats.
            This argument is effective only for supported formats, such as ``"wav"``, ``""amb"``
            and ``"sph"``. Valid values are;

                - ``"PCM_S"`` (signed integer Linear PCM)
                - ``"PCM_U"`` (unsigned integer Linear PCM)
                - ``"PCM_F"`` (floating point PCM)
                - ``"ULAW"`` (mu-law)
                - ``"ALAW"`` (a-law)

            Default values
                If not provided, the default value is picked based on ``format`` and ``bits_per_sample``.

                ``"wav"``, ``"amb"``
                    - | If both ``encoding`` and ``bits_per_sample`` are not provided, the ``dtype`` of the
                      | Tensor is used to determine the default value.
                        - ``"PCM_U"`` if dtype is ``uint8``
                        - ``"PCM_S"`` if dtype is ``int16`` or ``int32`
                        - ``"PCM_F"`` if dtype is ``float32``

                    - ``"PCM_U"`` if ``bits_per_sample=8``
                    - ``"PCM_S"`` otherwise

                ``"sph"`` format;
                    - the default value is ``"PCM_S"``

        bits_per_sample (int, optional): Changes the bit depth for the supported formats.
            When ``format`` is one of ``"wav"``, ``"flac"``, ``"sph"``, or ``"amb"``, you can change the
            bit depth. Valid values are ``8``, ``16``, ``32`` and ``64``.

            Default Value;
                If not provided, the default values are picked based on ``format`` and ``"encoding"``;

                ``"wav"``, ``"amb"``;
                    - | If both ``encoding`` and ``bits_per_sample`` are not provided, the ``dtype`` of the
                      | Tensor is used.
                        - ``8`` if dtype is ``uint8``
                        - ``16`` if dtype is ``int16``
                        - ``32`` if dtype is  ``int32`` or ``float32``

                    - ``8`` if ``encoding`` is ``"PCM_U"``, ``"ULAW"`` or ``"ALAW"``
                    - ``16`` if ``encoding`` is ``"PCM_S"``
                    - ``32`` if ``encoding`` is ``"PCM_F"``

                ``"flac"`` format;
                    - the default value is ``24``

                ``"sph"`` format;
                    - ``16`` if ``encoding`` is ``"PCM_U"``, ``"PCM_S"``, ``"PCM_F"`` or not provided.
                    - ``8`` if ``encoding`` is ``"ULAW"`` or ``"ALAW"``

                ``"amb"`` format;
                    - ``8`` if ``encoding`` is ``"PCM_U"``, ``"ULAW"`` or ``"ALAW"``
                    - ``16`` if ``encoding`` is ``"PCM_S"`` or not provided.
                    - ``32`` if ``encoding`` is ``"PCM_F"``

    Supported formats/encodings/bit depth/compression are;

    ``"wav"``, ``"amb"``
        - 32-bit floating-point PCM
        - 32-bit signed integer PCM
        - 24-bit signed integer PCM
        - 16-bit signed integer PCM
        - 8-bit unsigned integer PCM
        - 8-bit mu-law
        - 8-bit a-law

        Note: Default encoding/bit depth is determined by the dtype of the input Tensor.

    ``"mp3"``
        Fixed bit rate (such as 128kHz) and variable bit rate compression.
        Default: VBR with high quality.

    ``"flac"``
        - 8-bit
        - 16-bit
        - 24-bit (default)

    ``"ogg"``, ``"vorbis"``
        - Different quality level. Default: approx. 112kbps

    ``"sph"``
        - 8-bit signed integer PCM
        - 16-bit signed integer PCM
        - 24-bit signed integer PCM
        - 32-bit signed integer PCM (default)
        - 8-bit mu-law
        - 8-bit a-law
        - 16-bit a-law
        - 24-bit a-law
        - 32-bit a-law

    ``"amr-nb"``
        Bitrate ranging from 4.75 kbit/s to 12.2 kbit/s. Default: 4.75 kbit/s

    ``"gsm"``
        Lossy Speech Compression, CPU intensive.

    ``"htk"``
        Uses a default single-channel 16-bit PCM format.

    Note:
        To save into formats that ``libsox`` does not handle natively, (such as ``"mp3"``,
        ``"flac"``, ``"ogg"`` and ``"vorbis"``), your installation of ``torchaudio`` has
        to be linked to ``libsox`` and corresponding codec libraries such as ``libmad``
        or ``libmp3lame`` etc.
    """
    if not torch.jit.is_scripting():
        if hasattr(filepath, 'write'):
            torchaudio._torchaudio.save_audio_fileobj(
                filepath, src, sample_rate, channels_first, compression,
                format, encoding, bits_per_sample)
            return
        filepath = os.fspath(filepath)
    torch.ops.torchaudio.sox_io_save_audio_file(
        filepath, src, sample_rate, channels_first, compression, format, encoding, bits_per_sample)


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
