import os
from typing import Optional, Tuple

import torch
import torchaudio
from torchaudio import AudioMetaData

sox_ext = torchaudio._extension.lazy_import_sox_ext()


def info(
    filepath: str,
    format: Optional[str] = None,
) -> AudioMetaData:
    """Get signal information of an audio file.

    Args:
        filepath (str):
            Source of audio data.

        format (str or None, optional):
            Override the format detection with the given format.
            Providing the argument might help when libsox can not infer the format
            from header or extension.

    Returns:
        AudioMetaData: Metadata of the given audio.
    """
    if not torch.jit.is_scripting():
        if hasattr(filepath, "read"):
            raise RuntimeError("sox_io backend does not support file-like object.")
        filepath = os.fspath(filepath)
    sinfo = sox_ext.get_info(filepath, format)
    return AudioMetaData(*sinfo)


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
            * 24-bit signed integer
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
    ``float32`` dtype, and the shape of `[channel, time]`.

    .. warning::

       ``normalize`` argument does not perform volume normalization.
       It only converts the sample type to `torch.float32` from the native sample
       type.

       When the input format is WAV with integer type, such as 32-bit signed integer, 16-bit
       signed integer, 24-bit signed integer, and 8-bit unsigned integer, by providing ``normalize=False``,
       this function can return integer Tensor, where the samples are expressed within the whole range
       of the corresponding dtype, that is, ``int32`` tensor for 32-bit signed PCM,
       ``int16`` for 16-bit signed PCM and ``uint8`` for 8-bit unsigned PCM. Since torch does not
       support ``int24`` dtype, 24-bit signed PCM are converted to ``int32`` tensors.

       ``normalize`` argument has no effect on 32-bit floating-point WAV and other formats, such as
       ``flac`` and ``mp3``.

       For these formats, this function always returns ``float32`` Tensor with values.

    Args:
        filepath (path-like object): Source of audio data.
        frame_offset (int):
            Number of frames to skip before start reading data.
        num_frames (int, optional):
            Maximum number of frames to read. ``-1`` reads all the remaining samples,
            starting from ``frame_offset``.
            This function may return the less number of frames if there is not enough
            frames in the given file.
        normalize (bool, optional):
            When ``True``, this function converts the native sample type to ``float32``.
            Default: ``True``.

            If input file is integer WAV, giving ``False`` will change the resulting Tensor type to
            integer type.
            This argument has no effect for formats other than integer WAV type.

        channels_first (bool, optional):
            When True, the returned Tensor has dimension `[channel, time]`.
            Otherwise, the returned Tensor's dimension is `[time, channel]`.
        format (str or None, optional):
            Override the format detection with the given format.
            Providing the argument might help when libsox can not infer the format
            from header or extension.

    Returns:
        (torch.Tensor, int): Resulting Tensor and sample rate.
            If the input file has integer wav format and ``normalize=False``, then it has
            integer type, else ``float32`` type. If ``channels_first=True``, it has
            `[channel, time]` else `[time, channel]`.
    """
    if not torch.jit.is_scripting():
        if hasattr(filepath, "read"):
            raise RuntimeError("sox_io backend does not support file-like object.")
        filepath = os.fspath(filepath)
    return sox_ext.load_audio_file(filepath, frame_offset, num_frames, normalize, channels_first, format)


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
        filepath (path-like object): Path to save file.
        src (torch.Tensor): Audio data to save. must be 2D tensor.
        sample_rate (int): sampling rate
        channels_first (bool, optional): If ``True``, the given tensor is interpreted as `[channel, time]`,
            otherwise `[time, channel]`.
        compression (float or None, optional): Used for formats other than WAV.
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
        format (str or None, optional): Override the audio format.
            When ``filepath`` argument is path-like object, audio format is infered from
            file extension. If file extension is missing or different, you can specify the
            correct format with this argument.

            When ``filepath`` argument is file-like object, this argument is required.

            Valid values are ``"wav"``, ``"mp3"``, ``"ogg"``, ``"vorbis"``, ``"amr-nb"``,
            ``"amb"``, ``"flac"``, ``"sph"``, ``"gsm"``, and ``"htk"``.

        encoding (str or None, optional): Changes the encoding for the supported formats.
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
                        - ``"PCM_S"`` if dtype is ``int16`` or ``int32``
                        - ``"PCM_F"`` if dtype is ``float32``

                    - ``"PCM_U"`` if ``bits_per_sample=8``
                    - ``"PCM_S"`` otherwise

                ``"sph"`` format;
                    - the default value is ``"PCM_S"``

        bits_per_sample (int or None, optional): Changes the bit depth for the supported formats.
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
        if hasattr(filepath, "write"):
            raise RuntimeError("sox_io backend does not handle file-like object.")
        filepath = os.fspath(filepath)
    sox_ext.save_audio_file(
        filepath,
        src,
        sample_rate,
        channels_first,
        compression,
        format,
        encoding,
        bits_per_sample,
    )
