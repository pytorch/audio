"""The new soundfile backend which will become default in 0.8.0 onward"""
from typing import Tuple, Optional
import warnings

import torch
from torchaudio._internal import module_utils as _mod_utils
from .common import AudioMetaData


if _mod_utils.is_module_available("soundfile"):
    import soundfile


@_mod_utils.requires_module("soundfile")
def info(filepath: str, format: Optional[str] = None) -> AudioMetaData:
    """Get signal information of an audio file.

    Args:
        filepath (str or pathlib.Path): Path to audio file.
            This functionalso handles ``pathlib.Path`` objects, but is annotated as ``str``
            for the consistency with "sox_io" backend, which has a restriction on type annotation
            for TorchScript compiler compatiblity.
        format (str, optional):
            Not used. PySoundFile does not accept format hint.

    Returns:
        AudioMetaData: meta data of the given audio.
    """
    sinfo = soundfile.info(filepath)
    return AudioMetaData(sinfo.samplerate, sinfo.frames, sinfo.channels)


_SUBTYPE2DTYPE = {
    "PCM_S8": "int8",
    "PCM_U8": "uint8",
    "PCM_16": "int16",
    "PCM_32": "int32",
    "FLOAT": "float32",
    "DOUBLE": "float64",
}


@_mod_utils.requires_module("soundfile")
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
        The formats this function can handle depend on the soundfile installation.
        This function is tested on the following formats;

        * WAV

            * 32-bit floating-point
            * 32-bit signed integer
            * 16-bit signed integer
            * 8-bit unsigned integer

        * FLAC
        * OGG/VORBIS
        * SPHERE

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
            Source of audio data.
            Note:
                  * This argument is intentionally annotated as ``str`` only,
                    for the consistency with "sox_io" backend, which has a restriction
                    on type annotation due to TorchScript compiler compatiblity.
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
            Not used. PySoundFile does not accept format hint.

    Returns:
        Tuple[torch.Tensor, int]: Resulting Tensor and sample rate.
            If the input file has integer wav format and normalization is off, then it has
            integer type, else ``float32`` type. If ``channels_first=True``, it has
            ``[channel, time]`` else ``[time, channel]``.
    """
    with soundfile.SoundFile(filepath, "r") as file_:
        if file_.format != "WAV" or normalize:
            dtype = "float32"
        elif file_.subtype not in _SUBTYPE2DTYPE:
            raise ValueError(f"Unsupported subtype: {file_.subtype}")
        else:
            dtype = _SUBTYPE2DTYPE[file_.subtype]

        frames = file_._prepare_read(frame_offset, None, num_frames)
        waveform = file_.read(frames, dtype, always_2d=True)
        sample_rate = file_.samplerate

    waveform = torch.from_numpy(waveform)
    if channels_first:
        waveform = waveform.t()
    return waveform, sample_rate


@_mod_utils.requires_module("soundfile")
def save(
    filepath: str,
    src: torch.Tensor,
    sample_rate: int,
    channels_first: bool = True,
    compression: Optional[float] = None,
    format: Optional[str] = None,
):
    """Save audio data to file.

    Note:
        The formats this function can handle depend on the soundfile installation.
        This function is tested on the following formats;

        * WAV

            * 32-bit floating-point
            * 32-bit signed integer
            * 16-bit signed integer
            * 8-bit unsigned integer

        * FLAC
        * OGG/VORBIS
        * SPHERE

    Args:
        filepath (str or pathlib.Path): Path to audio file.
            This functionalso handles ``pathlib.Path`` objects, but is annotated as ``str``
            for the consistency with "sox_io" backend, which has a restriction on type annotation
            for TorchScript compiler compatiblity.
        tensor (torch.Tensor): Audio data to save. must be 2D tensor.
        sample_rate (int): sampling rate
        channels_first (bool):
            If ``True``, the given tensor is interpreted as ``[channel, time]``,
            otherwise ``[time, channel]``.
        compression (Optional[float]):
            Not used. It is here only for interface compatibility reson with "sox_io" backend.
        format (str, optional):
            Output audio format. This is required when the output audio format cannot be infered from
            ``filepath``, (such as file extension or ``name`` attribute of the given file object).
    """
    if src.ndim != 2:
        raise ValueError(f"Expected 2D Tensor, got {src.ndim}D.")
    if compression is not None:
        warnings.warn(
            '`save` function of "soundfile" backend does not support "compression" parameter. '
            "The argument is silently ignored."
        )
    if hasattr(filepath, 'write'):
        if format is None:
            raise RuntimeError('`format` is required when saving to file object.')
        ext = format
    else:
        ext = str(filepath).split(".")[-1].lower()

    if ext != "wav":
        subtype = None
    elif src.dtype == torch.uint8:
        subtype = "PCM_U8"
    elif src.dtype == torch.int16:
        subtype = "PCM_16"
    elif src.dtype == torch.int32:
        subtype = "PCM_32"
    elif src.dtype == torch.float32:
        subtype = "FLOAT"
    elif src.dtype == torch.float64:
        subtype = "DOUBLE"
    else:
        raise ValueError(f"Unsupported dtype for WAV: {src.dtype}")

    # sph is a extension used in TED-LIUM but soundfile does not recognize it as NIST format,
    # so we extend the extensions manually here
    if ext in ["nis", "nist", "sph"] and format is None:
        format = "NIST"

    if channels_first:
        src = src.t()

    soundfile.write(
        file=filepath, data=src, samplerate=sample_rate, subtype=subtype, format=format
    )


@_mod_utils.requires_module("soundfile")
@_mod_utils.deprecated('Please use "torchaudio.load".', "0.9.0")
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
    return load(
        filepath,
        frame_offset,
        num_frames,
        normalize=False,
        channels_first=channels_first,
    )
