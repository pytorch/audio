"""The new soundfile backend which will become default in 0.8.0 onward"""
import warnings
from typing import Optional, Tuple

import torch
from torchaudio._internal import module_utils as _mod_utils

from .common import AudioMetaData


_IS_SOUNDFILE_AVAILABLE = False

# TODO: import soundfile only when it is used.
if _mod_utils.is_module_available("soundfile"):
    try:
        import soundfile

        _requires_soundfile = _mod_utils.no_op
        _IS_SOUNDFILE_AVAILABLE = True
    except Exception:
        _requires_soundfile = _mod_utils.fail_with_message(
            "requires soundfile, but we failed to import it. Please check the installation of soundfile."
        )
else:
    _requires_soundfile = _mod_utils.fail_with_message(
        "requires soundfile, but it is not installed. Please install soundfile."
    )


# Mapping from soundfile subtype to number of bits per sample.
# This is mostly heuristical and the value is set to 0 when it is irrelevant
# (lossy formats) or when it can't be inferred.
# For ADPCM (and G72X) subtypes, it's hard to infer the bit depth because it's not part of the standard:
# According to https://en.wikipedia.org/wiki/Adaptive_differential_pulse-code_modulation#In_telephony,
# the default seems to be 8 bits but it can be compressed further to 4 bits.
# The dict is inspired from
# https://github.com/bastibe/python-soundfile/blob/744efb4b01abc72498a96b09115b42a4cabd85e4/soundfile.py#L66-L94
_SUBTYPE_TO_BITS_PER_SAMPLE = {
    "PCM_S8": 8,  # Signed 8 bit data
    "PCM_16": 16,  # Signed 16 bit data
    "PCM_24": 24,  # Signed 24 bit data
    "PCM_32": 32,  # Signed 32 bit data
    "PCM_U8": 8,  # Unsigned 8 bit data (WAV and RAW only)
    "FLOAT": 32,  # 32 bit float data
    "DOUBLE": 64,  # 64 bit float data
    "ULAW": 8,  # U-Law encoded. See https://en.wikipedia.org/wiki/G.711#Types
    "ALAW": 8,  # A-Law encoded. See https://en.wikipedia.org/wiki/G.711#Types
    "IMA_ADPCM": 0,  # IMA ADPCM.
    "MS_ADPCM": 0,  # Microsoft ADPCM.
    "GSM610": 0,  # GSM 6.10 encoding. (Wikipedia says 1.625 bit depth?? https://en.wikipedia.org/wiki/Full_Rate)
    "VOX_ADPCM": 0,  # OKI / Dialogix ADPCM
    "G721_32": 0,  # 32kbs G721 ADPCM encoding.
    "G723_24": 0,  # 24kbs G723 ADPCM encoding.
    "G723_40": 0,  # 40kbs G723 ADPCM encoding.
    "DWVW_12": 12,  # 12 bit Delta Width Variable Word encoding.
    "DWVW_16": 16,  # 16 bit Delta Width Variable Word encoding.
    "DWVW_24": 24,  # 24 bit Delta Width Variable Word encoding.
    "DWVW_N": 0,  # N bit Delta Width Variable Word encoding.
    "DPCM_8": 8,  # 8 bit differential PCM (XI only)
    "DPCM_16": 16,  # 16 bit differential PCM (XI only)
    "VORBIS": 0,  # Xiph Vorbis encoding. (lossy)
    "ALAC_16": 16,  # Apple Lossless Audio Codec (16 bit).
    "ALAC_20": 20,  # Apple Lossless Audio Codec (20 bit).
    "ALAC_24": 24,  # Apple Lossless Audio Codec (24 bit).
    "ALAC_32": 32,  # Apple Lossless Audio Codec (32 bit).
}


def _get_bit_depth(subtype):
    if subtype not in _SUBTYPE_TO_BITS_PER_SAMPLE:
        warnings.warn(
            f"The {subtype} subtype is unknown to TorchAudio. As a result, the bits_per_sample "
            "attribute will be set to 0. If you are seeing this warning, please "
            "report by opening an issue on github (after checking for existing/closed ones). "
            "You may otherwise ignore this warning."
        )
    return _SUBTYPE_TO_BITS_PER_SAMPLE.get(subtype, 0)


_SUBTYPE_TO_ENCODING = {
    "PCM_S8": "PCM_S",
    "PCM_16": "PCM_S",
    "PCM_24": "PCM_S",
    "PCM_32": "PCM_S",
    "PCM_U8": "PCM_U",
    "FLOAT": "PCM_F",
    "DOUBLE": "PCM_F",
    "ULAW": "ULAW",
    "ALAW": "ALAW",
    "VORBIS": "VORBIS",
}


def _get_encoding(format: str, subtype: str):
    if format == "FLAC":
        return "FLAC"
    return _SUBTYPE_TO_ENCODING.get(subtype, "UNKNOWN")


@_requires_soundfile
def info(filepath: str, format: Optional[str] = None) -> AudioMetaData:
    """Get signal information of an audio file.

    Note:
        ``filepath`` argument is intentionally annotated as ``str`` only, even though it accepts
        ``pathlib.Path`` object as well. This is for the consistency with ``"sox_io"`` backend,
        which has a restriction on type annotation due to TorchScript compiler compatiblity.

    Args:
        filepath (path-like object or file-like object):
            Source of audio data.
        format (str or None, optional):
            Not used. PySoundFile does not accept format hint.

    Returns:
        AudioMetaData: meta data of the given audio.

    """
    sinfo = soundfile.info(filepath)
    return AudioMetaData(
        sinfo.samplerate,
        sinfo.frames,
        sinfo.channels,
        bits_per_sample=_get_bit_depth(sinfo.subtype),
        encoding=_get_encoding(sinfo.format, sinfo.subtype),
    )


_SUBTYPE2DTYPE = {
    "PCM_S8": "int8",
    "PCM_U8": "uint8",
    "PCM_16": "int16",
    "PCM_32": "int32",
    "FLOAT": "float32",
    "DOUBLE": "float64",
}


@_requires_soundfile
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

    Note:
        ``filepath`` argument is intentionally annotated as ``str`` only, even though it accepts
        ``pathlib.Path`` object as well. This is for the consistency with ``"sox_io"`` backend,
        which has a restriction on type annotation due to TorchScript compiler compatiblity.

    Args:
        filepath (path-like object or file-like object):
            Source of audio data.
        frame_offset (int, optional):
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
            Not used. PySoundFile does not accept format hint.

    Returns:
        (torch.Tensor, int): Resulting Tensor and sample rate.
            If the input file has integer wav format and normalization is off, then it has
            integer type, else ``float32`` type. If ``channels_first=True``, it has
            `[channel, time]` else `[time, channel]`.
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


def _get_subtype_for_wav(dtype: torch.dtype, encoding: str, bits_per_sample: int):
    if not encoding:
        if not bits_per_sample:
            subtype = {
                torch.uint8: "PCM_U8",
                torch.int16: "PCM_16",
                torch.int32: "PCM_32",
                torch.float32: "FLOAT",
                torch.float64: "DOUBLE",
            }.get(dtype)
            if not subtype:
                raise ValueError(f"Unsupported dtype for wav: {dtype}")
            return subtype
        if bits_per_sample == 8:
            return "PCM_U8"
        return f"PCM_{bits_per_sample}"
    if encoding == "PCM_S":
        if not bits_per_sample:
            return "PCM_32"
        if bits_per_sample == 8:
            raise ValueError("wav does not support 8-bit signed PCM encoding.")
        return f"PCM_{bits_per_sample}"
    if encoding == "PCM_U":
        if bits_per_sample in (None, 8):
            return "PCM_U8"
        raise ValueError("wav only supports 8-bit unsigned PCM encoding.")
    if encoding == "PCM_F":
        if bits_per_sample in (None, 32):
            return "FLOAT"
        if bits_per_sample == 64:
            return "DOUBLE"
        raise ValueError("wav only supports 32/64-bit float PCM encoding.")
    if encoding == "ULAW":
        if bits_per_sample in (None, 8):
            return "ULAW"
        raise ValueError("wav only supports 8-bit mu-law encoding.")
    if encoding == "ALAW":
        if bits_per_sample in (None, 8):
            return "ALAW"
        raise ValueError("wav only supports 8-bit a-law encoding.")
    raise ValueError(f"wav does not support {encoding}.")


def _get_subtype_for_sphere(encoding: str, bits_per_sample: int):
    if encoding in (None, "PCM_S"):
        return f"PCM_{bits_per_sample}" if bits_per_sample else "PCM_32"
    if encoding in ("PCM_U", "PCM_F"):
        raise ValueError(f"sph does not support {encoding} encoding.")
    if encoding == "ULAW":
        if bits_per_sample in (None, 8):
            return "ULAW"
        raise ValueError("sph only supports 8-bit for mu-law encoding.")
    if encoding == "ALAW":
        return "ALAW"
    raise ValueError(f"sph does not support {encoding}.")


def _get_subtype(dtype: torch.dtype, format: str, encoding: str, bits_per_sample: int):
    if format == "wav":
        return _get_subtype_for_wav(dtype, encoding, bits_per_sample)
    if format == "flac":
        if encoding:
            raise ValueError("flac does not support encoding.")
        if not bits_per_sample:
            return "PCM_16"
        if bits_per_sample > 24:
            raise ValueError("flac does not support bits_per_sample > 24.")
        return "PCM_S8" if bits_per_sample == 8 else f"PCM_{bits_per_sample}"
    if format in ("ogg", "vorbis"):
        if bits_per_sample:
            raise ValueError("ogg/vorbis does not support bits_per_sample.")
        if encoding is None or encoding == "vorbis":
            return "VORBIS"
        if encoding == "opus":
            return "OPUS"
        raise ValueError(f"Unexpected encoding: {encoding}")
    if format == "mp3":
        return "MPEG_LAYER_III"
    if format == "sph":
        return _get_subtype_for_sphere(encoding, bits_per_sample)
    if format in ("nis", "nist"):
        return "PCM_16"
    raise ValueError(f"Unsupported format: {format}")


@_requires_soundfile
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

    Note:
        ``filepath`` argument is intentionally annotated as ``str`` only, even though it accepts
        ``pathlib.Path`` object as well. This is for the consistency with ``"sox_io"`` backend,
        which has a restriction on type annotation due to TorchScript compiler compatiblity.

    Args:
        filepath (str or pathlib.Path): Path to audio file.
        src (torch.Tensor): Audio data to save. must be 2D tensor.
        sample_rate (int): sampling rate
        channels_first (bool, optional): If ``True``, the given tensor is interpreted as `[channel, time]`,
            otherwise `[time, channel]`.
        compression (float of None, optional): Not used.
            It is here only for interface compatibility reson with "sox_io" backend.
        format (str or None, optional): Override the audio format.
            When ``filepath`` argument is path-like object, audio format is
            inferred from file extension. If the file extension is missing or
            different, you can specify the correct format with this argument.

            When ``filepath`` argument is file-like object,
            this argument is required.

            Valid values are ``"wav"``, ``"ogg"``, ``"vorbis"``,
            ``"flac"`` and ``"sph"``.
        encoding (str or None, optional): Changes the encoding for supported formats.
            This argument is effective only for supported formats, sush as
            ``"wav"``, ``""flac"`` and ``"sph"``. Valid values are;

                - ``"PCM_S"`` (signed integer Linear PCM)
                - ``"PCM_U"`` (unsigned integer Linear PCM)
                - ``"PCM_F"`` (floating point PCM)
                - ``"ULAW"`` (mu-law)
                - ``"ALAW"`` (a-law)

        bits_per_sample (int or None, optional): Changes the bit depth for the
            supported formats.
            When ``format`` is one of ``"wav"``, ``"flac"`` or ``"sph"``,
            you can change the bit depth.
            Valid values are ``8``, ``16``, ``24``, ``32`` and ``64``.

    Supported formats/encodings/bit depth/compression are:

    ``"wav"``
        - 32-bit floating-point PCM
        - 32-bit signed integer PCM
        - 24-bit signed integer PCM
        - 16-bit signed integer PCM
        - 8-bit unsigned integer PCM
        - 8-bit mu-law
        - 8-bit a-law

        Note:
            Default encoding/bit depth is determined by the dtype of
            the input Tensor.

    ``"flac"``
        - 8-bit
        - 16-bit (default)
        - 24-bit

    ``"ogg"``, ``"vorbis"``
        - Doesn't accept changing configuration.

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

    """
    if src.ndim != 2:
        raise ValueError(f"Expected 2D Tensor, got {src.ndim}D.")
    if compression is not None:
        warnings.warn(
            '`save` function of "soundfile" backend does not support "compression" parameter. '
            "The argument is silently ignored."
        )
    if hasattr(filepath, "write"):
        if format is None:
            raise RuntimeError("`format` is required when saving to file object.")
        ext = format.lower()
    else:
        ext = str(filepath).split(".")[-1].lower()

    if bits_per_sample not in (None, 8, 16, 24, 32, 64):
        raise ValueError("Invalid bits_per_sample.")
    if bits_per_sample == 24:
        warnings.warn(
            "Saving audio with 24 bits per sample might warp samples near -1. "
            "Using 16 bits per sample might be able to avoid this."
        )
    subtype = _get_subtype(src.dtype, ext, encoding, bits_per_sample)

    # sph is a extension used in TED-LIUM but soundfile does not recognize it as NIST format,
    # so we extend the extensions manually here
    if ext in ["nis", "nist", "sph"] and format is None:
        format = "NIST"

    if channels_first:
        src = src.t()

    soundfile.write(file=filepath, data=src, samplerate=sample_rate, subtype=subtype, format=format)
