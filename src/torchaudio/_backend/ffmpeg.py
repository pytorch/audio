import os
import re
import sys
from typing import BinaryIO, Optional, Tuple, Union

import torch
import torchaudio

from .backend import Backend
from .common import AudioMetaData

InputType = Union[BinaryIO, str, os.PathLike]


def info_audio(
    src: InputType,
    format: Optional[str],
    buffer_size: int = 4096,
) -> AudioMetaData:
    s = torchaudio.io.StreamReader(src, format, None, buffer_size)
    sinfo = s.get_src_stream_info(s.default_audio_stream)
    if sinfo.num_frames == 0:
        waveform = _load_audio(s)
        num_frames = waveform.size(1)
    else:
        num_frames = sinfo.num_frames
    return AudioMetaData(
        int(sinfo.sample_rate),
        num_frames,
        sinfo.num_channels,
        sinfo.bits_per_sample,
        sinfo.codec.upper(),
    )


def _get_load_filter(
    frame_offset: int = 0,
    num_frames: int = -1,
    convert: bool = True,
) -> Optional[str]:
    if frame_offset < 0:
        raise RuntimeError("Invalid argument: frame_offset must be non-negative. Found: {}".format(frame_offset))
    if num_frames == 0 or num_frames < -1:
        raise RuntimeError("Invalid argument: num_frames must be -1 or greater than 0. Found: {}".format(num_frames))

    # All default values -> no filter
    if frame_offset == 0 and num_frames == -1 and not convert:
        return None
    # Only convert
    aformat = "aformat=sample_fmts=fltp"
    if frame_offset == 0 and num_frames == -1 and convert:
        return aformat
    # At least one of frame_offset or num_frames has non-default value
    if num_frames > 0:
        atrim = "atrim=start_sample={}:end_sample={}".format(frame_offset, frame_offset + num_frames)
    else:
        atrim = "atrim=start_sample={}".format(frame_offset)
    if not convert:
        return atrim
    return "{},{}".format(atrim, aformat)


def _load_audio(
    s: "torchaudio.io.StreamReader",
    filter: Optional[str] = None,
    channels_first: bool = True,
) -> torch.Tensor:
    s.add_audio_stream(-1, -1, filter_desc=filter)
    s.process_all_packets()
    chunk = s.pop_chunks()[0]
    if chunk is None:
        raise RuntimeError("Failed to decode audio.")
    waveform = chunk._elem
    return waveform.T if channels_first else waveform


def load_audio(
    src: InputType,
    frame_offset: int = 0,
    num_frames: int = -1,
    convert: bool = True,
    channels_first: bool = True,
    format: Optional[str] = None,
    buffer_size: int = 4096,
) -> Tuple[torch.Tensor, int]:
    if hasattr(src, "read") and format == "vorbis":
        format = "ogg"
    s = torchaudio.io.StreamReader(src, format, None, buffer_size)
    sample_rate = int(s.get_src_stream_info(s.default_audio_stream).sample_rate)
    filter = _get_load_filter(frame_offset, num_frames, convert)
    waveform = _load_audio(s, filter, channels_first)
    return waveform, sample_rate


def _get_sample_format(dtype: torch.dtype) -> str:
    dtype_to_format = {
        torch.uint8: "u8",
        torch.int16: "s16",
        torch.int32: "s32",
        torch.int64: "s64",
        torch.float32: "flt",
        torch.float64: "dbl",
    }
    format = dtype_to_format.get(dtype)
    if format is None:
        raise ValueError(f"No format found for dtype {dtype}; dtype must be one of {list(dtype_to_format.keys())}.")
    return format


def _native_endianness() -> str:
    if sys.byteorder == "little":
        return "le"
    else:
        return "be"


def _get_encoder_for_wav(encoding: str, bits_per_sample: int) -> str:
    if bits_per_sample not in {None, 8, 16, 24, 32, 64}:
        raise ValueError(f"Invalid bits_per_sample {bits_per_sample} for WAV encoding.")
    endianness = _native_endianness()
    if not encoding:
        if not bits_per_sample:
            # default to PCM S16
            return f"pcm_s16{endianness}"
        if bits_per_sample == 8:
            return "pcm_u8"
        return f"pcm_s{bits_per_sample}{endianness}"
    if encoding == "PCM_S":
        if not bits_per_sample:
            bits_per_sample = 16
        if bits_per_sample == 8:
            raise ValueError("For WAV signed PCM, 8-bit encoding is not supported.")
        return f"pcm_s{bits_per_sample}{endianness}"
    if encoding == "PCM_U":
        if bits_per_sample in (None, 8):
            return "pcm_u8"
        raise ValueError("For WAV unsigned PCM, only 8-bit encoding is supported.")
    if encoding == "PCM_F":
        if not bits_per_sample:
            bits_per_sample = 32
        if bits_per_sample in (32, 64):
            return f"pcm_f{bits_per_sample}{endianness}"
        raise ValueError("For WAV float PCM, only 32- and 64-bit encodings are supported.")
    if encoding == "ULAW":
        if bits_per_sample in (None, 8):
            return "pcm_mulaw"
        raise ValueError("For WAV PCM mu-law, only 8-bit encoding is supported.")
    if encoding == "ALAW":
        if bits_per_sample in (None, 8):
            return "pcm_alaw"
        raise ValueError("For WAV PCM A-law, only 8-bit encoding is supported.")
    raise ValueError(f"WAV encoding {encoding} is not supported.")


def _get_flac_sample_fmt(bps):
    if bps is None or bps == 16:
        return "s16"
    if bps == 24:
        return "s32"
    raise ValueError(f"FLAC only supports bits_per_sample values of 16 and 24 ({bps} specified).")


def _parse_save_args(
    ext: Optional[str],
    format: Optional[str],
    encoding: Optional[str],
    bps: Optional[int],
):
    # torchaudio's save function accepts the followings, which do not 1to1 map
    # to FFmpeg.
    #
    # - format: audio format
    # - bits_per_sample: encoder sample format
    # - encoding: such as PCM_U8.
    #
    # In FFmpeg, format is specified with the following three (and more)
    #
    # - muxer: could be audio format or container format.
    # the one we passed to the constructor of StreamWriter
    # - encoder: the audio encoder used to encode audio
    # - encoder sample format: the format used by encoder to encode audio.
    #
    # If encoder sample format is different from source sample format, StreamWriter
    # will insert a filter automatically.
    #
    def _type(spec):
        # either format is exactly the specified one
        # or extension matches to the spec AND there is no format override.
        return format == spec or (format is None and ext == spec)

    if _type("wav") or _type("amb"):
        # wav is special because it supports different encoding through encoders
        # each encoder only supports one encoder format
        #
        # amb format is a special case originated from libsox.
        # It is basically a WAV format, with slight modification.
        # https://github.com/chirlu/sox/commit/4a4ea33edbca5972a1ed8933cc3512c7302fa67a#diff-39171191a858add9df87f5f210a34a776ac2c026842ae6db6ce97f5e68836795
        # It is a format so that decoders will recognize it as ambisonic.
        # https://www.ambisonia.com/Members/mleese/file-format-for-b-format/
        # FFmpeg does not recognize amb because it is basically a WAV format.
        muxer = "wav"
        encoder = _get_encoder_for_wav(encoding, bps)
        sample_fmt = None
    elif _type("vorbis"):
        # FFpmeg does not recognize vorbis extension, while libsox used to do.
        # For the sake of bakward compatibility, (and the simplicity),
        # we support the case where users want to do save("foo.vorbis")
        muxer = "ogg"
        encoder = "vorbis"
        sample_fmt = None
    else:
        muxer = format
        encoder = None
        sample_fmt = None
        if _type("flac"):
            sample_fmt = _get_flac_sample_fmt(bps)
        if _type("ogg"):
            sample_fmt = _get_flac_sample_fmt(bps)
    return muxer, encoder, sample_fmt


def save_audio(
    uri: InputType,
    src: torch.Tensor,
    sample_rate: int,
    channels_first: bool = True,
    format: Optional[str] = None,
    encoding: Optional[str] = None,
    bits_per_sample: Optional[int] = None,
    buffer_size: int = 4096,
    compression: Optional[torchaudio.io.CodecConfig] = None,
) -> None:
    ext = None
    if hasattr(uri, "write"):
        if format is None:
            raise RuntimeError("'format' is required when saving to file object.")
    else:
        uri = os.path.normpath(uri)
        if tokens := str(uri).split(".")[1:]:
            ext = tokens[-1].lower()

    muxer, encoder, enc_fmt = _parse_save_args(ext, format, encoding, bits_per_sample)

    if channels_first:
        src = src.T

    s = torchaudio.io.StreamWriter(uri, format=muxer, buffer_size=buffer_size)
    s.add_audio_stream(
        sample_rate,
        num_channels=src.size(-1),
        format=_get_sample_format(src.dtype),
        encoder=encoder,
        encoder_format=enc_fmt,
        codec_config=compression,
    )
    with s.open():
        s.write_audio_chunk(0, src)


def _map_encoding(encoding: str) -> str:
    for dst in ["PCM_S", "PCM_U", "PCM_F"]:
        if dst in encoding:
            return dst
    if encoding == "PCM_MULAW":
        return "ULAW"
    elif encoding == "PCM_ALAW":
        return "ALAW"
    return encoding


def _get_bits_per_sample(encoding: str, bits_per_sample: int) -> str:
    if m := re.search(r"PCM_\w(\d+)\w*", encoding):
        return int(m.group(1))
    elif encoding in ["PCM_ALAW", "PCM_MULAW"]:
        return 8
    return bits_per_sample


class FFmpegBackend(Backend):
    @staticmethod
    def info(uri: InputType, format: Optional[str], buffer_size: int = 4096) -> AudioMetaData:
        metadata = info_audio(uri, format, buffer_size)
        metadata.bits_per_sample = _get_bits_per_sample(metadata.encoding, metadata.bits_per_sample)
        metadata.encoding = _map_encoding(metadata.encoding)
        return metadata

    @staticmethod
    def load(
        uri: InputType,
        frame_offset: int = 0,
        num_frames: int = -1,
        normalize: bool = True,
        channels_first: bool = True,
        format: Optional[str] = None,
        buffer_size: int = 4096,
    ) -> Tuple[torch.Tensor, int]:
        return load_audio(uri, frame_offset, num_frames, normalize, channels_first, format)

    @staticmethod
    def save(
        uri: InputType,
        src: torch.Tensor,
        sample_rate: int,
        channels_first: bool = True,
        format: Optional[str] = None,
        encoding: Optional[str] = None,
        bits_per_sample: Optional[int] = None,
        buffer_size: int = 4096,
        compression: Optional[Union[torchaudio.io.CodecConfig, float, int]] = None,
    ) -> None:
        if not isinstance(compression, (torchaudio.io.CodecConfig, type(None))):
            raise ValueError(
                "FFmpeg backend expects non-`None` value for argument `compression` to be of ",
                f"type `torchaudio.io.CodecConfig`, but received value of type {type(compression)}",
            )
        save_audio(
            uri,
            src,
            sample_rate,
            channels_first,
            format,
            encoding,
            bits_per_sample,
            buffer_size,
            compression,
        )

    @staticmethod
    def can_decode(uri: InputType, format: Optional[str]) -> bool:
        return True

    @staticmethod
    def can_encode(uri: InputType, format: Optional[str]) -> bool:
        return True
