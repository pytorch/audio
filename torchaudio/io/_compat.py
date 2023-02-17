import os
import sys
from typing import BinaryIO, Dict, Optional, Tuple, Union

import torch
import torchaudio
from torchaudio.backend.common import AudioMetaData
from torchaudio.io import StreamWriter


# Note: need to comply TorchScript syntax -- need annotation and no f-string nor global
def _info_audio(
    s: torch.classes.torchaudio.ffmpeg_StreamReader,
):
    i = s.find_best_audio_stream()
    sinfo = s.get_src_stream_info(i)
    if sinfo[5] == 0:
        waveform, _ = _load_audio(s)
        num_frames = waveform.size(1)
    else:
        num_frames = sinfo[5]
    return AudioMetaData(
        int(sinfo[8]),
        num_frames,
        sinfo[9],
        sinfo[6],
        sinfo[1].upper(),
    )


def info_audio(
    src: str,
    format: Optional[str],
) -> AudioMetaData:
    s = torch.classes.torchaudio.ffmpeg_StreamReader(src, format, None)
    return _info_audio(s)


def info_audio_fileobj(
    src,
    format: Optional[str],
    buffer_size: int = 4096,
) -> AudioMetaData:
    s = torchaudio.lib._torchaudio_ffmpeg.StreamReaderFileObj(src, format, None, buffer_size)
    return _info_audio(s)


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


# Note: need to comply TorchScript syntax -- need annotation and no f-string nor global
def _load_audio(
    s: torch.classes.torchaudio.ffmpeg_StreamReader,
    frame_offset: int = 0,
    num_frames: int = -1,
    convert: bool = True,
    channels_first: bool = True,
) -> Tuple[torch.Tensor, int]:
    i = s.find_best_audio_stream()
    sinfo = s.get_src_stream_info(i)
    sample_rate = int(sinfo[8])
    option: Dict[str, str] = {}
    s.add_audio_stream(i, -1, -1, _get_load_filter(frame_offset, num_frames, convert), None, option)
    s.process_all_packets()
    chunk = s.pop_chunks()[0]
    if chunk is None:
        raise RuntimeError("Failed to decode audio.")
    assert chunk is not None
    waveform = chunk[0]
    if channels_first:
        waveform = waveform.T
    return waveform, sample_rate


def load_audio(
    src: str,
    frame_offset: int = 0,
    num_frames: int = -1,
    convert: bool = True,
    channels_first: bool = True,
    format: Optional[str] = None,
) -> Tuple[torch.Tensor, int]:
    s = torch.classes.torchaudio.ffmpeg_StreamReader(src, format, None)
    return _load_audio(s, frame_offset, num_frames, convert, channels_first)


def load_audio_fileobj(
    src: BinaryIO,
    frame_offset: int = 0,
    num_frames: int = -1,
    convert: bool = True,
    channels_first: bool = True,
    format: Optional[str] = None,
    buffer_size: int = 4096,
) -> Tuple[torch.Tensor, int]:
    s = torchaudio.lib._torchaudio_ffmpeg.StreamReaderFileObj(src, format, None, buffer_size)
    return _load_audio(s, frame_offset, num_frames, convert, channels_first)


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


def _get_encoder_for_wav(dtype: torch.dtype, encoding: str, bits_per_sample: int) -> str:
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
    elif encoding == "PCM_U":
        if bits_per_sample in (None, 8):
            return "pcm_u8"
        raise ValueError("For WAV unsigned PCM, only 8-bit encoding is supported.")
    elif encoding == "PCM_F":
        if not bits_per_sample:
            bits_per_sample = 32
        if bits_per_sample in (32, 64):
            return f"pcm_f{bits_per_sample}{endianness}"
        raise ValueError("For WAV float PCM, only 32- and 64-bit encodings are supported.")
    elif encoding == "ULAW":
        if bits_per_sample in (None, 8):
            return "pcm_mulaw"
        raise ValueError("For WAV PCM mu-law, only 8-bit encoding is supported.")
    elif encoding == "ALAW":
        if bits_per_sample in (None, 8):
            return "pcm_alaw"
        raise ValueError("For WAV PCM A-law, only 8-bit encoding is supported.")
    raise ValueError(f"WAV encoding {encoding} is not supported.")


def _get_encoder(dtype: torch.dtype, format: str, encoding: str, bits_per_sample: int) -> str:
    if format == "wav":
        return _get_encoder_for_wav(dtype, encoding, bits_per_sample)
    if format == "flac":
        return "flac"
    if format in ("ogg", "vorbis"):
        if encoding or bits_per_sample:
            raise ValueError("ogg/vorbis does not support encoding/bits_per_sample.")
        return "vorbis"
    return format


def _get_encoder_format(format: str, bits_per_sample: Optional[int]) -> str:
    if format == "flac":
        if not bits_per_sample:
            return "s16"
        if bits_per_sample == 24:
            return "s32"
        if bits_per_sample == 16:
            return "s16"
        raise ValueError(f"FLAC only supports bits_per_sample values of 16 and 24 ({bits_per_sample} specified).")
    return None


# NOTE: in contrast to load_audio* and info_audio*, this function is NOT compatible with TorchScript.
def save_audio(
    uri: Union[BinaryIO, str, os.PathLike],
    src: torch.Tensor,
    sample_rate: int,
    channels_first: bool = True,
    format: Optional[str] = None,
    encoding: Optional[str] = None,
    bits_per_sample: Optional[int] = None,
    buffer_size: int = 4096,
) -> None:
    if hasattr(uri, "write") and format is None:
        raise RuntimeError("'format' is required when saving to file object.")
    s = StreamWriter(uri, format=format, buffer_size=buffer_size)
    if format is None:
        tokens = str(uri).split(".")
        if len(tokens) > 1:
            format = tokens[-1].lower()

    if channels_first:
        src = src.T
    s.add_audio_stream(
        sample_rate,
        src.size(-1),
        _get_sample_format(src.dtype),
        _get_encoder(src.dtype, format, encoding, bits_per_sample),
        {"strict": "experimental"},
        _get_encoder_format(format, bits_per_sample),
    )
    with s.open():
        s.write_audio_chunk(0, src)
