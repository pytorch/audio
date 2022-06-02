from typing import Dict, Optional, Tuple

import torch
import torchaudio
from torchaudio.backend.common import AudioMetaData


# Note: need to comply TorchScript syntax -- need annotation and no f-string nor global
def _info_audio(
    s: torch.classes.torchaudio.ffmpeg_StreamReader,
):
    i = s.find_best_audio_stream()
    sinfo = s.get_src_stream_info(i)
    return AudioMetaData(
        int(sinfo[7]),
        sinfo[5],
        sinfo[8],
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
) -> AudioMetaData:
    s = torchaudio._torchaudio_ffmpeg.StreamReaderFileObj(src, format, None, 4096)
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
    sample_rate = int(sinfo[7])
    option: Dict[str, str] = {}
    s.add_audio_stream(i, -1, -1, _get_load_filter(frame_offset, num_frames, convert), None, option)
    s.process_all_packets()
    waveform = s.pop_chunks()[0]
    if waveform is None:
        raise RuntimeError("Failed to decode audio.")
    assert waveform is not None
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
    src: str,
    frame_offset: int = 0,
    num_frames: int = -1,
    convert: bool = True,
    channels_first: bool = True,
    format: Optional[str] = None,
) -> Tuple[torch.Tensor, int]:
    s = torchaudio._torchaudio_ffmpeg.StreamReaderFileObj(src, format, None, 4096)
    return _load_audio(s, frame_offset, num_frames, convert, channels_first)
