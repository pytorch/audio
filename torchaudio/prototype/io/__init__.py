import torch
import torchaudio

torchaudio._extension._load_lib("libtorchaudio_ffmpeg")
torch.ops.torchaudio.ffmpeg_init()

from .io import (
    info,
    load,
)
from .streamer import (
    Streamer,
    SourceStream,
    SourceAudioStream,
    SourceVideoStream,
    OutputStream,
)

__all__ = [
    "info",
    "load",
    "Streamer",
    "SourceStream",
    "SourceAudioStream",
    "SourceVideoStream",
    "OutputStream",
]
