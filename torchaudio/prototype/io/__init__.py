import torch
import torchaudio

torchaudio._extension._load_lib("libtorchaudio_ffmpeg")
torch.ops.torchaudio.ffmpeg_init()

from .streamer import (
    Streamer,
    SourceStream,
    SourceAudioStream,
    SourceVideoStream,
    OutputStream,
)

__all__ = [
    "Streamer",
    "SourceStream",
    "SourceAudioStream",
    "SourceVideoStream",
    "OutputStream",
]
