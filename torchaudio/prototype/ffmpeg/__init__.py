import torch
import torchaudio

torchaudio._extension._load_lib("libtorchaudio_ffmpeg")
torch.ops.torchaudio.ffmpeg_init()

from .io import (
    info,
    load,
    Streamer,
)

__all__ = [
    "info",
    "load",
    "Streamer",
]
