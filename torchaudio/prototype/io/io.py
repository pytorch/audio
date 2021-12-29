from __future__ import annotations

from typing import List, Tuple

import torch
import torchaudio

from .streamer import Streamer, _parse_si


def info(src: str) -> List[torchaudio.prototype.io.SourceStream]:
    """Get the stream information of the source

    Args:
        src (str): Source URI.

    Returns:
        list of SourceStream: Stream information.
    """
    s = Streamer(src)
    return [
        _parse_si(torch.ops.torchaudio.ffmpeg_streamer_get_src_stream_info(s._s, i)) for i in range(s.num_src_streams)
    ]


def load(src: str) -> Tuple[torch.Tensor, int]:
    """Load audio from source

    Args:
        src (str): Source URI.

    Returns:
        `(Tensor, int)`:
            Audio Tensor and its sampling rate.
    """
    return torch.ops.torchaudio.ffmpeg_load(src)
