import os.path
from typing import Optional, Tuple

import torch
from torch import Tensor
import torchaudio
from torchaudio._soundfile_backend import SignalInfo, EncodingInfo


def load(filepath: str,
         out: Optional[Tensor] = None,
         normalization: bool = True,
         channels_first: bool = True,
         num_frames: int = 0,
         offset: int = 0,
         signalinfo: SignalInfo = None,
         encodinginfo: EncodingInfo = None,
         filetype: Optional[str] = None) -> Tuple[Tensor, int]:
    r"""See torchaudio.load"""

    # stringify if `pathlib.Path` (noop if already `str`)
    filepath = str(filepath)
    # check if valid file
    if not os.path.isfile(filepath):
        raise OSError("{} not found or is a directory".format(filepath))

    # initialize output tensor
    if out is not None:
        torchaudio.check_input(out)
    else:
        out = torch.FloatTensor()

    if num_frames < -1:
        raise ValueError("Expected value for num_samples -1 (entire file) or >=0")
    if offset < 0:
        raise ValueError("Expected positive offset value")

    from . import _torchaudio
    sample_rate = _torchaudio.read_audio_file(
        filepath,
        out,
        channels_first,
        num_frames,
        offset,
        signalinfo,
        encodinginfo,
        filetype
    )

    # normalize if needed
    torchaudio._audio_normalization(out, normalization)

    return out, sample_rate


def save(filepath: str, src: Tensor, sample_rate: int, precision: int = 16, channels_first: bool = True) -> None:
    r"""See torchaudio.save"""

    si = torchaudio.sox_signalinfo_t()
    ch_idx = 0 if channels_first else 1
    si.rate = sample_rate
    si.channels = 1 if src.dim() == 1 else src.size(ch_idx)
    si.length = src.numel()
    si.precision = precision
    return torchaudio.save_encinfo(filepath, src, channels_first, si)


def info(filepath: str) -> Tuple[SignalInfo, EncodingInfo]:
    r"""See torchaudio.info"""
    info = torch.ops.torchaudio.sox_get_info(filepath)
    si = info.GetSignalInfo()
    ei = info.GetEncodingInfo()
    return (
        SignalInfo(si.GetChannels(), si.GetRate(), si.GetPrecision(), si.GetLength()),
        EncodingInfo(
            ei.GetEncoding(), ei.GetBPS(), ei.GetCompression(),
            ei.GetReverseBytes(), ei.GetReverseNibbles(), ei.GetReverseBits(),
            ei.GetOppositeEndian()),
    )
