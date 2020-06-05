import os.path
from typing import Optional, Tuple

import torch
from torch import Tensor

import torchaudio
from torchaudio._internal import (
    module_utils as _mod_utils,
    misc_ops as _misc_ops,
)
from torchaudio._soundfile_backend import SignalInfo, EncodingInfo

if _mod_utils.is_module_available('torchaudio._torchaudio'):
    from . import _torchaudio


@_mod_utils.requires_module('torchaudio._torchaudio')
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
        _misc_ops.check_input(out)
    else:
        out = torch.FloatTensor()

    if num_frames < -1:
        raise ValueError("Expected value for num_samples -1 (entire file) or >=0")
    if offset < 0:
        raise ValueError("Expected positive offset value")

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
    _misc_ops.normalize_audio(out, normalization)

    return out, sample_rate


@_mod_utils.requires_module('torchaudio._torchaudio')
def save(filepath: str, src: Tensor, sample_rate: int, precision: int = 16, channels_first: bool = True) -> None:
    r"""See torchaudio.save"""

    si = torchaudio.sox_signalinfo_t()
    ch_idx = 0 if channels_first else 1
    si.rate = sample_rate
    si.channels = 1 if src.dim() == 1 else src.size(ch_idx)
    si.length = src.numel()
    si.precision = precision
    return torchaudio.save_encinfo(filepath, src, channels_first, si)


@_mod_utils.requires_module('torchaudio._torchaudio')
def info(filepath: str) -> Tuple[SignalInfo, EncodingInfo]:
    r"""See torchaudio.info"""
    return _torchaudio.get_info(filepath)
