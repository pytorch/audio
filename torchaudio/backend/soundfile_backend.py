import os
from typing import Optional, Tuple

import torch
from torch import Tensor

from torchaudio._internal import (
    module_utils as _mod_utils,
    misc_ops as _misc_ops,
)
from . import common
from .common import SignalInfo, EncodingInfo

if _mod_utils.is_module_available('soundfile'):
    import soundfile


_subtype_to_precision = {
    'PCM_S8': 8,
    'PCM_16': 16,
    'PCM_24': 24,
    'PCM_32': 32,
    'PCM_U8': 8
}


@_mod_utils.requires_module('soundfile')
@common._impl_load
def load(filepath: str,
         out: Optional[Tensor] = None,
         normalization: Optional[bool] = True,
         channels_first: Optional[bool] = True,
         num_frames: int = 0,
         offset: int = 0,
         signalinfo: SignalInfo = None,
         encodinginfo: EncodingInfo = None,
         filetype: Optional[str] = None) -> Tuple[Tensor, int]:
    r"""See torchaudio.load"""

    assert out is None
    assert normalization
    assert signalinfo is None
    assert encodinginfo is None

    # stringify if `pathlib.Path` (noop if already `str`)
    filepath = str(filepath)

    # check if valid file
    if not os.path.isfile(filepath):
        raise OSError("{} not found or is a directory".format(filepath))

    if num_frames < -1:
        raise ValueError("Expected value for num_samples -1 (entire file) or >=0")
    if num_frames == 0:
        num_frames = -1
    if offset < 0:
        raise ValueError("Expected positive offset value")

    # initialize output tensor
    # TODO call libsoundfile directly to avoid numpy
    out, sample_rate = soundfile.read(
        filepath, frames=num_frames, start=offset, dtype="float32", always_2d=True
    )
    out = torch.from_numpy(out).t()

    if not channels_first:
        out = out.t()

    # normalize if needed
    # _audio_normalization(out, normalization)

    return out, sample_rate


@_mod_utils.requires_module('soundfile')
@common._impl_load_wav
def load_wav(filepath, **kwargs):
    # kwargs['normalization'] = 1 << 16
    return load(filepath, **kwargs)


@_mod_utils.requires_module('soundfile')
@common._impl_save
def save(filepath: str, src: Tensor, sample_rate: int, precision: int = 16, channels_first: bool = True) -> None:
    r"""See torchaudio.save"""

    ch_idx, len_idx = (0, 1) if channels_first else (1, 0)

    # check if save directory exists
    abs_dirpath = os.path.dirname(os.path.abspath(filepath))
    if not os.path.isdir(abs_dirpath):
        raise OSError("Directory does not exist: {}".format(abs_dirpath))
    # check that src is a CPU tensor
    _misc_ops.check_input(src)
    # Check/Fix shape of source data
    if src.dim() == 1:
        # 1d tensors as assumed to be mono signals
        src.unsqueeze_(ch_idx)
    elif src.dim() > 2 or src.size(ch_idx) > 16:
        # assumes num_channels < 16
        raise ValueError(
            "Expected format where C < 16, but found {}".format(src.size()))

    if channels_first:
        src = src.t()

    if src.dtype == torch.int64:
        # Soundfile doesn't support int64
        src = src.type(torch.int32)

    precision = "PCM_S8" if precision == 8 else "PCM_" + str(precision)

    return soundfile.write(filepath, src, sample_rate, precision)


@_mod_utils.requires_module('soundfile')
@common._impl_info
def info(filepath: str) -> Tuple[SignalInfo, EncodingInfo]:
    r"""See torchaudio.info"""

    sfi = soundfile.info(filepath)

    precision = _subtype_to_precision[sfi.subtype]
    si = SignalInfo(sfi.channels, sfi.samplerate, precision, sfi.frames)
    ei = EncodingInfo(bits_per_sample=precision)
    return si, ei
