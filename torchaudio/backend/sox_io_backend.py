import torch
from torchaudio._internal import (
    module_utils as _mod_utils,
)


@_mod_utils.requires_module('torchaudio._torchaudio')
def info(filepath: str) -> torch.classes.torchaudio.SignalInfo:
    """Get signal information of an audio file."""
    return torch.ops.torchaudio.sox_io_get_info(filepath)
