import warnings

import torch
from torchaudio._internal import module_utils as _mod_utils


def _init_extension():
    if _mod_utils.is_module_available('torchaudio._torchaudio'):
        import torchaudio._torchaudio  # noqa
        _init_script_module(torchaudio._torchaudio.__file__)
    else:
        warnings.warn('torchaudio C++ extension is not available.')


def _init_script_module(path):
    torch.classes.load_library(path)
    torch.ops.load_library(path)
