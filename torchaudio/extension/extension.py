import warnings
import importlib

import torch
from torchaudio._internal import module_utils as _mod_utils


def _init_extension():
    ext = 'torchaudio._torchaudio'
    if _mod_utils.is_module_available(ext):
        _init_script_module(ext)
    else:
        warnings.warn('torchaudio C++ extension is not available.')


def _init_script_module(module):
    path = importlib.util.find_spec(module).origin
    torch.classes.load_library(path)
    torch.ops.load_library(path)
