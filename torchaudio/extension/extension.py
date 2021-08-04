import warnings

import torch
from torchaudio._internal import module_utils as _mod_utils


def _init_extension():
    if _mod_utils.is_module_available('torchaudio._torchaudio'):
        # Note this import has two purposes
        # 1. to extract the path of the extension module so that
        #    we can initialize the script module with the path.
        # 2. so that torchaudio._torchaudio is accessible in other modules.
        #    Look at sox_io_backend which uses `torchaudio._torchaudio.XXX`,
        #    assuming that the module `_torchaudio` is accessible.
        import torchaudio._torchaudio
        _init_script_module(torchaudio._torchaudio.__file__)
    else:
        warnings.warn('torchaudio C++ extension is not available.')


def _init_script_module(path):
    torch.classes.load_library(path)
    torch.ops.load_library(path)
