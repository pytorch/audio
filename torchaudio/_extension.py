import os
import warnings
from pathlib import Path

import torch
from torchaudio._internal import module_utils as _mod_utils  # noqa: F401


def _init_extension():
    if not _mod_utils.is_module_available('torchaudio._torchaudio'):
        warnings.warn('torchaudio C++ extension is not available.')
        return

    suffix = 'pyd' if os.name == 'nt' else 'so'
    path = Path(__file__).parent / 'lib' / f'libtorchaudio.{suffix}'
    # In case `torchaudio` is deployed with `pex` format, this file does not exist.
    # In this case, we expect that `libtorchaudio` is available somewhere
    # in the search path of dynamic loading mechanism, and importing `_torchaudio`,
    # which depends on `libtorchaudio` and dynamic loader will handle it for us.
    if path.exists():
        torch.ops.load_library(path)
        torch.classes.load_library(path)
    # This import is for initializing the methods registered via PyBind11
    from torchaudio import _torchaudio  # noqa


_init_extension()
