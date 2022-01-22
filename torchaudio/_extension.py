import os
import warnings
from pathlib import Path

import torch
from torchaudio._internal import module_utils as _mod_utils  # noqa: F401

_LIB_DIR = Path(__file__).parent / "lib"


def _get_lib_path(lib: str):
    suffix = "pyd" if os.name == "nt" else "so"
    path = _LIB_DIR / f"{lib}.{suffix}"
    return path


def _load_lib(lib: str) -> bool:
    """Load extension module

    Note:
        In case `torchaudio` is deployed with `pex` format, the library file does not
        exist as a stand alone file.
        In this case, we expect that `libtorchaudio` is available somewhere
        in the search path of dynamic loading mechanism, so that importing
        `_torchaudio` will have library loader find and load `libtorchaudio`.
        This is the reason why the function should not raising an error when the library
        file is not found.

    Returns:
        bool:
            False if the library file is not found.
            True if the library file is found AND the library loaded without failure.

    Raises:
        Exception:
            Exception thrown by the underlying `ctypes.CDLL`.
            Expected case is `OSError` when a dynamic dependency is not found.
    """
    path = _get_lib_path(lib)
    if not path.exists():
        return False
    torch.ops.load_library(path)
    torch.classes.load_library(path)
    return True


def _init_extension():
    if not _mod_utils.is_module_available("torchaudio._torchaudio"):
        warnings.warn("torchaudio C++ extension is not available.")
        return

    _load_lib("libtorchaudio")
    # This import is for initializing the methods registered via PyBind11
    # This has to happen after the base library is loaded
    from torchaudio import _torchaudio  # noqa


_init_extension()
