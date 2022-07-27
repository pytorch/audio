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
        In case `torchaudio` is deployed with `pex` format, the library file
        is not in a standard location.
        In this case, we expect that `libtorchaudio` is available somewhere
        in the search path of dynamic loading mechanism, so that importing
        `_torchaudio` will have library loader find and load `libtorchaudio`.
        This is the reason why the function should not raising an error when the library
        file is not found.

    Returns:
        bool:
            True if the library file is found AND the library loaded without failure.
            False if the library file is not found (like in the case where torchaudio
            is deployed with pex format, thus the shared library file is
            in a non-standard location.).
            If the library file is found but there is an issue loading the library,
            (such as missing dependency) then this function raises the exception as-is.

    Raises:
        Exception:
            If the library file is found, but there is an issue loading the library file,
            (when underlying `ctype.DLL` throws an exception), this function will pass
            the exception as-is, instead of catching it and returning bool.
            The expected case is `OSError` thrown by `ctype.DLL` when a dynamic dependency
            is not found.
            This behavior was chosen because the expected failure case is not recoverable.
            If a dependency is missing, then users have to install it.
    """
    path = _get_lib_path(lib)
    if not path.exists():
        return False
    torch.ops.load_library(path)
    torch.classes.load_library(path)
    return True


_FFMPEG_INITIALIZED = False


def _init_ffmpeg():
    global _FFMPEG_INITIALIZED
    if _FFMPEG_INITIALIZED:
        return

    if not torch.ops.torchaudio.is_ffmpeg_available():
        raise RuntimeError(
            "torchaudio is not compiled with FFmpeg integration. Please set USE_FFMPEG=1 when compiling torchaudio."
        )

    try:
        _load_lib("libtorchaudio_ffmpeg")
    except OSError as err:
        raise ImportError("FFmpeg libraries are not found. Please install FFmpeg.") from err

    import torchaudio._torchaudio_ffmpeg  # noqa

    torch.ops.torchaudio.ffmpeg_init()
    if torch.ops.torchaudio.ffmpeg_get_log_level() > 8:
        torch.ops.torchaudio.ffmpeg_set_log_level(8)

    _FFMPEG_INITIALIZED = True


def _init_extension():
    if not _mod_utils.is_module_available("torchaudio._torchaudio"):
        warnings.warn("torchaudio C++ extension is not available.")
        return

    _load_lib("libtorchaudio")
    # This import is for initializing the methods registered via PyBind11
    # This has to happen after the base library is loaded
    from torchaudio import _torchaudio  # noqa

    # Because this part is executed as part of `import torchaudio`, we ignore the
    # initialization failure.
    # If the FFmpeg integration is not properly initialized, then detailed error
    # will be raised when client code attempts to import the dedicated feature.
    try:
        _init_ffmpeg()
    except Exception:
        pass


_init_extension()
