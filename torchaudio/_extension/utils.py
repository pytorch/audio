"""Module to implement logics used for initializing extensions.

The implementations here should be stateless.
They should not depend on external state.
Anything that depends on external state should happen in __init__.py
"""


import importlib
import logging
import os
import platform
import warnings
from functools import wraps
from pathlib import Path

import torch
import torchaudio

_LG = logging.getLogger(__name__)
_LIB_DIR = Path(__file__).parent.parent / "lib"


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


def _init_sox():
    _load_lib("libtorchaudio_sox")
    import torchaudio.lib._torchaudio_sox  # noqa

    torchaudio.lib._torchaudio_sox.set_verbosity(0)

    import atexit

    torch.ops.torchaudio.sox_effects_initialize_sox_effects()
    atexit.register(torch.ops.torchaudio.sox_effects_shutdown_sox_effects)


def _try_access_avutil(ffmpeg_ver):
    libname_template = {
        "Linux": "libavutil.so.{ver}",
        "Darwin": "libavutil.{ver}.dylib",
        "Windows": "avutil-{ver}.dll",
    }[platform.system()]
    avutil_ver = {"6": 58, "5": 57, "4": 56}[ffmpeg_ver]
    libavutil = libname_template.format(ver=avutil_ver)
    torchaudio.lib._torchaudio.find_avutil(libavutil)


def _find_versionsed_ffmpeg_extension(ffmpeg_ver: str):
    _LG.debug("Attempting to load FFmpeg version %s.", ffmpeg_ver)

    library = f"libtorchaudio_ffmpeg{ffmpeg_ver}"
    extension = f"_torchaudio_ffmpeg{ffmpeg_ver}"

    if not _get_lib_path(extension).exists():
        raise RuntimeError(f"FFmpeg {ffmpeg_ver} extension is not available.")

    if ffmpeg_ver:
        # A simple check for FFmpeg availability.
        # This is not technically sufficient as other libraries could be missing,
        # but usually this is sufficient.
        #
        # Note: the reason why this check is performed is because I don't know
        # if the next `_load_lib` (which calls `ctypes.CDLL` under the hood),
        # could leak handle to shared libraries of dependencies, in case it fails.
        #
        # i.e. If the `ctypes.CDLL("foo")` fails because one of `foo`'s dependency
        # does not exist while `foo` and some other dependencies exist, is it guaranteed
        # that none-of them are kept in memory after the failure??
        _try_access_avutil(ffmpeg_ver)

    _load_lib(library)

    _LG.debug("Found FFmpeg version %s.", ffmpeg_ver)
    return importlib.import_module(f"torchaudio.lib.{extension}")


_FFMPEG_VERS = ["6", "5", "4", ""]


def _find_ffmpeg_extension(ffmpeg_vers, show_error):
    logger = _LG.error if show_error else _LG.debug
    for ffmpeg_ver in ffmpeg_vers:
        try:
            return _find_versionsed_ffmpeg_extension(ffmpeg_ver)
        except Exception:
            logger("Failed to load FFmpeg %s extension.", ffmpeg_ver, exc_info=True)
            continue
    raise ImportError(f"Failed to intialize FFmpeg extension. Tried versions: {ffmpeg_vers}")


def _find_available_ffmpeg_ext():
    ffmpeg_vers = ["6", "5", "4", ""]
    return [v for v in ffmpeg_vers if _get_lib_path(f"_torchaudio_ffmpeg{v}").exists()]


def _init_ffmpeg(show_error=False):
    ffmpeg_vers = _find_available_ffmpeg_ext()
    if not ffmpeg_vers:
        raise RuntimeError(
            # fmt: off
            "TorchAudio is not built with FFmpeg integration. "
            "Please build torchaudio with USE_FFMPEG=1."
            # fmt: on
        )

    # User override
    if ffmpeg_ver := os.environ.get("TORCHAUDIO_USE_FFMPEG_VERSION"):
        if ffmpeg_vers == [""]:
            warnings.warn("TorchAudio is built in single FFmpeg mode. TORCHAUDIO_USE_FFMPEG_VERSION is ignored.")
        else:
            if ffmpeg_ver not in ffmpeg_vers:
                raise ValueError(
                    f"The FFmpeg version {ffmpeg_ver} (read from TORCHAUDIO_USE_FFMPEG_VERSION) "
                    f"is not available. Available versions are {[v for v in ffmpeg_vers if v]}"
                )
            ffmpeg_vers = [ffmpeg_ver]

    ext = _find_ffmpeg_extension(ffmpeg_vers, show_error)
    ext.init()
    if ext.get_log_level() > 8:
        ext.set_log_level(8)
    return ext


def _init_dll_path():
    # On Windows Python-3.8+ has `os.add_dll_directory` call,
    # which is called to configure dll search path.
    # To find cuda related dlls we need to make sure the
    # conda environment/bin path is configured Please take a look:
    # https://stackoverflow.com/questions/59330863/cant-import-dll-module-in-python
    # Please note: if some path can't be added using add_dll_directory we simply ignore this path
    for path in os.environ.get("PATH", "").split(";"):
        if os.path.exists(path):
            try:
                os.add_dll_directory(path)
            except Exception:
                pass


def _check_cuda_version():
    version = torchaudio.lib._torchaudio.cuda_version()
    if version is not None and torch.version.cuda is not None:
        version_str = str(version)
        ta_version = f"{version_str[:-3]}.{version_str[-2]}"
        t_version = torch.version.cuda.split(".")
        t_version = f"{t_version[0]}.{t_version[1]}"
        if ta_version != t_version:
            raise RuntimeError(
                "Detected that PyTorch and TorchAudio were compiled with different CUDA versions. "
                f"PyTorch has CUDA version {t_version} whereas TorchAudio has CUDA version {ta_version}. "
                "Please install the TorchAudio version that matches your PyTorch version."
            )
    return version


def _fail_since_no_sox(func):
    @wraps(func)
    def wrapped(*_args, **_kwargs):
        try:
            # Note:
            # We run _init_sox again just to show users the stacktrace.
            # _init_sox would not succeed here.
            _init_sox()
        except Exception as err:
            raise RuntimeError(
                f"{func.__name__} requires sox extension which is not available. "
                "Please refer to the stacktrace above for how to resolve this."
            ) from err
        # This should not happen in normal execution, but just in case.
        return func(*_args, **_kwargs)

    return wrapped


def _fail_since_no_ffmpeg(func):
    @wraps(func)
    def wrapped(*_args, **_kwargs):
        try:
            # Note:
            # We run _init_ffmpeg again just to show users the stacktrace.
            # _init_ffmpeg would not succeed here.
            _init_ffmpeg(show_error=True)
        except Exception as err:
            raise RuntimeError(
                f"{func.__name__} requires FFmpeg extension which is not available. "
                "Please refer to the stacktrace above for how to resolve this."
            ) from err
        # This should not happen in normal execution, but just in case.
        return func(*_args, **_kwargs)

    return wrapped
