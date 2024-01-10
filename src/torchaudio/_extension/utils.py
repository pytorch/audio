"""Module to implement logics used for initializing extensions.

The implementations here should be stateless.
They should not depend on external state.
Anything that depends on external state should happen in __init__.py
"""
import importlib
import logging
import os
import types
from pathlib import Path

import torch
from torchaudio._internal.module_utils import eval_env

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
    return True


def _import_sox_ext():
    if os.name == "nt":
        raise RuntimeError("sox extension is not supported on Windows")
    if not eval_env("TORCHAUDIO_USE_SOX", True):
        raise RuntimeError("sox extension is disabled. (TORCHAUDIO_USE_SOX=0)")

    ext = "torchaudio.lib._torchaudio_sox"

    if not importlib.util.find_spec(ext):
        raise RuntimeError(
            # fmt: off
            "TorchAudio is not built with sox extension. "
            "Please build TorchAudio with libsox support. (BUILD_SOX=1)"
            # fmt: on
        )

    _load_lib("libtorchaudio_sox")
    return importlib.import_module(ext)


def _init_sox():
    ext = _import_sox_ext()
    ext.set_verbosity(0)

    import atexit

    torch.ops.torchaudio_sox.initialize_sox_effects()
    atexit.register(torch.ops.torchaudio_sox.shutdown_sox_effects)

    # Bundle functions registered with TORCH_LIBRARY into extension
    # so that they can also be accessed in the same (lazy) manner
    # from the extension.
    keys = [
        "get_info",
        "load_audio_file",
        "save_audio_file",
        "apply_effects_tensor",
        "apply_effects_file",
    ]
    for key in keys:
        setattr(ext, key, getattr(torch.ops.torchaudio_sox, key))

    return ext


class _LazyImporter(types.ModuleType):
    """Lazily import module/extension."""

    def __init__(self, name, import_func):
        super().__init__(name)
        self.import_func = import_func
        self.module = None

    # Note:
    # Python caches what was retrieved with `__getattr__`, so this method will not be
    # called again for the same item.
    def __getattr__(self, item):
        self._import_once()
        return getattr(self.module, item)

    def __repr__(self):
        if self.module is None:
            return f"<module '{self.__module__}.{self.__class__.__name__}(\"{self.name}\")'>"
        return repr(self.module)

    def __dir__(self):
        self._import_once()
        return dir(self.module)

    def _import_once(self):
        if self.module is None:
            self.module = self.import_func()
            # Note:
            # By attaching the module attributes to self,
            # module attributes are directly accessible.
            # This allows to avoid calling __getattr__ for every attribute access.
            self.__dict__.update(self.module.__dict__)

    def is_available(self):
        try:
            self._import_once()
        except Exception:
            return False
        return True


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
    import torchaudio.lib._torchaudio

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
