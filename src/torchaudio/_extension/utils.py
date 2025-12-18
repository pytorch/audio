"""Module to implement logics used for initializing extensions.

The implementations here should be stateless.
They should not depend on external state.
Anything that depends on external state should happen in __init__.py
"""
import logging
import os
import types
import warnings
from pathlib import Path

import torch

_LG = logging.getLogger(__name__)
_LIB_DIR = Path(__file__).parent.parent / "lib"


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
    suffix = ".pyd" if os.name == "nt" else ".so"
    paths = list(_LIB_DIR.glob(f"{lib}*{suffix}"))
    if not paths:
        return False
    if len(paths) > 1:
        warnings.warn(f"Expected a single file path to {lib}, got {paths=}")
    torch.ops.load_library(paths[0])
    return True


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
    version = torch.ops._torchaudio.cuda_version()
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
