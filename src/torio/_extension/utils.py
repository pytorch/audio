import importlib
import logging
import os
import types
from pathlib import Path

import torch

_LG = logging.getLogger(__name__)
_LIB_DIR = Path(__file__).parent.parent / "lib"


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


def _get_lib_path(lib: str):
    suffix = "pyd" if os.name == "nt" else "so"
    path = _LIB_DIR / f"{lib}.{suffix}"
    return path


def _load_lib(lib: str) -> bool:
    """Load extension module

    Note:
        In case `torio` is deployed with `pex` format, the library file
        is not in a standard location.
        In this case, we expect that `libtorio` is available somewhere
        in the search path of dynamic loading mechanism, so that importing
        `_torio` will have library loader find and load `libtorio`.
        This is the reason why the function should not raising an error when the library
        file is not found.

    Returns:
        bool:
            True if the library file is found AND the library loaded without failure.
            False if the library file is not found (like in the case where torio
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


_FFMPEG_VERS = ["6", "5", "4", ""]


def _find_versionsed_ffmpeg_extension(version: str):
    ext = f"torio.lib._torio_ffmpeg{version}"
    lib = f"libtorio_ffmpeg{version}"

    if not importlib.util.find_spec(ext):
        raise RuntimeError(f"FFmpeg{version} extension is not available.")

    _load_lib(lib)
    return importlib.import_module(ext)


def _find_ffmpeg_extension(ffmpeg_vers):
    for ffmpeg_ver in ffmpeg_vers:
        _LG.debug("Loading FFmpeg%s", ffmpeg_ver)
        try:
            ext = _find_versionsed_ffmpeg_extension(ffmpeg_ver)
            _LG.debug("Successfully loaded FFmpeg%s", ffmpeg_ver)
            return ext
        except Exception:
            _LG.debug("Failed to load FFmpeg%s extension.", ffmpeg_ver, exc_info=True)
            continue
    raise ImportError(
        f"Failed to intialize FFmpeg extension. Tried versions: {ffmpeg_vers}. "
        "Enable DEBUG logging to see more details about the error."
    )


def _get_ffmpeg_versions():
    ffmpeg_vers = _FFMPEG_VERS
    # User override
    if (ffmpeg_ver := os.environ.get("TORIO_USE_FFMPEG_VERSION")) is not None:
        if ffmpeg_ver not in ffmpeg_vers:
            raise ValueError(
                f"The FFmpeg version '{ffmpeg_ver}' (read from TORIO_USE_FFMPEG_VERSION) "
                f"is not one of supported values. Possible values are {ffmpeg_vers}"
            )
        ffmpeg_vers = [ffmpeg_ver]
    return ffmpeg_vers


def _init_ffmpeg():
    ffmpeg_vers = _get_ffmpeg_versions()
    ext = _find_ffmpeg_extension(ffmpeg_vers)
    ext.init()
    if ext.get_log_level() > 8:
        ext.set_log_level(8)
    return ext
