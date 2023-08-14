"""Defines utilities for switching audio backends"""
import warnings
from typing import List, Optional

import torchaudio
from torchaudio._backend import soundfile_backend
from torchaudio._internal import module_utils as _mod_utils

from . import _no_backend as no_backend, _sox_io_backend as sox_io_backend

__all__ = [
    "list_audio_backends",
    "get_audio_backend",
    "set_audio_backend",
]


def list_audio_backends() -> List[str]:
    """List available backends

    Returns:
        List[str]: The list of available backends.
    """
    backends = []
    if _mod_utils.is_module_available("soundfile"):
        backends.append("soundfile")
    if torchaudio._extension._SOX_INITIALIZED:
        backends.append("sox_io")
    return backends


def set_audio_backend(backend: Optional[str]):
    """Set the backend for I/O operation

    Args:
        backend (str or None): Name of the backend.
            One of ``"sox_io"`` or ``"soundfile"`` based on availability
            of the system. If ``None`` is provided the  current backend is unassigned.
    """
    if backend is not None and backend not in list_audio_backends():
        raise RuntimeError(f'Backend "{backend}" is not one of ' f"available backends: {list_audio_backends()}.")

    if backend is None:
        module = no_backend
    elif backend == "sox_io":
        module = sox_io_backend
    elif backend == "soundfile":
        module = soundfile_backend
    else:
        raise NotImplementedError(f'Unexpected backend "{backend}"')

    for func in ["save", "load", "info"]:
        setattr(torchaudio, func, getattr(module, func))


def _init_backend():
    warnings.warn(
        "TorchAudio's global backend is now deprecated. "
        "Please enable distpatcher by setting `TORCHAUDIO_USE_BACKEND_DISPATCHER=1`, "
        "and specify backend when calling load/info/save function.",
        stacklevel=3,
    )
    backends = list_audio_backends()
    if "sox_io" in backends:
        set_audio_backend("sox_io")
    elif "soundfile" in backends:
        set_audio_backend("soundfile")
    else:
        set_audio_backend(None)


def get_audio_backend() -> Optional[str]:
    """Get the name of the current backend

    Returns:
        Optional[str]: The name of the current backend or ``None`` if no backend is assigned.
    """
    if torchaudio.load == no_backend.load:
        return None
    if torchaudio.load == sox_io_backend.load:
        return "sox_io"
    if torchaudio.load == soundfile_backend.load:
        return "soundfile"
    raise ValueError("Unknown backend.")
