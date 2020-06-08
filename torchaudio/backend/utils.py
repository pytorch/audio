from typing import Any, Optional

from torchaudio._internal import module_utils as _mod_utils
from . import soundfile_backend, sox_backend

_BACKEND = None
_BACKENDS = {}

if _mod_utils.is_module_available('soundfile'):
    _BACKENDS['soundfile'] = soundfile_backend
if _mod_utils.is_module_available('torchaudio._torchaudio'):
    _BACKENDS['sox'] = sox_backend

if 'sox' in _BACKENDS:
    _BACKEND = 'sox'
elif 'soundfile' in _BACKENDS:
    _BACKEND = 'soundfile'


def list_audio_backends():
    return list(_BACKENDS.keys())


def set_audio_backend(backend: str) -> None:
    """
    Specifies the package used to load.
    Args:
        backend (str): Name of the backend. One of "sox" or "soundfile",
            based on availability of the system.
    """
    if backend not in _BACKENDS:
        raise RuntimeError(
            f'Backend "{backend}" is not one of '
            f'available backends: {list_audio_backends()}.')
    global _BACKEND
    _BACKEND = backend


def get_audio_backend() -> Optional[str]:
    """
    Gets the name of the package used to load.
    """
    return _BACKEND


def _get_audio_backend_module() -> Any:
    """
    Gets the module backend to load.
    """
    if _BACKEND is None:
        raise RuntimeError('Backend is not initialized.')
    return _BACKENDS[_BACKEND]
