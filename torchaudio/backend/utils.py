"""Defines utilities for switching audio backends"""
import warnings
from typing import Optional, List

import torchaudio
from torchaudio._internal.module_utils import is_module_available
from . import (
    no_backend,
    sox_backend,
    sox_io_backend,
    soundfile_backend,
)

__all__ = [
    'list_audio_backends',
    'get_audio_backend',
    'set_audio_backend',
]


def list_audio_backends() -> List[str]:
    """List available backends"""
    backends = []
    if is_module_available('soundfile'):
        backends.append('soundfile')
    if is_module_available('torchaudio._torchaudio'):
        backends.append('sox')
        backends.append('sox_io')
    return backends


def set_audio_backend(backend: Optional[str]) -> None:
    """Set the backend for I/O operation

    Args:
        backend (str): Name of the backend. One of "sox" or "soundfile",
            based on availability of the system.
    """
    if backend is not None and backend not in list_audio_backends():
        raise RuntimeError(
            f'Backend "{backend}" is not one of '
            f'available backends: {list_audio_backends()}.')

    if backend is None:
        module = no_backend
    elif backend == 'sox':
        module = sox_backend
    elif backend == 'sox_io':
        warnings.warn('"sox_io" backend is currently beta. Function signatures might change.')
        module = sox_io_backend
    elif backend == 'soundfile':
        module = soundfile_backend
    else:
        raise NotImplementedError(f'Unexpected backend "{backend}"')

    for func in ['save', 'load', 'load_wav', 'info']:
        setattr(torchaudio, func, getattr(module, func))


def _init_audio_backend():
    backends = list_audio_backends()
    if 'sox' in backends:
        set_audio_backend('sox')
    elif 'soundfile' in backends:
        set_audio_backend('soundfile')
    else:
        warnings.warn('No audio backend is available.')
        set_audio_backend(None)


def get_audio_backend() -> Optional[str]:
    """Get the name of the current backend"""
    if torchaudio.load == no_backend.load:
        return None
    if torchaudio.load == sox_backend.load:
        return 'sox'
    if torchaudio.load == sox_io_backend.load:
        return 'sox_io'
    if torchaudio.load == soundfile_backend.load:
        return 'soundfile'
    raise ValueError('Unknown backend.')
