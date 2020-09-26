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
    _soundfile_backend,
)

__all__ = [
    'list_audio_backends',
    'get_audio_backend',
    'set_audio_backend',
    'use_soundfile_legacy_interface',
]


def list_audio_backends() -> List[str]:
    """List available backends

    Returns:
        List[str]: The list of available backends.
    """
    backends = []
    if is_module_available('soundfile'):
        backends.append('soundfile')
    if is_module_available('torchaudio._torchaudio'):
        backends.append('sox')
        backends.append('sox_io')
    return backends


_USE_SOUNDFILE_LEGACY_INTERFACE = True


def use_soundfile_legacy_interface(value: bool):
    """Switch soundfile backend interface.
    """
    global _USE_SOUNDFILE_LEGACY_INTERFACE
    _USE_SOUNDFILE_LEGACY_INTERFACE = value


def set_audio_backend(backend: Optional[str]):
    """Set the backend for I/O operation

    Args:
        backend (Optional[str]): Name of the backend.
            One of ``"sox"``, ``"sox_io"`` or ``"soundfile"`` based on availability
            of the system. If ``None`` is provided the  current backend is unassigned.
    """
    if backend is not None and backend not in list_audio_backends():
        raise RuntimeError(
            f'Backend "{backend}" is not one of '
            f'available backends: {list_audio_backends()}.')

    if backend is None:
        module = no_backend
    elif backend == 'sox':
        warnings.warn(
            '"sox" backend is being deprecated. '
            'The default backend will be changed to "sox_io" backend in 0.8.0 and '
            '"sox" backend will be removed in 0.9.0. Please migrate to "sox_io" backend. '
            'Please refer to https://github.com/pytorch/audio/issues/903 for the detail.')
        module = sox_backend
    elif backend == 'sox_io':
        module = sox_io_backend
    elif backend == 'soundfile':
        if _USE_SOUNDFILE_LEGACY_INTERFACE:
            warnings.warn(
                'The interface of "soundfile" backend is planned to change in 0.8.0 to '
                'match that of "sox_io" backend and the current interface will be removed in 0.9.0. '
                'To use the new interface, do '
                '`torchaudio.backend.utils.use_soundfile_legacy_interface(False)` '
                'before setting the backend to "soundfile". '
                'Please refer to https://github.com/pytorch/audio/issues/903 for the detail.'
            )
            module = soundfile_backend
        else:
            module = _soundfile_backend
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
    """Get the name of the current backend

    Returns:
        Optional[str]: The name of the current backend or ``None`` if no backend is assigned.
    """
    if torchaudio.load == no_backend.load:
        return None
    if torchaudio.load == sox_backend.load:
        return 'sox'
    if torchaudio.load == sox_io_backend.load:
        return 'sox_io'
    if torchaudio.load in [soundfile_backend.load, _soundfile_backend.load]:
        return 'soundfile'
    raise ValueError('Unknown backend.')
