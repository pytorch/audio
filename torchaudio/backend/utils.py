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


def set_audio_backend(backend: Optional[str]):
    """Set the backend for I/O operation

    Args:
        backend (Optional[str]): Name of the backend.
            One of ``"sox"``, ``"sox_io"`` or ``"soundfile"`` based on availability
            of the system. If ``None`` is provided the  current backend is unassigned.
    """
    if torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE is not None:
        warnings.warn(
            '"torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE" flag is deprecated and will be removed in 0.9.0. '
            'Please remove the use of flag.'
        )

    if backend is not None and backend not in list_audio_backends():
        raise RuntimeError(
            f'Backend "{backend}" is not one of '
            f'available backends: {list_audio_backends()}.')

    if backend is None:
        module = no_backend
    elif backend == 'sox':
        warnings.warn(
            '"sox" backend is deprecated and will be removed in 0.9.0. '
            'Please use "sox_io" backend.'
        )
        module = sox_backend
    elif backend == 'sox_io':
        module = sox_io_backend
    elif backend == 'soundfile':
        if torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE:
            warnings.warn(
                'The legacy interface of "soundfile" backend is deprecated and will be removed in 0.9.0. '
                'Please migrate to the new interface.'
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
        set_audio_backend('sox_io')
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
