from functools import wraps

import platform
import torch

from . import _soundfile_backend, _sox_backend


_audio_backend = "soundfile" if platform.system() == "Windows" else "sox"
_audio_backends = {"sox": _sox_backend, "soundfile": _soundfile_backend}


def set_audio_backend(backend):
    """
    Specifies the package used to load.
    Args:
        backend (string): Name of the backend. One of {}.
    """.format(_audio_backends.keys())
    global _audio_backend
    if backend not in _audio_backends:
        raise ValueError(
            "Invalid backend '{}'. Options are {}.".format(backend, _audio_backends.keys())
        )
    _audio_backend = backend


def get_audio_backend():
    """
    Gets the name of the package used to load.
    """
    return _audio_backend


def _get_audio_backend_module():
    """
    Gets the module backend to load.
    """
    backend = get_audio_backend()
    return _audio_backends[backend]


def _audio_backend_guard(backends):

    if isinstance(backends, str):
        backends = [backends]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if get_audio_backend() not in backends:
                raise RuntimeError("Function {} requires backend to be one of {}.".format(func.__name__, backends))
            return func(*args, **kwargs)
        return wrapper

    return decorator


def check_input(src):
    if not torch.is_tensor(src):
        raise TypeError('Expected a tensor, got %s' % type(src))
    if src.is_cuda:
        raise TypeError('Expected a CPU based tensor, got %s' % type(src))
