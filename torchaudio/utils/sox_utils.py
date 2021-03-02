from typing import List, Dict
from functools import wraps

import torch

from torchaudio._internal import (
    module_utils as _mod_utils,
)

def is_sox_available():
    return torch.ops.torchaudio.is_sox_available()

def requires_sox():
    if is_sox_available():
        def decorator(func):
            return func
    else:
        def decorator(func):
            @wraps(func)
            def wrapped(*args, **kwargs):
                raise RuntimeError(f'{func.__module__}.{func.__name__} requires sox')
            return wrapped
    return decorator

@requires_sox()
def set_seed(seed: int):
    """Set libsox's PRNG

    Args:
        seed (int): seed value. valid range is int32.

    See Also:
        http://sox.sourceforge.net/sox.html
    """
    torch.ops.torchaudio.sox_utils_set_seed(seed)


@requires_sox()
def set_verbosity(verbosity: int):
    """Set libsox's verbosity

    Args:
        verbosity (int): Set verbosity level of libsox.

            * ``1`` failure messages
            * ``2`` warnings
            * ``3`` details of processing
            * ``4``-``6`` increasing levels of debug messages

    See Also:
        http://sox.sourceforge.net/sox.html
    """
    torch.ops.torchaudio.sox_utils_set_verbosity(verbosity)


@requires_sox()
def set_buffer_size(buffer_size: int):
    """Set buffer size for sox effect chain

    Args:
        buffer_size (int): Set the size in bytes of the buffers used for processing audio.

    See Also:
        http://sox.sourceforge.net/sox.html
    """
    torch.ops.torchaudio.sox_utils_set_buffer_size(buffer_size)


@requires_sox()
def set_use_threads(use_threads: bool):
    """Set multithread option for sox effect chain

    Args:
        use_threads (bool): When ``True``, enables ``libsox``'s parallel effects channels processing.
            To use mutlithread, the underlying ``libsox`` has to be compiled with OpenMP support.

    See Also:
        http://sox.sourceforge.net/sox.html
    """
    torch.ops.torchaudio.sox_utils_set_use_threads(use_threads)


@requires_sox()
def list_effects() -> Dict[str, str]:
    """List the available sox effect names

    Returns:
        Dict[str, str]: Mapping from ``effect name`` to ``usage``
    """
    return dict(torch.ops.torchaudio.sox_utils_list_effects())


@requires_sox()
def list_read_formats() -> List[str]:
    """List the supported audio formats for read

    Returns:
        List[str]: List of supported audio formats
    """
    return torch.ops.torchaudio.sox_utils_list_read_formats()


@requires_sox()
def list_write_formats() -> List[str]:
    """List the supported audio formats for write

    Returns:
        List[str]: List of supported audio formats
    """
    return torch.ops.torchaudio.sox_utils_list_write_formats()
