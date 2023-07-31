"""Module to change the configuration of libsox, which is used by I/O functions like
:py:mod:`~torchaudio.backend.sox_io_backend` and :py:mod:`~torchaudio.sox_effects`.
"""

from typing import Dict, List

import torchaudio


@torchaudio._extension.fail_if_no_sox
def set_seed(seed: int):
    """Set libsox's PRNG

    Args:
        seed (int): seed value. valid range is int32.

    See Also:
        http://sox.sourceforge.net/sox.html
    """
    torchaudio.lib._torchaudio_sox.set_seed(seed)


@torchaudio._extension.fail_if_no_sox
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
    torchaudio.lib._torchaudio_sox.set_verbosity(verbosity)


@torchaudio._extension.fail_if_no_sox
def set_buffer_size(buffer_size: int):
    """Set buffer size for sox effect chain

    Args:
        buffer_size (int): Set the size in bytes of the buffers used for processing audio.

    See Also:
        http://sox.sourceforge.net/sox.html
    """
    torchaudio.lib._torchaudio_sox.set_buffer_size(buffer_size)


@torchaudio._extension.fail_if_no_sox
def set_use_threads(use_threads: bool):
    """Set multithread option for sox effect chain

    Args:
        use_threads (bool): When ``True``, enables ``libsox``'s parallel effects channels processing.
            To use mutlithread, the underlying ``libsox`` has to be compiled with OpenMP support.

    See Also:
        http://sox.sourceforge.net/sox.html
    """
    torchaudio.lib._torchaudio_sox.set_use_threads(use_threads)


@torchaudio._extension.fail_if_no_sox
def list_effects() -> Dict[str, str]:
    """List the available sox effect names

    Returns:
        Dict[str, str]: Mapping from ``effect name`` to ``usage``
    """
    return dict(torchaudio.lib._torchaudio_sox.list_effects())


@torchaudio._extension.fail_if_no_sox
def list_read_formats() -> List[str]:
    """List the supported audio formats for read

    Returns:
        List[str]: List of supported audio formats
    """
    return torchaudio.lib._torchaudio_sox.list_read_formats()


@torchaudio._extension.fail_if_no_sox
def list_write_formats() -> List[str]:
    """List the supported audio formats for write

    Returns:
        List[str]: List of supported audio formats
    """
    return torchaudio.lib._torchaudio_sox.list_write_formats()


@torchaudio._extension.fail_if_no_sox
def get_buffer_size() -> int:
    """Get buffer size for sox effect chain

    Returns:
        int: size in bytes of buffers used for processing audio.
    """
    return torchaudio.lib._torchaudio_sox.get_buffer_size()
