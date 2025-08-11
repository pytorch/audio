"""Module to change the configuration of libsox, which is used by I/O functions like
:py:mod:`~torchaudio.backend.sox_io_backend` and :py:mod:`~torchaudio.sox_effects`.

.. warning::
    Starting with version 2.8, we are refactoring TorchAudio to transition it
    into a maintenance phase. As a result:

    - Some APIs are deprecated in 2.8 and will be removed in 2.9.
    - The decoding and encoding capabilities of PyTorch for both audio and video
      are being consolidated into TorchCodec.

    Please see https://github.com/pytorch/audio/issues/3902 for more information.
"""

from typing import Dict, List

import torchaudio

sox_ext = torchaudio._extension.lazy_import_sox_ext()

from torchaudio._internal.module_utils import dropping_support

@dropping_support
def set_seed(seed: int):
    """Set libsox's PRNG

    Args:
        seed (int): seed value. valid range is int32.

    See Also:
        http://sox.sourceforge.net/sox.html
    """
    sox_ext.set_seed(seed)


@dropping_support
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
    sox_ext.set_verbosity(verbosity)


@dropping_support
def set_buffer_size(buffer_size: int):
    """Set buffer size for sox effect chain

    Args:
        buffer_size (int): Set the size in bytes of the buffers used for processing audio.

    See Also:
        http://sox.sourceforge.net/sox.html
    """
    sox_ext.set_buffer_size(buffer_size)


@dropping_support
def set_use_threads(use_threads: bool):
    """Set multithread option for sox effect chain

    Args:
        use_threads (bool): When ``True``, enables ``libsox``'s parallel effects channels processing.
            To use mutlithread, the underlying ``libsox`` has to be compiled with OpenMP support.

    See Also:
        http://sox.sourceforge.net/sox.html
    """
    sox_ext.set_use_threads(use_threads)


@dropping_support
def list_effects() -> Dict[str, str]:
    """List the available sox effect names

    Returns:
        Dict[str, str]: Mapping from ``effect name`` to ``usage``
    """
    return dict(sox_ext.list_effects())


@dropping_support
def list_read_formats() -> List[str]:
    """List the supported audio formats for read

    Returns:
        List[str]: List of supported audio formats
    """
    return sox_ext.list_read_formats()


@dropping_support
def list_write_formats() -> List[str]:
    """List the supported audio formats for write

    Returns:
        List[str]: List of supported audio formats
    """
    return sox_ext.list_write_formats()


@dropping_support
def get_buffer_size() -> int:
    """Get buffer size for sox effect chain

    Returns:
        int: size in bytes of buffers used for processing audio.
    """
    return sox_ext.get_buffer_size()
