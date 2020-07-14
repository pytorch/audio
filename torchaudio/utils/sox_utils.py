from typing import List, Dict

import torch

from torchaudio._internal import (
    module_utils as _mod_utils,
)


@_mod_utils.requires_module('torchaudio._torchaudio')
def set_seed(seed: int):
    """Set libsox's PRNG

    Args:
        seed: seed value. valid range is int32.

    See Also:
        http://sox.sourceforge.net/sox.html
    """
    torch.ops.torchaudio.sox_utils_set_seed(seed)


@_mod_utils.requires_module('torchaudio._torchaudio')
def set_verbosity(verbosity: int):
    """Set libsox's verbosity

    Args:
        verbosity: Set verbosity level of libsox.
            1: failure messages
            2: warnings
            3: details of processing
            4-6: increasing levels of debug messages

    See Also:
        http://sox.sourceforge.net/sox.html
    """
    torch.ops.torchaudio.sox_utils_set_verbosity(verbosity)


@_mod_utils.requires_module('torchaudio._torchaudio')
def set_buffer_size(buffer_size: int):
    """Set buffer size for sox effect chain

    Args:
        buffer_size: Set the size in bytes of the buffers used for processing audio.

    See Also:
        http://sox.sourceforge.net/sox.html
    """
    torch.ops.torchaudio.sox_utils_set_buffer_size(buffer_size)


@_mod_utils.requires_module('torchaudio._torchaudio')
def set_use_threads(use_threads: bool):
    """Set multithread option for sox effect chain

    Args:
        use_threads: When True, enables libsox's parallel effects channels processing.
            To use mutlithread, the underlying libsox has to be compiled with OpenMP support.

    See Also:
        http://sox.sourceforge.net/sox.html
    """
    torch.ops.torchaudio.sox_utils_set_use_threads(use_threads)


@_mod_utils.requires_module('torchaudio._torchaudio')
def list_effects() -> Dict[str, str]:
    """List the available sox effect names

    Returns:
        Mapping from "effect name" to "usage"
    """
    return dict(torch.ops.torchaudio.sox_utils_list_effects())


@_mod_utils.requires_module('torchaudio._torchaudio')
def list_formats() -> List[str]:
    """List the supported audio formats

    Returns: list[str]
        List of supported audio formats
    """
    return torch.ops.torchaudio.sox_utils_list_formats()
