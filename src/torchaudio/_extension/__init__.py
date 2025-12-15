import logging
import os
import sys

import torch

from torchaudio._internal.module_utils import fail_with_message, no_op

from .utils import _check_cuda_version, _init_dll_path, _load_lib

_LG = logging.getLogger(__name__)


# Note:
# `_check_cuda_version` is not meant to be used by regular users.
# Builder uses it for debugging purpose, so we export it.
# https://github.com/pytorch/builder/blob/e2e4542b8eb0bdf491214451a1a4128bd606cce2/test/smoke_test/smoke_test.py#L80
__all__ = [
    "_check_cuda_version",
    "_IS_TORCHAUDIO_EXT_AVAILABLE",
]


if os.name == "nt":  # and (3, 8) <= sys.version_info < (3, 9):
    _init_dll_path()

# When the extension module is built, we initialize it.
# In case of an error, we do not catch the failure as it suggests there is something
# wrong with the installation.
_IS_TORCHAUDIO_EXT_AVAILABLE = _load_lib("_torchaudio")
assert _IS_TORCHAUDIO_EXT_AVAILABLE
_IS_ALIGN_AVAILABLE = False
if _IS_TORCHAUDIO_EXT_AVAILABLE:
    if not _load_lib("libtorchaudio"):
        assert 0  # unreachable

    _check_cuda_version()
    _IS_ALIGN_AVAILABLE = torch.ops._torchaudio.is_align_available()

fail_if_no_align = (
    no_op
    if _IS_ALIGN_AVAILABLE
    else fail_with_message(
        "Requires alignment extension, but TorchAudio is not compiled with it. \
        Please build TorchAudio with alignment support."
    )
)
