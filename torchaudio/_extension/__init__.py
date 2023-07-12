import logging
import os
import sys

from torchaudio._internal.module_utils import fail_with_message, is_module_available, no_op

from .utils import _check_cuda_version, _fail_since_no_ffmpeg, _init_dll_path, _init_ffmpeg, _init_sox, _load_lib

_LG = logging.getLogger(__name__)


# Note:
# `_check_cuda_version` is not meant to be used by regular users.
# Builder uses it for debugging purpose, so we export it.
# https://github.com/pytorch/builder/blob/e2e4542b8eb0bdf491214451a1a4128bd606cce2/test/smoke_test/smoke_test.py#L80
__all__ = [
    "fail_if_no_sox",
    "fail_if_no_ffmpeg",
    "_check_cuda_version",
    "_IS_TORCHAUDIO_EXT_AVAILABLE",
    "_IS_RIR_AVAILABLE",
    "_SOX_INITIALIZED",
    "_FFMPEG_EXT",
]


if os.name == "nt" and (3, 8) <= sys.version_info < (3, 9):
    _init_dll_path()


# When the extension module is built, we initialize it.
# In case of an error, we do not catch the failure as it suggests there is something
# wrong with the installation.
_IS_TORCHAUDIO_EXT_AVAILABLE = is_module_available("torchaudio.lib._torchaudio")
# RIR features are implemented in _torchaudio extension, but they can be individually
# turned on/off at build time. Available means that _torchaudio is loaded properly, and
# RIR features are found there.
_IS_RIR_AVAILABLE = False
_IS_ALIGN_AVAILABLE = False
if _IS_TORCHAUDIO_EXT_AVAILABLE:
    _load_lib("libtorchaudio")

    import torchaudio.lib._torchaudio  # noqa

    _check_cuda_version()
    _IS_RIR_AVAILABLE = torchaudio.lib._torchaudio.is_rir_available()
    _IS_ALIGN_AVAILABLE = torchaudio.lib._torchaudio.is_align_available()


# Similar to libtorchaudio, sox-related features should be importable when present.
#
# Note: This will be change in the future when sox is dynamically linked.
# At that point, this initialization should handle the case where
# sox integration is built but libsox is not found.
_SOX_INITIALIZED = False
if is_module_available("torchaudio.lib._torchaudio_sox"):
    _init_sox()
    _SOX_INITIALIZED = True


# Initialize FFmpeg-related features
_FFMPEG_EXT = None
if _IS_TORCHAUDIO_EXT_AVAILABLE:
    try:
        _FFMPEG_EXT = _init_ffmpeg()
    except Exception:
        # The initialization of FFmpeg extension will fail if supported FFmpeg
        # libraries are not found in the system.
        # Since the rest of the torchaudio works without it, we do not report the
        # error here.
        # The error will be raised when user code attempts to use these features.
        _LG.debug("Failed to initialize ffmpeg bindings", exc_info=True)


fail_if_no_sox = (
    no_op
    if _SOX_INITIALIZED
    else fail_with_message(
        "requires sox extension, but TorchAudio is not compiled with it. Please build TorchAudio with libsox support."
    )
)

fail_if_no_ffmpeg = _fail_since_no_ffmpeg if _FFMPEG_EXT is None else no_op

fail_if_no_rir = (
    no_op
    if _IS_RIR_AVAILABLE
    else fail_with_message(
        "requires RIR extension, but TorchAudio is not compiled with it. Please build TorchAudio with RIR support."
    )
)

fail_if_no_align = (
    no_op
    if _IS_ALIGN_AVAILABLE
    else fail_with_message(
        "Requires alignment extension, but TorchAudio is not compiled with it. \
        Please build TorchAudio with alignment support."
    )
)
