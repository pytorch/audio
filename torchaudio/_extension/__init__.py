import logging
import os
import sys

from torchaudio._internal.module_utils import fail_with_message, is_module_available, no_op

from .utils import (
    _check_cuda_version,
    _fail_since_no_ffmpeg,
    _fail_since_no_sox,
    _init_dll_path,
    _init_ffmpeg,
    _init_sox,
    _load_lib,
)

_LG = logging.getLogger(__name__)


# Note:
# `_check_cuda_version` is not meant to be used by regular users.
# Builder uses it for debugging purpose, so we export it.
# https://github.com/pytorch/builder/blob/e2e4542b8eb0bdf491214451a1a4128bd606cce2/test/smoke_test/smoke_test.py#L80
__all__ = [
    "fail_if_no_kaldi",
    "fail_if_no_sox",
    "fail_if_no_ffmpeg",
    "_check_cuda_version",
    "_IS_TORCHAUDIO_EXT_AVAILABLE",
    "_IS_KALDI_AVAILABLE",
    "_IS_RIR_AVAILABLE",
    "_SOX_INITIALIZED",
    "_FFMPEG_INITIALIZED",
]


if os.name == "nt" and (3, 8) <= sys.version_info < (3, 9):
    _init_dll_path()


# When the extension module is built, we initialize it.
# In case of an error, we do not catch the failure as it suggests there is something
# wrong with the installation.
_IS_TORCHAUDIO_EXT_AVAILABLE = is_module_available("torchaudio.lib._torchaudio")
# Kaldi and RIR features are implemented in _torchaudio extension, but they can be individually
# turned on/off at build time. Available means that _torchaudio is loaded properly, and
# Kaldi or RIR features are found there.
_IS_RIR_AVAILABLE = False
_IS_KALDI_AVAILABLE = False
_IS_ALIGN_AVAILABLE = False
if _IS_TORCHAUDIO_EXT_AVAILABLE:
    _load_lib("libtorchaudio")

    import torchaudio.lib._torchaudio  # noqa

    _check_cuda_version()
    _IS_RIR_AVAILABLE = torchaudio.lib._torchaudio.is_rir_available()
    _IS_KALDI_AVAILABLE = torchaudio.lib._torchaudio.is_kaldi_available()
    _IS_ALIGN_AVAILABLE = torchaudio.lib._torchaudio.is_align_available()


# Initialize libsox-related features
_SOX_INITIALIZED = False
if is_module_available("torchaudio.lib._torchaudio_sox"):
    try:
        _init_sox()
        _SOX_INITIALIZED = True
    except Exception:
        _LG.debug("Failed to initialize libsox bindings", exc_info=True)


# Initialize FFmpeg-related features
_FFMPEG_INITIALIZED = False
if is_module_available("torchaudio.lib._torchaudio_ffmpeg"):
    try:
        _init_ffmpeg()
        _FFMPEG_INITIALIZED = True
    except Exception:
        # The initialization of FFmpeg extension will fail if supported FFmpeg
        # libraries are not found in the system.
        # Since the rest of the torchaudio works without it, we do not report the
        # error here.
        # The error will be raised when user code attempts to use these features.
        _LG.debug("Failed to initialize ffmpeg bindings", exc_info=True)


fail_if_no_kaldi = (
    no_op
    if _IS_KALDI_AVAILABLE
    else fail_with_message(
        "requires kaldi extension, but TorchAudio is not compiled with it. Please build TorchAudio with kaldi support."
    )
)
fail_if_no_sox = no_op if _SOX_INITIALIZED else _fail_since_no_sox
fail_if_no_ffmpeg = no_op if _FFMPEG_INITIALIZED else _fail_since_no_ffmpeg

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
