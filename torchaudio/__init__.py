from . import extension
from torchaudio._internal import module_utils as _mod_utils
from torchaudio import (
    compliance,
    datasets,
    kaldi_io,
    utils,
    sox_effects,
    transforms
)

USE_SOUNDFILE_LEGACY_INTERFACE = True

from torchaudio.backend import (
    list_audio_backends,
    get_audio_backend,
    set_audio_backend,
    save_encinfo,
    sox_signalinfo_t,
    sox_encodinginfo_t,
    get_sox_option_t,
    get_sox_encoding_t,
    get_sox_bool,
    SignalInfo,
    EncodingInfo,
)
from torchaudio.sox_effects import (
    init_sox_effects as _init_sox_effects,
    shutdown_sox_effects as _shutdown_sox_effects,
)

try:
    from .version import __version__, git_version  # noqa: F401
except ImportError:
    pass


@_mod_utils.deprecated(
    "Please remove the function call to initialize_sox. "
    "Resource initialization is now automatically handled.",
    "0.8.0")
def initialize_sox():
    """Initialize sox effects.

    This function is deprecated. See :py:func:`torchaudio.sox_effects.init_sox_effects`
    """
    _init_sox_effects()


@_mod_utils.deprecated(
    "Please remove the function call to torchaudio.shutdown_sox. "
    "Resource clean up is now automatically handled. "
    "In the unlikely event that you need to manually shutdown sox, "
    "please use torchaudio.sox_effects.shutdown_sox_effects.",
    "0.8.0")
def shutdown_sox():
    """Shutdown sox effects.

    This function is deprecated. See :py:func:`torchaudio.sox_effects.shutdown_sox_effects`
    """
    _shutdown_sox_effects()
