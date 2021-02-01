from . import extension
from torchaudio._internal import module_utils as _mod_utils
from torchaudio import (
    compliance,
    datasets,
    kaldi_io,
    utils,
    sox_effects,
    transforms,
)

USE_SOUNDFILE_LEGACY_INTERFACE = None

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

try:
    from .version import __version__, git_version  # noqa: F401
except ImportError:
    pass
