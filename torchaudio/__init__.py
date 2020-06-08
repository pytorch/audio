from torchaudio._internal import module_utils as _mod_utils
from torchaudio import (
    compliance,
    datasets,
    kaldi_io,
    sox_effects,
    transforms
)
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
    init_sox_effects,
    shutdown_sox_effects,
)

try:
    from .version import __version__, git_version  # noqa: F401
except ImportError:
    pass


@_mod_utils.depricate("Use `torchaudio.sox_effects.init_sox_effects`.")
def initialize_sox():
    init_sox_effects()


@_mod_utils.depricate("Use `torchaudio.sox_effects.shutdown_sox`.")
def shutdown_sox():
    shutdown_sox_effects()
