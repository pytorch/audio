from torchaudio._internal.module_utils import dropping_io_support, dropping_class_io_support

# Initialize extension and backend first
from . import _extension  # noqa  # usort: skip
from ._backend import (  # noqa  # usort: skip
    AudioMetaData as _AudioMetaData,
    get_audio_backend as _get_audio_backend,
    info as _info,
    list_audio_backends as _list_audio_backends,
    load,
    save,
    set_audio_backend as _set_audio_backend,
)
from ._torchcodec import load_with_torchcodec, save_with_torchcodec

AudioMetaData = dropping_class_io_support(_AudioMetaData)
get_audio_backend = dropping_io_support(_get_audio_backend)
info = dropping_io_support(_info)
list_audio_backends = dropping_io_support(_list_audio_backends)
set_audio_backend = dropping_io_support(_set_audio_backend)

from . import (  # noqa: F401
    compliance,
    datasets,
    functional,
    io,
    kaldi_io,
    models,
    pipelines,
    sox_effects,
    transforms,
    utils,
)

# For BC
from . import backend  # noqa # usort: skip

try:
    from .version import __version__, git_version  # noqa: F401
except ImportError:
    pass


__all__ = [
    "AudioMetaData",
    "load",
    "load_with_torchcodec",
    "save_with_torchcodec",
    "info",
    "save",
    "io",
    "compliance",
    "datasets",
    "functional",
    "models",
    "pipelines",
    "kaldi_io",
    "utils",
    "sox_effects",
    "transforms",
    "list_audio_backends",
    "get_audio_backend",
    "set_audio_backend",
]
