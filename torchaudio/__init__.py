# Initialize extension and backend first
from . import (  # noqa  # usort: skip
    _extension,
    _backend,
)
from . import (  # noqa: F401
    backend,  # For BC
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
from ._backend import AudioMetaData, get_audio_backend, info, list_audio_backends, load, save, set_audio_backend

try:
    from .version import __version__, git_version  # noqa: F401
except ImportError:
    pass


__all__ = [
    "AudioMetaData",
    "load",
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
