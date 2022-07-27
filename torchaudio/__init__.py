from torchaudio import (  # noqa: F401
    _extension,
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
from torchaudio.backend import get_audio_backend, list_audio_backends, set_audio_backend

try:
    from .version import __version__, git_version  # noqa: F401
except ImportError:
    pass

__all__ = [
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
