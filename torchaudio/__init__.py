from . import (  # noqa: F401
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
from .backend.common import AudioMetaData


try:
    from .version import __version__, git_version  # noqa: F401
except ImportError:
    pass


def _is_backend_dispatcher_enabled():
    import os

    return os.getenv("TORCHAUDIO_USE_BACKEND_DISPATCHER", default="1") == "1"


if _is_backend_dispatcher_enabled():
    from ._backend import _init_backend, get_audio_backend, list_audio_backends, set_audio_backend
else:
    from .backend import _init_backend, get_audio_backend, list_audio_backends, set_audio_backend


_init_backend()


__all__ = [
    "AudioMetaData",
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
