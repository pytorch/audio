# Initialize extension and backend first
from . import _extension  # noqa  # usort: skip
from ._backend import (  # noqa  # usort: skip
    AudioMetaData,
    get_audio_backend,
    info,
    list_audio_backends,
    load,
    save,
    set_audio_backend,
)

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
