# NOTE:
# The entire `torchaudio.backend` module is deprecated.
# New things should be added to `torchaudio._backend`.
# Only things related to backward compatibility should be placed here.

from .utils import _init_backend, get_audio_backend, list_audio_backends, set_audio_backend


__all__ = ["_init_backend", "get_audio_backend", "list_audio_backends", "set_audio_backend"]


def __getattr__(name: str):
    if name == "common":
        from . import _common

        return _common

    if name in ["no_backend", "sox_io_backend", "soundfile_backend"]:
        import warnings

        warnings.warn(
            "Torchaudio's I/O functions now support par-call bakcend dispatch. "
            "Importing backend implementation directly is no longer guaranteed to work. "
            "Please use `backend` keyword with load/save/info function, instead of "
            "calling the udnerlying implementation directly.",
            stacklevel=2,
        )

        if name == "sox_io_backend":
            from . import _sox_io_backend

            return _sox_io_backend
        if name == "soundfile_backend":
            from torchaudio._backend import soundfile_backend

            return soundfile_backend

        if name == "no_backend":
            from . import _no_backend

            return _no_backend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
