from typing import List, Optional

from torchaudio._internal.module_utils import deprecated

from . import utils
from .common import AudioMetaData

__all__ = [
    "AudioMetaData",
    "load",
    "info",
    "save",
    "list_audio_backends",
    "get_audio_backend",
    "set_audio_backend",
]


info = utils.get_info_func()
load = utils.get_load_func()
save = utils.get_save_func()


def list_audio_backends() -> List[str]:
    """List available backends

    Returns:
        list of str: The list of available backends.

        The possible values are; ``"ffmpeg"``, ``"sox"`` and ``"soundfile"``.
    """

    return list(utils.get_available_backends().keys())


# Temporary until global backend is removed
@deprecated("With dispatcher enabled, this function is no-op. You can remove the function call.")
def get_audio_backend() -> Optional[str]:
    """Get the name of the current global backend

    Returns:
        str or None:
            If dispatcher mode is enabled, returns ``None`` otherwise,
            the name of current backend or ``None`` (no backend is set).
    """
    return None


# Temporary until global backend is removed
@deprecated("With dispatcher enabled, this function is no-op. You can remove the function call.")
def set_audio_backend(backend: Optional[str]):  # noqa
    """Set the global backend.

    This is a no-op when dispatcher mode is enabled.

    Args:
        backend (str or None): Name of the backend.
            One of ``"sox_io"`` or ``"soundfile"`` based on availability
            of the system. If ``None`` is provided the  current backend is unassigned.
    """
    pass
