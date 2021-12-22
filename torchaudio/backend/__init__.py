# flake8: noqa
from . import utils
from .utils import (
    list_audio_backends,
    get_audio_backend,
    set_audio_backend,
)


utils._init_audio_backend()
