# flake8: noqa
from . import utils
from .utils import get_audio_backend, list_audio_backends, set_audio_backend


utils._init_audio_backend()
