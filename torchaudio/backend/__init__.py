# flake8: noqa
import torchaudio

from . import utils
from .utils import _is_backend_dispatcher_enabled, get_audio_backend, list_audio_backends, set_audio_backend

if _is_backend_dispatcher_enabled():
    from torchaudio._backend.utils import get_info_func, get_load_func, get_save_func

    torchaudio.info = get_info_func()
    torchaudio.load = get_load_func()
    torchaudio.save = get_save_func()
else:
    utils._init_audio_backend()
