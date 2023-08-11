import warnings
from typing import List, Optional

import torchaudio

from . import utils


# TODO: Once legacy global backend is removed, move this to torchaudio.__init__
def _init_backend():
    torchaudio.info = utils.get_info_func()
    torchaudio.load = utils.get_load_func()
    torchaudio.save = utils.get_save_func()


def list_audio_backends() -> List[str]:
    return list(utils.get_available_backends().keys())


# Temporary until global backend is removed
def get_audio_backend() -> Optional[str]:
    warnings.warn("I/O Dispatcher is enabled. There is no global audio backend.", stacklevel=2)
    return None


# Temporary until global backend is removed
def set_audio_backend(_: Optional[str]):
    warnings.warn("I/O Dispatcher is enabled. set_audio_backend is a no-op", stacklevel=2)
