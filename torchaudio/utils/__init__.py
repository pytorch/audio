from . import (
    sox_utils,
)
from torchaudio._internal.module_utils import is_sox_available


if is_sox_available():
    sox_utils.set_verbosity(1)
