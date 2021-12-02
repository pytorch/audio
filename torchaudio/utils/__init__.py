from torchaudio._internal import module_utils as _mod_utils

from . import (
    sox_utils,
)


if _mod_utils.is_sox_available():
    sox_utils.set_verbosity(1)
