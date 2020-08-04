from . import sox_utils

from torchaudio._internal import module_utils as _mod_utils


if _mod_utils.is_module_available("torchaudio._torchaudio"):
    sox_utils.set_verbosity(1)
