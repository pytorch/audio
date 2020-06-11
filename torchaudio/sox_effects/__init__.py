from torchaudio._internal import module_utils as _mod_utils
from .sox_effects import (
    _init_sox_effects,
    _shutdown_sox_effects,
    effect_names,
    SoxEffect,
    SoxEffectsChain,
)


if _mod_utils.is_module_available('torchaudio._torchaudio'):
    _init_sox_effects()
