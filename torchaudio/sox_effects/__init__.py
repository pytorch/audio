from torchaudio._internal import module_utils as _mod_utils
from .sox_effects import (
    init_sox_effects,
    shutdown_sox_effects,
    effect_names,
    apply_effects_tensor,
    apply_effects_file,
    SoxEffect,
    SoxEffectsChain,
)


if _mod_utils.is_module_available("torchaudio._torchaudio"):
    import atexit

    init_sox_effects()
    atexit.register(shutdown_sox_effects)
