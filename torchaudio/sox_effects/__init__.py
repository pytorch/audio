from torchaudio.utils import sox_utils
from .sox_effects import (
    init_sox_effects,
    shutdown_sox_effects,
    effect_names,
    apply_effects_tensor,
    apply_effects_file,
)


if sox_utils.is_sox_available():
    import atexit
    init_sox_effects()
    atexit.register(shutdown_sox_effects)

__all__ = [
    'init_sox_effects',
    'shutdown_sox_effects',
    'effect_names',
    'apply_effects_tensor',
    'apply_effects_file',
]
