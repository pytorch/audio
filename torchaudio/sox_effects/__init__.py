import torchaudio

from .sox_effects import apply_effects_file, apply_effects_tensor, effect_names, init_sox_effects, shutdown_sox_effects


if torchaudio._extension._SOX_INITIALIZED:
    import atexit

    init_sox_effects()
    atexit.register(shutdown_sox_effects)

__all__ = [
    "init_sox_effects",
    "shutdown_sox_effects",
    "effect_names",
    "apply_effects_tensor",
    "apply_effects_file",
]
