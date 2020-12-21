from .extension import (
    _init_extension,
    _init_transducer_extension,
)

_init_extension()
_init_transducer_extension()

del _init_extension
