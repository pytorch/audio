from .extension import (
    _init_extension,
)

try:
    from . import fb  # noqa
except Exception:
    pass

_init_extension()

del _init_extension
