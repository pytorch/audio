from . import (
    sox_utils,
)


if sox_utils.is_sox_available():
    sox_utils.set_verbosity(1)
