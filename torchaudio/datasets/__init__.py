from .commonvoice import COMMONVOICE
from .librispeech import LIBRISPEECH
from .speechcommands import SPEECHCOMMANDS
from .utils import bg_iterator, diskcache_iterator
from .vctk import VCTK, VCTK_092
from .gtzan import GTZAN
from .yesno import YESNO
from .ljspeech import LJSPEECH
from .cmuarctic import CMUARCTIC
from .libritts import LIBRITTS

__all__ = (
    "COMMONVOICE",
    "LIBRISPEECH",
    "SPEECHCOMMANDS",
    "VCTK",
    "VCTK_092",
    "YESNO",
    "LJSPEECH",
    "GTZAN",
    "CMUARCTIC",
    "LIBRITTS"
    "diskcache_iterator",
    "bg_iterator",
)
