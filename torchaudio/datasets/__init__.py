from .commonvoice import COMMONVOICE
from .librispeech import LIBRISPEECH
from .speechcommands import SPEECHCOMMANDS
from .utils import bg_iterator, diskcache_iterator
from .dr_vctk import DR_VCTK
from .vctk import VCTK, VCTK_092
from .gtzan import GTZAN
from .yesno import YESNO
from .ljspeech import LJSPEECH
from .cmuarctic import CMUARCTIC
from .cmudict import CMUDict
from .libritts import LIBRITTS
from .tedlium import TEDLIUM


__all__ = [
    "COMMONVOICE",
    "LIBRISPEECH",
    "SPEECHCOMMANDS",
    "DR_VCTK"
    "VCTK",
    "VCTK_092",
    "YESNO",
    "LJSPEECH",
    "GTZAN",
    "CMUARCTIC",
    "CMUDict",
    "LIBRITTS",
    "diskcache_iterator",
    "bg_iterator",
    "TEDLIUM",
]
