from .commonvoice import COMMONVOICE
from .librispeech import LIBRISPEECH
from .speechcommands import SPEECHCOMMANDS
from .utils import bg_iterator, diskcache_iterator
from .vctk import VCTK
from .gtzan import GTZAN
from .yesno import YESNO
from .ljspeech import LJSPEECH
from .cmu_arctic import CMU_ARCTIC

__all__ = (
    "COMMONVOICE",
    "LIBRISPEECH",
    "SPEECHCOMMANDS",
    "VCTK",
    "YESNO",
    "LJSPEECH",
    "GTZAN",
    "CMU_ARCTIC",
    "diskcache_iterator",
    "bg_iterator",
)
