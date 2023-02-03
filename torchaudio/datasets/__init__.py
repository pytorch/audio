from .cmuarctic import CMUARCTIC
from .cmudict import CMUDict
from .commonvoice import COMMONVOICE
from .dr_vctk import DR_VCTK
from .fluentcommands import FluentSpeechCommands
from .gtzan import GTZAN
from .iemocap import IEMOCAP
from .librilight_limited import LibriLightLimited
from .librimix import LibriMix
from .librispeech import LIBRISPEECH
from .librispeech_biasing import LibriSpeechBiasing
from .libritts import LIBRITTS
from .ljspeech import LJSPEECH
from .musdb_hq import MUSDB_HQ
from .quesst14 import QUESST14
from .snips import Snips
from .speechcommands import SPEECHCOMMANDS
from .tedlium import TEDLIUM
from .vctk import VCTK_092
from .voxceleb1 import VoxCeleb1Identification, VoxCeleb1Verification
from .yesno import YESNO


__all__ = [
    "COMMONVOICE",
    "LIBRISPEECH",
    "LibriSpeechBiasing",
    "LibriLightLimited",
    "SPEECHCOMMANDS",
    "VCTK_092",
    "DR_VCTK",
    "YESNO",
    "LJSPEECH",
    "GTZAN",
    "CMUARCTIC",
    "CMUDict",
    "LibriMix",
    "LIBRITTS",
    "TEDLIUM",
    "QUESST14",
    "MUSDB_HQ",
    "FluentSpeechCommands",
    "VoxCeleb1Identification",
    "VoxCeleb1Verification",
    "IEMOCAP",
    "Snips",
]
