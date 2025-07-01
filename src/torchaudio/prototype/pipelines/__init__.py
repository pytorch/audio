from ._vggish import VGGISH, VGGishBundle
from .hifigan_pipeline import HIFIGAN_VOCODER_V3_LJSPEECH as _HIFIGAN_VOCODER_V3_LJSPEECH, HiFiGANVocoderBundle
from .rnnt_pipeline import (
    EMFORMER_RNNT_BASE_MUSTC as _EMFORMER_RNNT_BASE_MUSTC,
    EMFORMER_RNNT_BASE_TEDLIUM3 as _EMFORMER_RNNT_BASE_TEDLIUM3
)
from torchaudio._internal.module_utils import dropping_const_support

EMFORMER_RNNT_BASE_MUSTC = dropping_const_support(_EMFORMER_RNNT_BASE_MUSTC)
EMFORMER_RNNT_BASE_TEDLIUM3 = dropping_const_support(_EMFORMER_RNNT_BASE_TEDLIUM3)
HIFIGAN_VOCODER_V3_LJSPEECH = dropping_const_support(_HIFIGAN_VOCODER_V3_LJSPEECH)


__all__ = [
    "EMFORMER_RNNT_BASE_MUSTC",
    "EMFORMER_RNNT_BASE_TEDLIUM3",
    "HIFIGAN_VOCODER_V3_LJSPEECH",
    "HiFiGANVocoderBundle",
    "VGGISH",
    "VGGishBundle",
]
