from ._vggish import VGGISH as _VGGISH, VGGishBundle as _VGGishBundle
from .hifigan_pipeline import (
    HIFIGAN_VOCODER_V3_LJSPEECH as _HIFIGAN_VOCODER_V3_LJSPEECH,
    HiFiGANVocoderBundle as _HiFiGANVocoderBundle
)
from .rnnt_pipeline import (
    EMFORMER_RNNT_BASE_MUSTC as _EMFORMER_RNNT_BASE_MUSTC,
    EMFORMER_RNNT_BASE_TEDLIUM3 as _EMFORMER_RNNT_BASE_TEDLIUM3
)
from torchaudio._internal.module_utils import dropping_const_support, dropping_class_support

EMFORMER_RNNT_BASE_MUSTC = dropping_const_support(_EMFORMER_RNNT_BASE_MUSTC, name="EMFORMER_RNNT_BASE_MUSTC")
EMFORMER_RNNT_BASE_TEDLIUM3 = dropping_const_support(_EMFORMER_RNNT_BASE_TEDLIUM3, name="EMFORMER_RNNT_BASE_TEDLIUM3")
HIFIGAN_VOCODER_V3_LJSPEECH = dropping_const_support(_HIFIGAN_VOCODER_V3_LJSPEECH, name="HIFIGAN_VOCODER_V3_LJSPEECH")
HiFiGANVocoderBundle = dropping_class_support(_HiFiGANVocoderBundle)
VGGISH = dropping_const_support(_VGGISH, name="VGGISH")
VGGishBundle = dropping_class_support(_VGGishBundle)

__all__ = [
    "EMFORMER_RNNT_BASE_MUSTC",
    "EMFORMER_RNNT_BASE_TEDLIUM3",
    "HIFIGAN_VOCODER_V3_LJSPEECH",
    "HiFiGANVocoderBundle",
    "VGGISH",
    "VGGishBundle",
]
