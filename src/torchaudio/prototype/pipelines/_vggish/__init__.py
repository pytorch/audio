from ._vggish_pipeline import VGGISH as _VGGISH, VGGishBundle
from torchaudio._internal.module_utils import dropping_const_support


VGGISH = dropping_const_support(_VGGISH, "VGGISH")

__all__ = ["VGGISH", "VGGishBundle"]
