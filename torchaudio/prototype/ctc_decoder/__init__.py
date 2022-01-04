import torchaudio

try:
    torchaudio._extension._load_lib("libtorchaudio_decoder")
    from .ctc_decoder import Hypothesis, KenLMLexiconDecoder, kenlm_lexicon_decoder
except ImportError as err:
    raise ImportError(
        "flashlight decoder bindings are required to use this functionality. "
        "Please set BUILD_CTC_DECODER=1 when building from source."
    ) from err


__all__ = [
    "Hypothesis",
    "KenLMLexiconDecoder",
    "kenlm_lexicon_decoder",
]
