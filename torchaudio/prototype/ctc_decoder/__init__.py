import torchaudio

try:
    torchaudio._extension._load_lib("libtorchaudio_decoder")
    from .ctc_decoder import Hypothesis, CTCDecoder, ctc_decoder, lexicon_decoder, download_pretrained_files
except ImportError as err:
    raise ImportError(
        "flashlight decoder bindings are required to use this functionality. "
        "Please set BUILD_CTC_DECODER=1 when building from source."
    ) from err


__all__ = [
    "Hypothesis",
    "CTCDecoder",
    "ctc_decoder",
    "lexicon_decoder",
    "download_pretrained_files",
]
