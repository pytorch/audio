_INITIALIZED = False
_LAZILY_IMPORTED = [
    "CTCHypothesis",
    "CTCDecoder",
    "CTCDecoderLM",
    "CTCDecoderLMState",
    "ctc_decoder",
    "download_pretrained_files",
]
_LAZILY_IMPORTED_CUCTC = [
    "CUCTCDecoder",
    "CUCTCHypothesis",
    "cuda_ctc_decoder",
]

def __getattr__(name: str):
    if name in _LAZILY_IMPORTED:
        try:
            from . import _ctc_decoder
        except AttributeError as err:
            raise RuntimeError(
                "CTC decoder requires the decoder extension. Please set BUILD_CTC_DECODER=1 when building from source."
            ) from err

        item = getattr(_ctc_decoder, name)
        globals()[name] = item
        return item
    elif name in _LAZILY_IMPORTED_CUCTC:
        try:
            from . import _cuda_ctc_decoder
        except AttributeError as err:
            raise RuntimeError(
                "CUCTC decoder requires the decoder extension. Please set BUILD_CUDA_CTC_DECODER=1 when building from source."
            ) from err

        item = getattr(_cuda_ctc_decoder, name)
        globals()[name] = item
        return item
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return sorted(__all__ + _LAZILY_IMPORTED + _LAZILY_IMPORTED_CUCTC)


__all__ = []
