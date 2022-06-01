_INITIALIZED = False
_LAZILY_IMPORTED = [
    "CTCHypothesis",
    "CTCDecoder",
    "ctc_decoder",
    "download_pretrained_files",
]


def _init_extension():
    import torchaudio

    torchaudio._extension._load_lib("libtorchaudio_decoder")

    global _INITIALIZED
    _INITIALIZED = True


def __getattr__(name: str):
    if name in _LAZILY_IMPORTED:
        if not _INITIALIZED:
            _init_extension()

        try:
            from . import _ctc_decoder
        except AttributeError as err:
            raise RuntimeError(
                "CTC decoder requires the decoder extension. Please set BUILD_CTC_DECODER=1 when building from source."
            ) from err

        item = getattr(_ctc_decoder, name)
        globals()[name] = item
        return item
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return sorted(__all__ + _LAZILY_IMPORTED)


__all__ = []
