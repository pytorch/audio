_INITIALIZED = False
_LAZILY_IMPORTED = [
    "StreamReader",
    "StreamReaderSourceStream",
    "StreamReaderSourceAudioStream",
    "StreamReaderSourceVideoStream",
    "StreamReaderOutputStream",
]


def _init_extension():
    import torch
    import torchaudio

    try:
        torchaudio._extension._load_lib("libtorchaudio_ffmpeg")
    except OSError as err:
        raise ImportError(
            "Stream API requires FFmpeg libraries (libavformat and such). Please install FFmpeg 4."
        ) from err
    try:
        torch.ops.torchaudio.ffmpeg_init()
    except RuntimeError as err:
        raise RuntimeError(
            "Stream API requires FFmpeg binding. Please set USE_FFMPEG=1 when building from source."
        ) from err

    global _INITIALIZED
    _INITIALIZED = True


def __getattr__(name: str):
    if name in _LAZILY_IMPORTED:
        if not _INITIALIZED:
            _init_extension()

        from . import _stream_reader

        item = getattr(_stream_reader, name)
        globals()[name] = item
        return item
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return sorted(__all__ + _LAZILY_IMPORTED)


__all__ = []
