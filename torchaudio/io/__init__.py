import torchaudio

_LAZILY_IMPORTED = [
    "StreamReader",
    "StreamReaderSourceStream",
    "StreamReaderSourceAudioStream",
    "StreamReaderSourceVideoStream",
    "StreamReaderOutputStream",
]


def __getattr__(name: str):
    if name in _LAZILY_IMPORTED:

        torchaudio._extension._init_ffmpeg()

        from . import _stream_reader

        item = getattr(_stream_reader, name)
        globals()[name] = item
        return item
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return sorted(__all__ + _LAZILY_IMPORTED)


__all__ = []
