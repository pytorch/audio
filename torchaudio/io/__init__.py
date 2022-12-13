import torchaudio

_STREAM_READER = [
    "StreamReader",
    "StreamReaderSourceStream",
    "StreamReaderSourceAudioStream",
    "StreamReaderSourceVideoStream",
    "StreamReaderOutputStream",
]

_STREAM_WRITER = [
    "StreamWriter",
]


_LAZILY_IMPORTED = _STREAM_READER + _STREAM_WRITER


def __getattr__(name: str):
    if name in _LAZILY_IMPORTED:
        torchaudio._extension._init_ffmpeg()

        if name in _STREAM_READER:
            from . import _stream_reader

            item = getattr(_stream_reader, name)

        else:
            from . import _stream_writer

            item = getattr(_stream_writer, name)

        globals()[name] = item
        return item
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return sorted(__all__ + _LAZILY_IMPORTED)


__all__ = []
