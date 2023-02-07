import torchaudio

_STREAM_READER = [
    "StreamReader",
]

_STREAM_WRITER = [
    "StreamWriter",
]

_PLAYBACK = [
    "play_audio",
]


_LAZILY_IMPORTED = _STREAM_READER + _STREAM_WRITER + _PLAYBACK


def __getattr__(name: str):
    if name in _LAZILY_IMPORTED:
        if not torchaudio._extension._FFMPEG_INITIALIZED:
            torchaudio._extension._init_ffmpeg()

        if name in _STREAM_READER:
            from . import _stream_reader

            item = getattr(_stream_reader, name)

        elif name in _STREAM_WRITER:
            from . import _stream_writer

            item = getattr(_stream_writer, name)

        elif name in _PLAYBACK:
            from . import _playback

            item = getattr(_playback, name)

        globals()[name] = item
        return item
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return sorted(__all__ + _LAZILY_IMPORTED)


__all__ = []
