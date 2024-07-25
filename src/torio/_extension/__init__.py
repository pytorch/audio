from .utils import _init_ffmpeg, _LazyImporter


_FFMPEG_EXT = None


def lazy_import_ffmpeg_ext():
    """Load FFmpeg integration based on availability in lazy manner"""

    global _FFMPEG_EXT
    if _FFMPEG_EXT is None:
        _FFMPEG_EXT = _LazyImporter("_torio_ffmpeg", _init_ffmpeg)
    return _FFMPEG_EXT
