def __getattr__(item):
    from torio.utils import ffmpeg_utils

    return getattr(ffmpeg_utils, item)
