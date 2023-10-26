"""Module to change the configuration of FFmpeg libraries (such as libavformat).

It affects functionalities in :py:mod:`torchaudio.io` (and indirectly :py:func:`torchaudio.load`).
"""


# This file is just for BC.
def __getattr__(item):
    from torio.utils import ffmpeg_utils

    return getattr(ffmpeg_utils, item)
