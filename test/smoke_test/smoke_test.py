"""Run smoke tests"""
import os
import sys


if os.name == "nt":
    if sys.version_info[:2] >= (3, 8) and "FFMPEG_ROOT" in os.environ:
        # FFMPEG_ROOT is expected to have the dlls like avcodec-58.dll
        os.add_dll_directory(os.environ["FFMPEG_ROOT"])


import torchaudio  # noqa: F401
import torchaudio.compliance.kaldi  # noqa: F401
import torchaudio.datasets  # noqa: F401
import torchaudio.functional  # noqa: F401
import torchaudio.models  # noqa: F401
import torchaudio.pipelines  # noqa: F401
import torchaudio.sox_effects  # noqa: F401
import torchaudio.transforms  # noqa: F401
import torchaudio.utils  # noqa: F401
from torchaudio.io import StreamReader  # noqa: F401
