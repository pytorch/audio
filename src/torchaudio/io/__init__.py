from torio.io import CodecConfig as _CodecConfig, StreamingMediaDecoder as _StreamReader, StreamingMediaEncoder as _StreamWriter
from torchaudio._internal.module_utils import dropping_class_io_support, dropping_class_support, dropping_io_support

from ._effector import AudioEffector as _AudioEffector
from ._playback import play_audio as _play_audio

CodecConfig = dropping_class_io_support(_CodecConfig)
StreamReader = dropping_class_io_support(_StreamReader)
StreamWriter = dropping_class_io_support(_StreamWriter)
AudioEffector = dropping_class_support(_AudioEffector)
play_audio = dropping_io_support(_play_audio)


__all__ = [
    "AudioEffector",
    "StreamReader",
    "StreamWriter",
    "CodecConfig",
    "play_audio",
]
