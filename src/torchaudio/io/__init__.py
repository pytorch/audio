from torio.io import CodecConfig, StreamingMediaDecoder as StreamReader, StreamingMediaEncoder as StreamWriter
from torchaudio._internal.module_utils import dropping_support

from ._effector import AudioEffector
from ._playback import play_audio as _play_audio

CodecConfig.__init__ = dropping_support(CodecConfig.__init__)
StreamReader.__init__ = dropping_support(StreamReader.__init__)
StreamWriter.__init__ = dropping_support(StreamWriter.__init__)
AudioEffector.__init__ = dropping_support(AudioEffector.__init__)
play_audio = dropping_support(_play_audio)


__all__ = [
    "AudioEffector",
    "StreamReader",
    "StreamWriter",
    "CodecConfig",
    "play_audio",
]
