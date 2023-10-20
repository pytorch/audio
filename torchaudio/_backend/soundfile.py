import os
from typing import BinaryIO, Optional, Tuple, Union

import torch
from torchaudio.io import CodecConfig

from . import soundfile_backend
from .backend import Backend
from .common import AudioMetaData


class SoundfileBackend(Backend):
    @staticmethod
    def info(uri: Union[BinaryIO, str, os.PathLike], format: Optional[str], buffer_size: int = 4096) -> AudioMetaData:
        return soundfile_backend.info(uri, format)

    @staticmethod
    def load(
        uri: Union[BinaryIO, str, os.PathLike],
        frame_offset: int = 0,
        num_frames: int = -1,
        normalize: bool = True,
        channels_first: bool = True,
        format: Optional[str] = None,
        buffer_size: int = 4096,
    ) -> Tuple[torch.Tensor, int]:
        return soundfile_backend.load(uri, frame_offset, num_frames, normalize, channels_first, format)

    @staticmethod
    def save(
        uri: Union[BinaryIO, str, os.PathLike],
        src: torch.Tensor,
        sample_rate: int,
        channels_first: bool = True,
        format: Optional[str] = None,
        encoding: Optional[str] = None,
        bits_per_sample: Optional[int] = None,
        buffer_size: int = 4096,
        compression: Optional[Union[CodecConfig, float, int]] = None,
    ) -> None:
        if compression:
            raise ValueError("soundfile backend does not support argument `compression`.")

        soundfile_backend.save(
            uri, src, sample_rate, channels_first, format=format, encoding=encoding, bits_per_sample=bits_per_sample
        )

    @staticmethod
    def can_decode(uri, format) -> bool:
        return True

    @staticmethod
    def can_encode(uri, format) -> bool:
        return True
